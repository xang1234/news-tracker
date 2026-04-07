"""User-facing intelligence layer endpoints.

UX-oriented routes for the frontend to consume. The existing ``intel.py``
has infrastructure-oriented routes (run metadata, manifest pointers,
review transitions). This module exposes:

    - Aggregate lane health + quality scorecard
    - Assertion and claim browsing
    - Divergence alerts from published objects
    - Theme basket membership and path explanations
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.api.auth import verify_api_key
from src.api.dependencies import (
    get_assertion_repository,
    get_claim_repository,
    get_database,
    get_publish_service,
)
from src.api.models import ErrorResponse
from src.assertions.repository import AssertionRepository
from src.claims.repository import ClaimRepository
from src.contracts.intelligence.lanes import ALL_LANES
from src.publish.lane_health import compute_lane_health
from src.publish.service import PublishService
from src.storage.database import Database

logger = structlog.get_logger(__name__)
router = APIRouter()


# -- Response models --------------------------------------------------------


class LaneHealthItem(BaseModel):
    """Health summary for a single lane."""

    lane: str
    freshness: str  # FRESH, AGING, STALE, UNKNOWN
    quality: str  # HEALTHY, DEGRADED, CRITICAL, UNKNOWN
    quarantine: str  # CLEAR, QUARANTINED, WATCH
    readiness: str  # READY, WARN, BLOCKED
    last_completed_at: datetime | None = None


class QualityMetricItem(BaseModel):
    """A single quality metric result."""

    metric_type: str
    value: float
    severity: str
    message: str


class IntelHealthResponse(BaseModel):
    """Aggregate lane health + quality scorecard."""

    lanes: list[LaneHealthItem]
    quality_metrics: list[QualityMetricItem]
    overall_severity: str


class AssertionResponse(BaseModel):
    """Serialized resolved assertion."""

    assertion_id: str
    subject_concept_id: str
    predicate: str
    object_concept_id: str | None = None
    confidence: float
    status: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    support_count: int
    contradiction_count: int
    first_seen_at: datetime | None = None
    last_evidence_at: datetime | None = None
    source_diversity: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ClaimLinkItem(BaseModel):
    """A claim linked to an assertion, with claim summary."""

    assertion_id: str
    claim_id: str
    link_type: str
    contribution_weight: float
    claim: ClaimSummary | None = None


class ClaimSummary(BaseModel):
    """Lightweight claim details for inline display."""

    claim_id: str
    lane: str
    source_id: str
    source_type: str = ""
    subject_text: str
    predicate: str
    object_text: str | None = None
    confidence: float
    extraction_method: str = ""
    status: str
    created_at: datetime | None = None
    source_published_at: datetime | None = None


# Rebuild ClaimLinkItem to resolve forward ref
ClaimLinkItem.model_rebuild()


class AssertionDetailResponse(BaseModel):
    """Assertion with linked claims."""

    assertion: AssertionResponse
    claim_links: list[ClaimLinkItem]


class ClaimResponse(BaseModel):
    """Serialized evidence claim."""

    claim_id: str
    claim_key: str
    lane: str
    run_id: str | None = None
    source_id: str
    source_type: str
    source_text: str | None = None
    subject_text: str
    subject_concept_id: str | None = None
    predicate: str
    object_text: str | None = None
    object_concept_id: str | None = None
    confidence: float
    extraction_method: str
    claim_valid_from: datetime | None = None
    claim_valid_to: datetime | None = None
    source_published_at: datetime | None = None
    contract_version: str
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AssertionListResponse(BaseModel):
    """Wrapped assertion list with total count."""

    assertions: list[AssertionResponse]
    total: int
    latency_ms: float = 0


class ClaimListResponse(BaseModel):
    """Wrapped claim list with total count."""

    claims: list[ClaimResponse]
    total: int
    latency_ms: float = 0


class DivergenceItem(BaseModel):
    """A divergence alert with payload fields flattened for the frontend."""

    id: str
    issuer_concept_id: str = ""
    issuer_name: str = ""
    theme_concept_id: str = ""
    theme_name: str = ""
    reason: str = ""
    severity: str = ""
    title: str = ""
    summary: str = ""
    narrative_score: float | None = None
    filing_adoption_score: float | None = None
    created_at: datetime


class DivergenceListResponse(BaseModel):
    """Wrapped divergence list with severity counts."""

    divergences: list[DivergenceItem]
    total: int
    severity_counts: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0


class PublishedObjectItem(BaseModel):
    """Raw published object envelope for adoption/drift payloads."""

    object_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    lane: str


class IssuerDivergenceResponse(BaseModel):
    """All divergence-related published objects for an issuer."""

    issuer_id: str
    divergences: list[DivergenceItem]
    adoptions: list[PublishedObjectItem]
    drifts: list[PublishedObjectItem]


class BasketMember(BaseModel):
    """A flattened member of a theme basket."""

    concept_id: str = ""
    concept_name: str = ""
    role: str = ""
    best_score: float = 0.0
    best_sign: float = 0.0
    min_hops: int = 0
    path_count: int = 0
    has_mixed_signals: bool = False


class BasketResponse(BaseModel):
    """Wrapped basket response with members extracted from payloads."""

    theme_id: str
    members: list[BasketMember]
    latency_ms: float = 0


class BasketPathResponse(BaseModel):
    """Path explanation for a concept within a theme basket."""

    theme_id: str
    concept_id: str
    paths: list[dict[str, Any]]


# -- Helper functions -------------------------------------------------------


def _assertion_to_response(a) -> AssertionResponse:
    return AssertionResponse(
        assertion_id=a.assertion_id,
        subject_concept_id=a.subject_concept_id,
        predicate=a.predicate,
        object_concept_id=a.object_concept_id,
        confidence=a.confidence,
        status=a.status,
        valid_from=a.valid_from,
        valid_to=a.valid_to,
        support_count=a.support_count,
        contradiction_count=a.contradiction_count,
        first_seen_at=a.first_seen_at,
        last_evidence_at=a.last_evidence_at,
        source_diversity=a.source_diversity,
        metadata=a.metadata,
        created_at=a.created_at,
        updated_at=a.updated_at,
    )


def _claim_to_response(c) -> ClaimResponse:
    return ClaimResponse(
        claim_id=c.claim_id,
        claim_key=c.claim_key,
        lane=c.lane,
        run_id=c.run_id,
        source_id=c.source_id,
        source_type=c.source_type,
        source_text=c.source_text,
        subject_text=c.subject_text,
        subject_concept_id=c.subject_concept_id,
        predicate=c.predicate,
        object_text=c.object_text,
        object_concept_id=c.object_concept_id,
        confidence=c.confidence,
        extraction_method=c.extraction_method,
        claim_valid_from=c.claim_valid_from,
        claim_valid_to=c.claim_valid_to,
        source_published_at=c.source_published_at,
        contract_version=c.contract_version,
        status=c.status,
        metadata=c.metadata,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


def _claim_to_summary(c) -> ClaimSummary:
    return ClaimSummary(
        claim_id=c.claim_id,
        lane=c.lane,
        source_id=c.source_id,
        source_type=getattr(c, "source_type", ""),
        subject_text=c.subject_text,
        predicate=c.predicate,
        object_text=c.object_text,
        confidence=c.confidence,
        extraction_method=getattr(c, "extraction_method", ""),
        status=c.status,
        created_at=getattr(c, "created_at", None),
        source_published_at=c.source_published_at,
    )


def _parse_payload(value: Any) -> dict[str, Any]:
    """Parse a JSONB payload that may be str, dict, or None."""
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value)


def _row_to_divergence_item(row, payload: dict[str, Any]) -> DivergenceItem:
    """Flatten a published_objects row + parsed payload into a DivergenceItem."""
    return DivergenceItem(
        id=row["object_id"],
        issuer_concept_id=payload.get("issuer_concept_id", payload.get("issuer_id", "")),
        issuer_name=payload.get("issuer_name", ""),
        theme_concept_id=payload.get("theme_concept_id", payload.get("theme_id", "")),
        theme_name=payload.get("theme_name", ""),
        reason=payload.get("reason", payload.get("reason_code", "")),
        severity=payload.get("severity", ""),
        title=payload.get("title", ""),
        summary=payload.get("summary", ""),
        narrative_score=payload.get("narrative_score"),
        filing_adoption_score=payload.get("filing_adoption_score"),
        created_at=row["created_at"],
    )


def _row_to_published_object(row, payload: dict[str, Any]) -> PublishedObjectItem:
    return PublishedObjectItem(
        object_id=row["object_id"],
        payload=payload,
        created_at=row["created_at"],
        lane=row["lane"],
    )


async def _batch_fetch_claims(claim_repo: ClaimRepository, claim_ids: list[str]):
    """Fetch multiple claims, falling back to sequential if no batch method."""
    if not claim_ids:
        return []
    # Use batch method if available, else sequential
    if hasattr(claim_repo, "get_claims_by_ids"):
        return await claim_repo.get_claims_by_ids(claim_ids)
    import asyncio

    results = await asyncio.gather(*(claim_repo.get_claim(cid) for cid in claim_ids))
    return [c for c in results if c is not None]


# -- Endpoints --------------------------------------------------------------


@router.get(
    "/intel/health",
    response_model=IntelHealthResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
    },
    summary="Aggregate lane health and quality scorecard",
    description=(
        "Returns per-lane freshness, quality, quarantine, and readiness "
        "status, plus quality metrics when data is available."
    ),
)
async def get_intel_health(
    api_key: str = Depends(verify_api_key),  # noqa: B008
    service: PublishService = Depends(get_publish_service),  # noqa: B008
) -> IntelHealthResponse:
    lane_items: list[LaneHealthItem] = []

    for lane in ALL_LANES:
        # Get the latest completed run for this lane
        runs = await service.list_runs(lane=lane, status="completed", limit=1)
        last_completed_at = runs[0].completed_at if runs else None

        health = compute_lane_health(
            lane,
            last_completed_at=last_completed_at,
        )

        lane_items.append(
            LaneHealthItem(
                lane=health.lane,
                freshness=health.freshness.value.upper(),
                quality=health.quality.value.upper(),
                quarantine=health.quarantine.value.upper(),
                readiness=health.readiness.value.upper(),
                last_completed_at=health.last_completed_at,
            )
        )

    # Quality metrics require pre-aggregated DB counts that we do not
    # have aggregation queries for yet. Return empty until those are wired.
    quality_metrics: list[QualityMetricItem] = []

    # Overall severity: worst of lane readiness states
    severity_map = {"READY": 0, "WARN": 1, "BLOCKED": 2}
    readiness_to_severity = {"READY": "ok", "WARN": "warning", "BLOCKED": "critical"}
    worst = max(lane_items, key=lambda li: severity_map.get(li.readiness, 0))
    overall = readiness_to_severity.get(worst.readiness, "ok") if lane_items else "ok"

    return IntelHealthResponse(
        lanes=lane_items,
        quality_metrics=quality_metrics,
        overall_severity=overall,
    )


@router.get(
    "/intel/assertions",
    response_model=AssertionListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
    },
    summary="List assertions with filters",
    description="Browse resolved assertions with optional concept, predicate, and status filters.",
)
async def list_assertions(
    concept_id: str | None = Query(default=None, description="Filter by subject or object concept"),
    predicate: str | None = Query(default=None, description="Filter by predicate"),
    assertion_status: str | None = Query(
        default=None, alias="status", description="Filter by status"
    ),
    min_confidence: float | None = Query(
        default=None, ge=0.0, le=1.0, description="Minimum confidence"
    ),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Results offset"),
    api_key: str = Depends(verify_api_key),  # noqa: B008
    repo: AssertionRepository = Depends(get_assertion_repository),  # noqa: B008
) -> AssertionListResponse:
    # Overfetch to get accurate total (repos don't have count methods)
    fetch_limit = max(500, (limit + offset) * 2)
    if concept_id is not None:
        assertions = await repo.list_for_concept(concept_id, limit=fetch_limit)
        if predicate is not None:
            assertions = [a for a in assertions if a.predicate == predicate]
        if assertion_status is not None:
            assertions = [a for a in assertions if a.status == assertion_status]
        if min_confidence is not None:
            assertions = [a for a in assertions if a.confidence >= min_confidence]
    else:
        assertions = await repo.list_assertions(
            predicate=predicate,
            status=assertion_status,
            limit=fetch_limit,
        )
        if min_confidence is not None:
            assertions = [a for a in assertions if a.confidence >= min_confidence]

    total = len(assertions)
    page = assertions[offset : offset + limit]

    return AssertionListResponse(
        assertions=[_assertion_to_response(a) for a in page],
        total=total,
    )


@router.get(
    "/intel/assertions/{assertion_id}",
    response_model=AssertionDetailResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Assertion not found"},
    },
    summary="Get assertion with linked claims",
    description="Fetch a single assertion and its claim links with claim summaries.",
)
async def get_assertion_detail(
    assertion_id: str,
    api_key: str = Depends(verify_api_key),  # noqa: B008
    assertion_repo: AssertionRepository = Depends(get_assertion_repository),  # noqa: B008
    claim_repo: ClaimRepository = Depends(get_claim_repository),  # noqa: B008
) -> AssertionDetailResponse:
    assertion = await assertion_repo.get_assertion(assertion_id)
    if assertion is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assertion not found: {assertion_id}",
        )

    links = await assertion_repo.get_links_for_assertion(assertion_id)

    # Batch-fetch all linked claims to avoid N+1
    claim_ids = [link.claim_id for link in links]
    claims_by_id = {c.claim_id: c for c in await _batch_fetch_claims(claim_repo, claim_ids)}

    claim_link_items: list[ClaimLinkItem] = []
    for link in links:
        claim = claims_by_id.get(link.claim_id)
        summary = _claim_to_summary(claim) if claim else None
        claim_link_items.append(
            ClaimLinkItem(
                assertion_id=link.assertion_id,
                claim_id=link.claim_id,
                link_type=link.link_type,
                contribution_weight=link.contribution_weight,
                claim=summary,
            )
        )

    return AssertionDetailResponse(
        assertion=_assertion_to_response(assertion),
        claim_links=claim_link_items,
    )


@router.get(
    "/intel/claims",
    response_model=ClaimListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
    },
    summary="List claims with filters",
    description="Browse evidence claims with optional lane, source, and status filters.",
)
async def list_claims(
    assertion_id: str | None = Query(
        default=None,
        description="Get claims linked to this assertion",
    ),
    lane: str | None = Query(default=None, description="Filter by lane"),
    source_id: str | None = Query(default=None, description="Filter by source ID"),
    claim_status: str | None = Query(default=None, alias="status", description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Results offset"),
    api_key: str = Depends(verify_api_key),  # noqa: B008
    claim_repo: ClaimRepository = Depends(get_claim_repository),  # noqa: B008
    assertion_repo: AssertionRepository = Depends(get_assertion_repository),  # noqa: B008
) -> ClaimListResponse:
    if assertion_id is not None:
        links = await assertion_repo.get_links_for_assertion(assertion_id)
        claim_ids = [link.claim_id for link in links]
        claims = await _batch_fetch_claims(claim_repo, claim_ids)
        # Apply optional filters
        if lane is not None:
            claims = [c for c in claims if c.lane == lane]
        if source_id is not None:
            claims = [c for c in claims if c.source_id == source_id]
        if claim_status is not None:
            claims = [c for c in claims if c.status == claim_status]
    else:
        fetch_limit = max(500, (limit + offset) * 2)
        claims = await claim_repo.list_claims(
            lane=lane,
            source_id=source_id,
            status=claim_status,
            limit=fetch_limit,
        )

    total = len(claims)
    claims = claims[offset : offset + limit]

    return ClaimListResponse(
        claims=[_claim_to_response(c) for c in claims],
        total=total,
    )


@router.get(
    "/intel/divergence",
    response_model=DivergenceListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
    },
    summary="List divergence alerts",
    description=(
        "List published divergence objects with optional severity, "
        "reason, issuer, and theme filters."
    ),
)
async def list_divergence(
    severity: str | None = Query(default=None, description="Filter by severity"),
    reason_code: str | None = Query(default=None, description="Filter by reason code"),
    issuer: str | None = Query(default=None, description="Filter by issuer ID"),
    theme: str | None = Query(default=None, description="Filter by theme ID"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Results offset"),
    api_key: str = Depends(verify_api_key),  # noqa: B008
    db: Database = Depends(get_database),  # noqa: B008
) -> DivergenceListResponse:
    # Push JSONB filters into SQL to fix pagination
    conditions = ["object_type = $1", "publish_state = 'published'"]
    params: list[Any] = ["divergence"]
    idx = 2

    if severity is not None:
        conditions.append(f"payload->>'severity' = ${idx}")
        params.append(severity)
        idx += 1
    if reason_code is not None:
        conditions.append(f"payload->>'reason_code' = ${idx}")
        params.append(reason_code)
        idx += 1
    if issuer is not None:
        conditions.append(f"payload->>'issuer_id' = ${idx}")
        params.append(issuer)
        idx += 1
    if theme is not None:
        conditions.append(f"payload->>'theme_id' = ${idx}")
        params.append(theme)
        idx += 1

    where = " AND ".join(conditions)

    count_row = await db.fetchrow(
        f"SELECT count(*) AS cnt FROM intel_pub.published_objects WHERE {where}",
        *params,
    )
    total = count_row["cnt"] if count_row else 0

    params.extend([limit, offset])
    rows = await db.fetch(
        f"""
        SELECT * FROM intel_pub.published_objects
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    items = [_row_to_divergence_item(row, _parse_payload(row["payload"])) for row in rows]

    # Severity counts from full filtered set
    sev_rows = await db.fetch(
        f"""
        SELECT payload->>'severity' AS sev, count(*) AS cnt
        FROM intel_pub.published_objects
        WHERE {where}
        GROUP BY payload->>'severity'
        """,
        *params[: idx - 1],  # exclude limit/offset
    )
    severity_counts = {r["sev"]: r["cnt"] for r in sev_rows if r["sev"]}

    return DivergenceListResponse(
        divergences=items,
        total=total,
        severity_counts=severity_counts,
    )


@router.get(
    "/intel/divergence/{issuer_id}",
    response_model=IssuerDivergenceResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
    },
    summary="Issuer divergence detail",
    description=(
        "Get all divergence, adoption, and drift published objects for a specific issuer."
    ),
)
async def get_issuer_divergence(
    issuer_id: str,
    limit: int = Query(default=50, ge=1, le=200, description="Max results per type"),
    api_key: str = Depends(verify_api_key),  # noqa: B008
    db: Database = Depends(get_database),  # noqa: B008
) -> IssuerDivergenceResponse:
    rows = await db.fetch(
        """
        SELECT * FROM intel_pub.published_objects
        WHERE object_type IN ('divergence', 'adoption', 'drift')
          AND publish_state = 'published'
          AND payload->>'issuer_id' = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        issuer_id,
        limit * 3,
    )

    divergences: list[DivergenceItem] = []
    adoptions: list[PublishedObjectItem] = []
    drifts: list[PublishedObjectItem] = []

    for row in rows:
        payload = _parse_payload(row["payload"])
        obj_type = row["object_type"]
        if obj_type == "divergence" and len(divergences) < limit:
            divergences.append(_row_to_divergence_item(row, payload))
        elif obj_type == "adoption" and len(adoptions) < limit:
            adoptions.append(_row_to_published_object(row, payload))
        elif obj_type == "drift" and len(drifts) < limit:
            drifts.append(_row_to_published_object(row, payload))

    return IssuerDivergenceResponse(
        issuer_id=issuer_id,
        divergences=divergences,
        adoptions=adoptions,
        drifts=drifts,
    )


@router.get(
    "/intel/baskets/{theme_id}",
    response_model=BasketResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
    },
    summary="Basket members for a theme",
    description="List published basket objects belonging to a theme.",
)
async def get_basket_members(
    theme_id: str,
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    api_key: str = Depends(verify_api_key),  # noqa: B008
    db: Database = Depends(get_database),  # noqa: B008
) -> BasketResponse:
    rows = await db.fetch(
        """
        SELECT * FROM intel_pub.published_objects
        WHERE object_type = 'basket' AND publish_state = 'published'
          AND payload->>'theme_id' = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        theme_id,
        limit,
    )

    members: list[BasketMember] = []
    for row in rows:
        payload = _parse_payload(row["payload"])
        for m in payload.get("members", []):
            members.append(
                BasketMember(
                    concept_id=m.get("concept_id", ""),
                    concept_name=m.get("concept_name", ""),
                    role=m.get("role", ""),
                    best_score=m.get("best_score", 0.0),
                    best_sign=m.get("best_sign", 0.0),
                    min_hops=m.get("min_hops", 0),
                    path_count=m.get("path_count", 0),
                    has_mixed_signals=m.get("has_mixed_signals", False),
                )
            )

    return BasketResponse(theme_id=theme_id, members=members)


@router.get(
    "/intel/baskets/{theme_id}/paths/{concept_id}",
    response_model=BasketPathResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
    },
    summary="Path explanation for a concept in a basket",
    description=(
        "Get the path explanation for how a specific concept is connected within a theme's basket."
    ),
)
async def get_basket_path(
    theme_id: str,
    concept_id: str,
    api_key: str = Depends(verify_api_key),  # noqa: B008
    db: Database = Depends(get_database),  # noqa: B008
) -> BasketPathResponse:
    rows = await db.fetch(
        """
        SELECT * FROM intel_pub.published_objects
        WHERE object_type = 'basket' AND publish_state = 'published'
          AND payload->>'theme_id' = $1
        ORDER BY created_at DESC
        LIMIT 20
        """,
        theme_id,
    )

    paths: list[dict[str, Any]] = []
    for row in rows:
        payload = _parse_payload(row["payload"])

        # Extract paths for the requested concept from the payload
        members = payload.get("members", [])
        for member in members:
            if member.get("concept_id") == concept_id:
                member_paths = member.get("paths", [])
                paths.extend(member_paths)

        # Also check top-level paths structure
        payload_paths = payload.get("paths", {})
        if concept_id in payload_paths:
            concept_paths = payload_paths[concept_id]
            if isinstance(concept_paths, list):
                paths.extend(concept_paths)

    return BasketPathResponse(
        theme_id=theme_id,
        concept_id=concept_id,
        paths=paths,
    )
