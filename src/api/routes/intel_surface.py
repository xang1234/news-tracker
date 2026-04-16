"""User-facing intelligence layer endpoints.

UX-oriented routes for the frontend to consume. The existing ``intel.py``
has infrastructure-oriented routes (run metadata, manifest pointers,
review transitions). This module exposes:

    - Aggregate lane health + quality scorecard
    - Assertion and claim browsing from published objects
    - Divergence alerts from published objects
    - Theme basket membership and path explanations
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import structlog
from asyncpg.exceptions import UndefinedTableError
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.api.auth import verify_api_key
from src.api.dependencies import (
    get_database,
    get_publish_service,
)
from src.api.models import ErrorResponse
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


def _parse_dt(value: Any, fallback: datetime | None = None) -> datetime | None:
    """Parse ISO datetime-like payload fields defensively."""
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
    return fallback


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _confidence_expr() -> str:
    """SQL expression that safely parses confidence into float with 0.0 fallback."""
    return (
        "CASE "
        "WHEN jsonb_typeof(payload->'confidence') = 'number' THEN (payload->>'confidence')::double precision "
        "WHEN (payload->>'confidence') ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN (payload->>'confidence')::double precision "
        "ELSE 0.0 "
        "END"
    )


def _row_to_assertion_response(row, payload: dict[str, Any]) -> AssertionResponse | None:
    """Map a published assertion object to API response shape."""
    subject = _as_str(payload.get("subject_concept_id"))
    predicate = _as_str(payload.get("predicate"))
    if not subject or not predicate:
        return None

    return AssertionResponse(
        assertion_id=_as_str(payload.get("assertion_id"), default=_as_str(row["object_id"])),
        subject_concept_id=subject,
        predicate=predicate,
        object_concept_id=payload.get("object_concept_id"),
        confidence=_as_float(payload.get("confidence"), default=0.0),
        status=_as_str(payload.get("status"), default="active"),
        valid_from=_parse_dt(payload.get("valid_from"), fallback=row.get("valid_from")),
        valid_to=_parse_dt(payload.get("valid_to"), fallback=row.get("valid_to")),
        support_count=_as_int(payload.get("support_count"), default=0),
        contradiction_count=_as_int(payload.get("contradiction_count"), default=0),
        first_seen_at=_parse_dt(payload.get("first_seen_at")),
        last_evidence_at=_parse_dt(payload.get("last_evidence_at")),
        source_diversity=_as_int(payload.get("source_diversity"), default=0),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        created_at=_parse_dt(payload.get("created_at"), fallback=row["created_at"]) or row["created_at"],
        updated_at=_parse_dt(payload.get("updated_at"), fallback=row["updated_at"]) or row["updated_at"],
    )


def _row_to_claim_response(row, payload: dict[str, Any]) -> ClaimResponse | None:
    """Map a published claim object to API response shape."""
    subject_text = _as_str(payload.get("subject_text"))
    predicate = _as_str(payload.get("predicate"))
    source_id = _as_str(payload.get("source_id"))
    if not subject_text or not predicate or not source_id:
        return None

    claim_id = _as_str(payload.get("claim_id"), default=_as_str(row["object_id"]))
    return ClaimResponse(
        claim_id=claim_id,
        claim_key=_as_str(payload.get("claim_key"), default=claim_id),
        lane=_as_str(row["lane"]),
        run_id=row.get("run_id"),
        source_id=source_id,
        source_type=_as_str(payload.get("source_type")),
        source_text=payload.get("source_text"),
        subject_text=subject_text,
        subject_concept_id=payload.get("subject_concept_id"),
        predicate=predicate,
        object_text=payload.get("object_text"),
        object_concept_id=payload.get("object_concept_id"),
        confidence=_as_float(payload.get("confidence"), default=0.0),
        extraction_method=_as_str(payload.get("extraction_method")),
        claim_valid_from=_parse_dt(payload.get("claim_valid_from"), fallback=row.get("valid_from")),
        claim_valid_to=_parse_dt(payload.get("claim_valid_to"), fallback=row.get("valid_to")),
        source_published_at=_parse_dt(payload.get("source_published_at")),
        contract_version=_as_str(
            payload.get("contract_version"),
            default=_as_str(row["contract_version"]),
        ),
        status=_as_str(payload.get("status"), default="active"),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        created_at=_parse_dt(payload.get("created_at"), fallback=row["created_at"]),
        updated_at=_parse_dt(payload.get("updated_at"), fallback=row["updated_at"]),
    )


def _claim_response_to_summary(claim: ClaimResponse) -> ClaimSummary:
    return ClaimSummary(
        claim_id=claim.claim_id,
        lane=claim.lane,
        source_id=claim.source_id,
        source_type=claim.source_type,
        subject_text=claim.subject_text,
        predicate=claim.predicate,
        object_text=claim.object_text,
        confidence=claim.confidence,
        extraction_method=claim.extraction_method,
        status=claim.status,
        created_at=claim.created_at,
        source_published_at=claim.source_published_at,
    )


def _extract_assertion_claim_links(payload: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Read optional embedded claim links from an assertion payload."""
    claim_links = payload.get("claim_links")
    if not isinstance(claim_links, list):
        return []
    links: list[tuple[str, str, float]] = []
    for link in claim_links:
        if not isinstance(link, dict):
            continue
        claim_id = _as_str(link.get("claim_id"))
        if not claim_id:
            continue
        link_type = _as_str(link.get("link_type"), default="support")
        contribution_weight = _as_float(link.get("contribution_weight"), default=1.0)
        links.append((claim_id, link_type, contribution_weight))
    return links


def _extract_assertion_claim_ids(payload: dict[str, Any]) -> list[str]:
    """Extract optional claim id references from an assertion payload."""
    ids: list[str] = []
    raw_ids = payload.get("claim_ids")
    if isinstance(raw_ids, list):
        ids.extend(_as_str(i) for i in raw_ids if _as_str(i))
    ids.extend(link[0] for link in _extract_assertion_claim_links(payload))
    # preserve order while de-duplicating
    return list(dict.fromkeys(ids))


def _claim_payload_references_assertion(
    payload: dict[str, Any],
    assertion_id: str,
    embedded_claim_ids: set[str] | None = None,
    object_id: str | None = None,
) -> bool:
    """Best-effort matcher for claim→assertion linkage in published payloads."""
    if payload.get("assertion_id") == assertion_id:
        return True
    if payload.get("linked_assertion_id") == assertion_id:
        return True
    assertion_ids = payload.get("assertion_ids")
    if isinstance(assertion_ids, list) and assertion_id in assertion_ids:
        return True
    links = payload.get("links")
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict) and link.get("assertion_id") == assertion_id:
                return True
    if embedded_claim_ids:
        claim_id = _as_str(payload.get("claim_id"), default=object_id or "")
        if claim_id and claim_id in embedded_claim_ids:
            return True
    return False


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
    _empty = IntelHealthResponse(lanes=[], quality_metrics=[], overall_severity="ok")
    try:
        lane_items: list[LaneHealthItem] = []

        for lane in ALL_LANES:
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
    except UndefinedTableError:
        logger.warning("intel_schema_missing", endpoint="health")
        return _empty

    quality_metrics: list[QualityMetricItem] = []

    severity_map = {"READY": 0, "WARN": 1, "BLOCKED": 2}
    readiness_to_severity = {"READY": "ok", "WARN": "warning", "BLOCKED": "critical"}
    if lane_items:
        worst = max(lane_items, key=lambda li: severity_map.get(li.readiness, 0))
        overall = readiness_to_severity.get(worst.readiness, "ok")
    else:
        overall = "ok"

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
    db: Database = Depends(get_database),  # noqa: B008
) -> AssertionListResponse:
    try:
        return await _list_assertions_impl(
            concept_id, predicate, assertion_status, min_confidence, limit, offset, db
        )
    except UndefinedTableError:
        logger.warning("intel_schema_missing", endpoint="assertions")
        return AssertionListResponse(assertions=[], total=0)


async def _list_assertions_impl(
    concept_id, predicate, assertion_status, min_confidence, limit, offset, db
):
    conditions = ["object_type = $1", "publish_state = 'published'"]
    params: list[Any] = ["assertion"]
    idx = 2

    if concept_id is not None:
        conditions.append(
            f"(payload->>'subject_concept_id' = ${idx} OR payload->>'object_concept_id' = ${idx})"
        )
        params.append(concept_id)
        idx += 1
    if predicate is not None:
        conditions.append(f"payload->>'predicate' = ${idx}")
        params.append(predicate)
        idx += 1
    if assertion_status is not None:
        conditions.append(f"payload->>'status' = ${idx}")
        params.append(assertion_status)
        idx += 1
    if min_confidence is not None:
        conditions.append(f"{_confidence_expr()} >= ${idx}")
        params.append(min_confidence)
        idx += 1

    where = " AND ".join(conditions)
    count_row = await db.fetchrow(
        f"""
        WITH filtered AS (
            SELECT *,
                   COALESCE(NULLIF(payload->>'assertion_id', ''), object_id) AS assertion_dedupe_id
            FROM intel_pub.published_objects
            WHERE {where}
        ),
        latest AS (
            SELECT DISTINCT ON (assertion_dedupe_id) *
            FROM filtered
            ORDER BY assertion_dedupe_id, created_at DESC
        )
        SELECT count(*) AS cnt FROM latest
        """,
        *params,
    )
    total = int(count_row["cnt"]) if count_row is not None else 0

    limit_idx = len(params) + 1
    page_params = [*params, limit, offset]
    rows = await db.fetch(
        f"""
        WITH filtered AS (
            SELECT *,
                   COALESCE(NULLIF(payload->>'assertion_id', ''), object_id) AS assertion_dedupe_id
            FROM intel_pub.published_objects
            WHERE {where}
        ),
        latest AS (
            SELECT DISTINCT ON (assertion_dedupe_id) *
            FROM filtered
            ORDER BY assertion_dedupe_id, created_at DESC
        )
        SELECT * FROM latest
        ORDER BY created_at DESC
        LIMIT ${limit_idx} OFFSET ${limit_idx + 1}
        """,
        *page_params,
    )

    assertions: list[AssertionResponse] = []
    for row in rows:
        payload = _parse_payload(row["payload"])
        assertion = _row_to_assertion_response(row, payload)
        if assertion is None:
            continue
        assertions.append(assertion)
    return AssertionListResponse(assertions=assertions, total=total)


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
    db: Database = Depends(get_database),  # noqa: B008
) -> AssertionDetailResponse:
    try:
        row = await db.fetchrow(
            """
            SELECT *
            FROM intel_pub.published_objects
            WHERE object_type = 'assertion'
              AND publish_state = 'published'
              AND (object_id = $1 OR payload->>'assertion_id' = $1)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            assertion_id,
        )
    except UndefinedTableError:
        logger.warning("intel_schema_missing", endpoint="assertion_detail")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Intelligence schema not initialized"
        ) from None
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assertion not found: {assertion_id}",
        )
    payload = _parse_payload(row["payload"])
    assertion = _row_to_assertion_response(row, payload)
    if assertion is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Published assertion payload is invalid: {assertion_id}",
        )

    canonical_assertion_id = assertion.assertion_id
    embedded_links = _extract_assertion_claim_links(payload)
    embedded_ids = set(_extract_assertion_claim_ids(payload))

    claim_rows = await db.fetch(
        """
        WITH linked AS (
            SELECT *,
                   COALESCE(NULLIF(payload->>'claim_id', ''), object_id) AS claim_dedupe_id
            FROM intel_pub.published_objects
            WHERE object_type = 'claim'
              AND publish_state = 'published'
              AND (
                    payload->>'assertion_id' = $1
                    OR payload->>'linked_assertion_id' = $1
                    OR COALESCE(payload->'assertion_ids', '[]'::jsonb) ? $1
                    OR EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements(COALESCE(payload->'links', '[]'::jsonb)) AS link
                        WHERE link->>'assertion_id' = $1
                    )
                    OR COALESCE(NULLIF(payload->>'claim_id', ''), object_id) = ANY($2::text[])
                  )
        ),
        latest AS (
            SELECT DISTINCT ON (claim_dedupe_id) *
            FROM linked
            ORDER BY claim_dedupe_id, created_at DESC
        )
        SELECT * FROM latest
        ORDER BY created_at DESC
        """,
        canonical_assertion_id,
        list(embedded_ids),
    )
    claims_by_id: dict[str, ClaimResponse] = {}
    for claim_row in claim_rows:
        claim_payload = _parse_payload(claim_row["payload"])
        parsed_claim = _row_to_claim_response(claim_row, claim_payload)
        if parsed_claim is None:
            continue
        if parsed_claim.claim_id not in claims_by_id:
            claims_by_id[parsed_claim.claim_id] = parsed_claim

    claim_link_items: list[ClaimLinkItem] = []
    seen_claim_ids: set[str] = set()

    for claim_id, link_type, contribution_weight in embedded_links:
        claim = claims_by_id.get(claim_id)
        summary = _claim_response_to_summary(claim) if claim else None
        claim_link_items.append(
            ClaimLinkItem(
                assertion_id=canonical_assertion_id,
                claim_id=claim_id,
                link_type=link_type,
                contribution_weight=contribution_weight,
                claim=summary,
            )
        )
        seen_claim_ids.add(claim_id)

    # If embedded links are absent/incomplete, synthesize links from claim payloads.
    for claim_id, claim in claims_by_id.items():
        if claim_id in seen_claim_ids:
            continue
        claim_link_items.append(
            ClaimLinkItem(
                assertion_id=canonical_assertion_id,
                claim_id=claim_id,
                link_type="support",
                contribution_weight=1.0,
                claim=_claim_response_to_summary(claim),
            )
        )

    return AssertionDetailResponse(
        assertion=assertion,
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
    db: Database = Depends(get_database),  # noqa: B008
) -> ClaimListResponse:
    try:
        return await _list_claims_impl(
            assertion_id, lane, source_id, claim_status, limit, offset, db
        )
    except UndefinedTableError:
        logger.warning("intel_schema_missing", endpoint="claims")
        return ClaimListResponse(claims=[], total=0)


async def _list_claims_impl(
    assertion_id, lane, source_id, claim_status, limit, offset, db
):
    conditions = ["object_type = $1", "publish_state = 'published'"]
    params: list[Any] = ["claim"]
    idx = 2

    if lane is not None:
        conditions.append(f"lane = ${idx}")
        params.append(lane)
        idx += 1
    if source_id is not None:
        conditions.append(f"payload->>'source_id' = ${idx}")
        params.append(source_id)
        idx += 1
    if claim_status is not None:
        conditions.append(f"payload->>'status' = ${idx}")
        params.append(claim_status)
        idx += 1

    embedded_claim_ids: set[str] = set()
    if assertion_id is not None:
        assertion_row = await db.fetchrow(
            """
            SELECT payload
            FROM intel_pub.published_objects
            WHERE object_type = 'assertion'
              AND publish_state = 'published'
              AND (object_id = $1 OR payload->>'assertion_id' = $1)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            assertion_id,
        )
        if assertion_row is not None:
            assertion_payload = _parse_payload(assertion_row["payload"])
            embedded_claim_ids = set(_extract_assertion_claim_ids(assertion_payload))

    if assertion_id is not None:
        conditions.append(
            f"""(
                payload->>'assertion_id' = ${idx}
                OR payload->>'linked_assertion_id' = ${idx}
                OR COALESCE(payload->'assertion_ids', '[]'::jsonb) ? ${idx}
                OR EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements(COALESCE(payload->'links', '[]'::jsonb)) AS link
                    WHERE link->>'assertion_id' = ${idx}
                )
                OR COALESCE(NULLIF(payload->>'claim_id', ''), object_id) = ANY(${idx + 1}::text[])
            )"""
        )
        params.append(assertion_id)
        params.append(list(embedded_claim_ids))
        idx += 2

    where = " AND ".join(conditions)
    count_row = await db.fetchrow(
        f"""
        WITH filtered AS (
            SELECT *,
                   COALESCE(NULLIF(payload->>'claim_id', ''), object_id) AS claim_dedupe_id
            FROM intel_pub.published_objects
            WHERE {where}
        ),
        latest AS (
            SELECT DISTINCT ON (claim_dedupe_id) *
            FROM filtered
            ORDER BY claim_dedupe_id, created_at DESC
        )
        SELECT count(*) AS cnt FROM latest
        """,
        *params,
    )
    total = int(count_row["cnt"]) if count_row is not None else 0

    limit_idx = len(params) + 1
    page_params = [*params, limit, offset]
    rows = await db.fetch(
        f"""
        WITH filtered AS (
            SELECT *,
                   COALESCE(NULLIF(payload->>'claim_id', ''), object_id) AS claim_dedupe_id
            FROM intel_pub.published_objects
            WHERE {where}
        ),
        latest AS (
            SELECT DISTINCT ON (claim_dedupe_id) *
            FROM filtered
            ORDER BY claim_dedupe_id, created_at DESC
        )
        SELECT * FROM latest
        ORDER BY created_at DESC
        LIMIT ${limit_idx} OFFSET ${limit_idx + 1}
        """,
        *page_params,
    )

    claims: list[ClaimResponse] = []
    for row in rows:
        payload = _parse_payload(row["payload"])
        claim = _row_to_claim_response(row, payload)
        if claim is None:
            continue
        claims.append(claim)
    return ClaimListResponse(claims=claims, total=total)


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
    try:
        return await _list_divergence_impl(severity, reason_code, issuer, theme, limit, offset, db)
    except UndefinedTableError:
        logger.warning("intel_schema_missing", endpoint="divergence")
        return DivergenceListResponse(divergences=[], total=0, severity_counts={})


async def _list_divergence_impl(severity, reason_code, issuer, theme, limit, offset, db):
    conditions = ["object_type = $1", "publish_state = 'published'"]
    params: list[Any] = ["divergence"]
    idx = 2

    if severity is not None:
        conditions.append(f"payload->>'severity' = ${idx}")
        params.append(severity)
        idx += 1
    if reason_code is not None:
        conditions.append(f"payload->>'reason' = ${idx}")
        params.append(reason_code)
        idx += 1
    if issuer is not None:
        conditions.append(f"payload->>'issuer_concept_id' = ${idx}")
        params.append(issuer)
        idx += 1
    if theme is not None:
        conditions.append(f"payload->>'theme_concept_id' = ${idx}")
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
    try:
        rows = await db.fetch(
            """
            SELECT * FROM intel_pub.published_objects
            WHERE object_type IN ('divergence', 'adoption', 'drift')
              AND publish_state = 'published'
              AND payload->>'issuer_concept_id' = $1
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
    except UndefinedTableError:
        logger.warning("intel_schema_missing", endpoint="issuer_divergence")
        return IssuerDivergenceResponse(
            issuer_id=issuer_id, divergences=[], adoptions=[], drifts=[]
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
    try:
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
    except UndefinedTableError:
        logger.warning("intel_schema_missing", endpoint="baskets")
        return BasketResponse(theme_id=theme_id, members=[])

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
    try:
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
    except UndefinedTableError:
        logger.warning("intel_schema_missing", endpoint="basket_paths")
        return BasketPathResponse(theme_id=theme_id, concept_id=concept_id, paths=[])

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
