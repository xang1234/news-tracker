"""Internal intelligence layer metadata and review endpoints.

Infrastructure-oriented routes for:
    - Lane run metadata lookup
    - Manifest and pointer inspection
    - Published object review submission
    - Contract version compatibility checks

These endpoints are producer-owned integration targets for downstream
consumers (e.g., stock-screener). They do not serve end-user product UX.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.api.auth import verify_api_key
from src.api.dependencies import get_publish_service
from src.api.models import ErrorResponse
from src.contracts.intelligence.db_schemas import VALID_RUN_STATUSES
from src.contracts.intelligence.lanes import ALL_LANES, VALID_LANES
from src.contracts.intelligence.ownership import (
    OwnershipPolicy,
    check_compatibility,
)
from src.contracts.intelligence.version import ContractRegistry
from src.publish.service import PublishService

logger = structlog.get_logger(__name__)
router = APIRouter()


# -- Response models -------------------------------------------------------


class LaneRunResponse(BaseModel):
    """Serialized lane run metadata."""

    run_id: str
    lane: str
    status: str
    contract_version: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ManifestResponse(BaseModel):
    """Serialized manifest metadata."""

    manifest_id: str
    lane: str
    run_id: str
    contract_version: str
    published_at: datetime | None = None
    object_count: int
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class PointerResponse(BaseModel):
    """Current manifest pointer for a lane."""

    lane: str
    manifest_id: str
    activated_at: datetime
    previous_manifest_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PublishedObjectResponse(BaseModel):
    """Serialized published object."""

    object_id: str
    object_type: str
    manifest_id: str
    lane: str
    publish_state: str
    contract_version: str
    source_ids: list[str] = Field(default_factory=list)
    run_id: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ReviewRequest(BaseModel):
    """Request to transition a published object's state."""

    target_state: str = Field(
        ...,
        description="Target publish state (review, published, retracted, draft)",
    )


class ReviewResponse(BaseModel):
    """Response after a state transition."""

    object: PublishedObjectResponse
    previous_state: str
    new_state: str


class CompatibilityResponse(BaseModel):
    """Result of a contract compatibility check."""

    compatible: bool
    current_version: str
    checked_version: str
    message: str


class ContractInfoResponse(BaseModel):
    """Current contract metadata."""

    current_version: str
    minimum_supported: str
    lanes: list[str]
    publishable_object_types: list[str]


# -- Helper ----------------------------------------------------------------


def _run_to_response(run) -> LaneRunResponse:
    return LaneRunResponse(
        run_id=run.run_id,
        lane=run.lane,
        status=run.status,
        contract_version=run.contract_version,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error_message=run.error_message,
        config_snapshot=run.config_snapshot,
        metrics=run.metrics,
        metadata=run.metadata,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _object_to_response(obj) -> PublishedObjectResponse:
    return PublishedObjectResponse(
        object_id=obj.object_id,
        object_type=obj.object_type,
        manifest_id=obj.manifest_id,
        lane=obj.lane,
        publish_state=obj.publish_state,
        contract_version=obj.contract_version,
        source_ids=obj.source_ids,
        run_id=obj.run_id,
        valid_from=obj.valid_from,
        valid_to=obj.valid_to,
        payload=obj.payload,
        lineage=obj.lineage,
        created_at=obj.created_at,
        updated_at=obj.updated_at,
    )


# -- Contract info endpoint ------------------------------------------------


@router.get(
    "/intel/contract",
    response_model=ContractInfoResponse,
    summary="Get contract metadata",
    description="Returns current contract version, supported lanes, and publishable types.",
)
async def get_contract_info() -> ContractInfoResponse:
    return ContractInfoResponse(
        current_version=str(ContractRegistry.CURRENT),
        minimum_supported=str(ContractRegistry.MINIMUM_SUPPORTED),
        lanes=list(ALL_LANES),
        publishable_object_types=sorted(OwnershipPolicy.PUBLISHABLE_OBJECT_TYPES),
    )


@router.get(
    "/intel/contract/compatibility",
    response_model=CompatibilityResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid version string"},
    },
    summary="Check contract compatibility",
    description="Check whether a contract version is compatible with the current contract.",
)
async def check_contract_compatibility(
    version: str = Query(..., description="Contract version to check (e.g., '0.1.0')"),
) -> CompatibilityResponse:
    try:
        result = check_compatibility(version)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    return CompatibilityResponse(
        compatible=result.compatible,
        current_version=str(result.current),
        checked_version=str(result.checked),
        message=result.message,
    )


# -- Lane run metadata endpoints -------------------------------------------


@router.get(
    "/intel/runs/{run_id}",
    response_model=LaneRunResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
    summary="Get lane run metadata",
    description="Fetch metadata for a specific lane run.",
)
async def get_run(
    run_id: str,
    api_key: str = Depends(verify_api_key),  # noqa: B008
    service: PublishService = Depends(get_publish_service),  # noqa: B008
) -> LaneRunResponse:
    run = await service.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lane run not found: {run_id}",
        )
    return _run_to_response(run)


@router.get(
    "/intel/runs",
    response_model=list[LaneRunResponse],
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Invalid filter parameter"},
    },
    summary="List lane runs",
    description="List lane runs with optional lane and status filters.",
)
async def list_runs(
    lane: str | None = Query(default=None, description="Filter by lane"),
    run_status: str | None = Query(default=None, alias="status", description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    api_key: str = Depends(verify_api_key),  # noqa: B008
    service: PublishService = Depends(get_publish_service),  # noqa: B008
) -> list[LaneRunResponse]:
    if lane is not None and lane not in VALID_LANES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid lane {lane!r}. Must be one of {sorted(VALID_LANES)}",
        )
    if run_status is not None and run_status not in VALID_RUN_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid status {run_status!r}. Must be one of {sorted(VALID_RUN_STATUSES)}",
        )
    runs = await service.list_runs(lane=lane, status=run_status, limit=limit)
    return [_run_to_response(r) for r in runs]


# -- Manifest endpoints ----------------------------------------------------


@router.get(
    "/intel/manifests/{manifest_id}",
    response_model=ManifestResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Manifest not found"},
    },
    summary="Get manifest metadata",
    description="Fetch metadata for a specific manifest.",
)
async def get_manifest(
    manifest_id: str,
    api_key: str = Depends(verify_api_key),  # noqa: B008
    service: PublishService = Depends(get_publish_service),  # noqa: B008
) -> ManifestResponse:
    m = await service.get_manifest(manifest_id)
    if m is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Manifest not found: {manifest_id}",
        )
    return ManifestResponse(
        manifest_id=m.manifest_id,
        lane=m.lane,
        run_id=m.run_id,
        contract_version=m.contract_version,
        published_at=m.published_at,
        object_count=m.object_count,
        checksum=m.checksum,
        metadata=m.metadata,
        created_at=m.created_at,
    )


# -- Pointer endpoints -----------------------------------------------------


@router.get(
    "/intel/pointers/{lane}",
    response_model=PointerResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "No pointer set for lane"},
        422: {"model": ErrorResponse, "description": "Invalid lane"},
    },
    summary="Get current manifest pointer",
    description="Get the current serving manifest pointer for a lane.",
)
async def get_pointer(
    lane: str,
    api_key: str = Depends(verify_api_key),  # noqa: B008
    service: PublishService = Depends(get_publish_service),  # noqa: B008
) -> PointerResponse:
    if lane not in VALID_LANES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid lane {lane!r}. Must be one of {sorted(VALID_LANES)}",
        )
    ptr = await service.get_pointer(lane)
    if ptr is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No manifest pointer set for lane: {lane}",
        )
    return PointerResponse(
        lane=ptr.lane,
        manifest_id=ptr.manifest_id,
        activated_at=ptr.activated_at,
        previous_manifest_id=ptr.previous_manifest_id,
        metadata=ptr.metadata,
    )


# -- Published object review endpoints -------------------------------------


@router.get(
    "/intel/objects/{object_id}",
    response_model=PublishedObjectResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Object not found"},
    },
    summary="Get published object",
    description="Fetch a published object by ID.",
)
async def get_object(
    object_id: str,
    api_key: str = Depends(verify_api_key),  # noqa: B008
    service: PublishService = Depends(get_publish_service),  # noqa: B008
) -> PublishedObjectResponse:
    obj = await service.get_object(object_id)
    if obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Published object not found: {object_id}",
        )
    return _object_to_response(obj)


@router.post(
    "/intel/objects/{object_id}/review",
    response_model=ReviewResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Object not found"},
        422: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
    summary="Submit review decision",
    description=(
        "Transition a published object's state. Valid transitions: "
        "draft→review, draft→published, review→published, review→draft, "
        "published→retracted."
    ),
)
async def submit_review(
    object_id: str,
    body: ReviewRequest,
    api_key: str = Depends(verify_api_key),  # noqa: B008
    service: PublishService = Depends(get_publish_service),  # noqa: B008
) -> ReviewResponse:
    try:
        previous_state, updated = await service.transition_object(object_id, body.target_state)
    except ValueError as e:
        detail = str(e)
        code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in detail.lower()
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=code, detail=detail) from e
    return ReviewResponse(
        object=_object_to_response(updated),
        previous_state=previous_state,
        new_state=updated.publish_state,
    )
