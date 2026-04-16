"""Sources Admin endpoints — CRUD for the sources table."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from redis.asyncio import Redis
from starlette.requests import Request

from src.api.admin_models import (
    BulkCreateSourcesRequest,
    BulkCreateSourcesResponse,
    CreateSourceRequest,
    SourceItem,
    SourcesListResponse,
    TriggerIngestionResponse,
    UpdateSourceRequest,
)
from src.api.auth import verify_api_key
from src.api.dependencies import get_redis_client, get_sources_repository
from src.api.models import ErrorResponse
from src.api.rate_limit import limiter
from src.config.settings import get_settings as _get_settings
from src.sources.repository import SourcesRepository
from src.sources.schemas import Source

if TYPE_CHECKING:
    from src.services.ingestion_service import IngestionService

logger = structlog.get_logger(__name__)
router = APIRouter()

_INGESTION_LOCK_KEY = "sources:trigger-ingestion:lock"
_RELEASE_LOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


async def _release_ingestion_lock(redis_client: Redis, owner_token: str) -> None:
    await redis_client.execute_command(
        "EVAL",
        _RELEASE_LOCK_SCRIPT,
        1,
        _INGESTION_LOCK_KEY,
        owner_token,
    )


async def _build_ingestion_service(request: Request) -> IngestionService:
    from src.services.ingestion_service import IngestionService
    from src.sources.service import SourcesService

    settings = _get_settings()
    twitter_sources = None
    reddit_sources = None
    substack_sources = None

    if settings.sources_enabled:
        db = await request.app.state.services.get_database()
        svc = SourcesService(db)
        twitter_sources = await svc.get_twitter_sources()
        reddit_sources = await svc.get_reddit_sources()
        substack_sources = await svc.get_substack_sources()

    return IngestionService(
        twitter_sources=twitter_sources,
        reddit_sources=reddit_sources,
        substack_sources=substack_sources,
    )


def _require_sources_enabled() -> None:
    settings = _get_settings()
    if not settings.sources_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sources endpoints require sources_enabled=true",
        )


def _source_to_item(s: Source) -> SourceItem:
    return SourceItem(
        platform=s.platform,
        identifier=s.identifier,
        display_name=s.display_name,
        description=s.description,
        is_active=s.is_active,
        metadata=s.metadata,
        created_at=s.created_at.isoformat() if s.created_at else None,
        updated_at=s.updated_at.isoformat() if s.updated_at else None,
    )


def _normalize_identifier(platform: str, identifier: str) -> str:
    """Apply platform-specific normalizations."""
    if platform == "twitter":
        return identifier.lstrip("@")
    if platform == "reddit":
        return identifier.lower()
    return identifier


@router.get(
    "/sources",
    response_model=SourcesListResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="List sources with filters",
)
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def list_sources(
    request: Request,
    platform: str | None = Query(default=None, description="Filter by platform"),
    search: str | None = Query(default=None, description="Search identifier/display_name"),
    active_only: bool = Query(default=False, description="Only active sources"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    api_key: str = Depends(verify_api_key),
    repo: SourcesRepository = Depends(get_sources_repository),
) -> SourcesListResponse:
    _require_sources_enabled()
    start = time.perf_counter()

    try:
        sources, total = await repo.list_sources(
            platform=platform,
            search=search,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

        items = [_source_to_item(s) for s in sources]

        latency_ms = (time.perf_counter() - start) * 1000
        return SourcesListResponse(
            sources=items,
            total=total,
            has_more=(offset + limit) < total,
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("list_sources_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list sources")


@router.post(
    "/sources",
    response_model=SourceItem,
    status_code=status.HTTP_201_CREATED,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Create a new source",
)
@limiter.limit(lambda: _get_settings().rate_limit_admin)
async def create_source(
    request: Request,
    body: CreateSourceRequest,
    api_key: str = Depends(verify_api_key),
    repo: SourcesRepository = Depends(get_sources_repository),
) -> SourceItem:
    _require_sources_enabled()

    try:
        identifier = _normalize_identifier(body.platform, body.identifier)

        source = Source(
            platform=body.platform,
            identifier=identifier,
            display_name=body.display_name,
            description=body.description,
            metadata=body.metadata,
        )
        await repo.upsert(source)

        # Re-fetch to get timestamps
        created = await repo.get_by_key(source.platform, source.identifier)
        if not created:
            raise HTTPException(status_code=500, detail="Source creation failed")

        logger.info("Source created", platform=source.platform, identifier=source.identifier)
        return _source_to_item(created)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("create_source_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create source")


@router.post(
    "/sources/bulk",
    response_model=BulkCreateSourcesResponse,
    status_code=status.HTTP_200_OK,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Bulk-create sources for a single platform",
)
@limiter.limit(lambda: _get_settings().rate_limit_admin)
async def bulk_create_sources(
    request: Request,
    body: BulkCreateSourcesRequest,
    api_key: str = Depends(verify_api_key),
    repo: SourcesRepository = Depends(get_sources_repository),
) -> BulkCreateSourcesResponse:
    _require_sources_enabled()
    start = time.perf_counter()

    try:
        # Normalize & deduplicate identifiers
        seen: set[str] = set()
        unique_ids: list[str] = []
        for raw_id in body.identifiers:
            normalized = _normalize_identifier(body.platform, raw_id)
            if normalized not in seen:
                seen.add(normalized)
                unique_ids.append(normalized)

        sources = [Source(platform=body.platform, identifier=ident) for ident in unique_ids]

        created, total = await repo.bulk_create(sources)
        skipped = total - created

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Bulk created sources",
            platform=body.platform,
            created=created,
            skipped=skipped,
            total=total,
        )
        return BulkCreateSourcesResponse(
            created=created,
            skipped=skipped,
            total=total,
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("bulk_create_sources_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to bulk create sources")


@router.post(
    "/sources/trigger-ingestion",
    response_model=TriggerIngestionResponse,
    responses={
        401: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Trigger a manual ingestion cycle",
)
@limiter.limit(lambda: _get_settings().rate_limit_admin)
async def trigger_ingestion(
    request: Request,
    api_key: str = Depends(verify_api_key),
    redis_client: Redis = Depends(get_redis_client),
) -> TriggerIngestionResponse:
    _require_sources_enabled()

    settings = _get_settings()
    owner_token = uuid.uuid4().hex
    lock_acquired = await redis_client.set(
        _INGESTION_LOCK_KEY,
        owner_token,
        ex=settings.sources_trigger_lock_ttl_seconds,
        nx=True,
    )
    if not lock_acquired:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ingestion is already running",
        )

    try:
        service = await _build_ingestion_service(request)
    except Exception as exc:
        from src.services.ingestion_service import IngestionConfigurationError

        try:
            await _release_ingestion_lock(redis_client, owner_token)
        except Exception as release_error:
            logger.warning(
                "Failed to release ingestion lock after preflight failure",
                error=str(release_error),
                exc_info=True,
            )

        if isinstance(exc, IngestionConfigurationError):
            logger.warning("Manual ingestion misconfigured", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        logger.error("Manual ingestion preflight failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start ingestion") from exc

    async def _run_ingestion() -> None:
        try:
            results = await service.run_once()
            total = sum(results.values())
            logger.info(
                "Manual ingestion completed",
                results={p.value: c for p, c in results.items()},
                total=total,
            )
        except Exception as e:
            logger.error("Manual ingestion failed", error=str(e), exc_info=True)
        finally:
            try:
                await _release_ingestion_lock(redis_client, owner_token)
            except Exception as e:
                logger.warning("Failed to release ingestion lock", error=str(e), exc_info=True)

    task = asyncio.create_task(_run_ingestion())
    request.app.state.background_tasks.add(task)
    task.add_done_callback(request.app.state.background_tasks.discard)

    return TriggerIngestionResponse(
        status="started",
        message="Ingestion cycle started in background",
    )


@router.put(
    "/sources/{platform}/{identifier}",
    response_model=SourceItem,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Update a source",
)
@limiter.limit(lambda: _get_settings().rate_limit_admin)
async def update_source(
    request: Request,
    platform: str,
    identifier: str,
    body: UpdateSourceRequest,
    api_key: str = Depends(verify_api_key),
    repo: SourcesRepository = Depends(get_sources_repository),
) -> SourceItem:
    _require_sources_enabled()

    try:
        existing = await repo.get_by_key(platform, identifier)
        if not existing:
            raise HTTPException(
                status_code=404,
                detail=f"Source {platform}/{identifier} not found",
            )

        # Apply updates
        updated = Source(
            platform=existing.platform,
            identifier=existing.identifier,
            display_name=body.display_name
            if body.display_name is not None
            else existing.display_name,
            description=body.description if body.description is not None else existing.description,
            is_active=body.is_active if body.is_active is not None else existing.is_active,
            metadata=body.metadata if body.metadata is not None else existing.metadata,
        )
        await repo.upsert(updated)

        result = await repo.get_by_key(updated.platform, updated.identifier)
        if result is None:
            raise HTTPException(status_code=500, detail="Source update failed")
        logger.info("Source updated", platform=platform, identifier=identifier)
        return _source_to_item(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_source_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update source")


@router.delete(
    "/sources/{platform}/{identifier}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Deactivate a source (soft delete)",
)
@limiter.limit(lambda: _get_settings().rate_limit_admin)
async def deactivate_source(
    request: Request,
    platform: str,
    identifier: str,
    api_key: str = Depends(verify_api_key),
    repo: SourcesRepository = Depends(get_sources_repository),
) -> None:
    _require_sources_enabled()

    try:
        deactivated = await repo.deactivate(platform, identifier)
        if not deactivated:
            raise HTTPException(
                status_code=404,
                detail=f"Source {platform}/{identifier} not found or already inactive",
            )

        logger.info("Source deactivated", platform=platform, identifier=identifier)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("deactivate_source_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to deactivate source")
