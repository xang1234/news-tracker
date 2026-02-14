"""Entity Explorer endpoints — browse, search, and manage extracted entities."""

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.requests import Request
import structlog

from src.api.auth import verify_api_key
from src.api.dependencies import get_document_repository, get_graph_repository
from src.api.rate_limit import limiter
from src.config.settings import get_settings as _get_settings
from src.api.models import (
    CooccurringEntityItem,
    CooccurrenceResponse,
    EntityDetailResponse,
    EntityListResponse,
    EntityMergeRequest,
    EntityMergeResponse,
    EntitySentimentResponse,
    EntityStatsResponse,
    EntitySummaryItem,
    ErrorResponse,
    ThemeDocumentItem,
    ThemeDocumentsResponse,
    TrendingEntitiesResponse,
    TrendingEntityItem,
)
from src.graph.storage import GraphRepository
from src.storage.repository import DocumentRepository

logger = structlog.get_logger(__name__)
router = APIRouter()

VALID_ENTITY_TYPES = {"COMPANY", "PRODUCT", "TECHNOLOGY", "TICKER", "METRIC"}


def _require_ner_enabled() -> None:
    settings = _get_settings()
    if not settings.ner_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entity endpoints require ner_enabled=true",
        )


# ── Static routes BEFORE parametric routes ────────────


@router.get(
    "/entities/stats",
    response_model=EntityStatsResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Entity statistics",
)
@limiter.limit(lambda: _get_settings().rate_limit_entities)
async def get_entity_stats(
    request: Request,
    api_key: str = Depends(verify_api_key),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> EntityStatsResponse:
    _require_ner_enabled()
    start = time.perf_counter()

    try:
        # Total distinct entities and by-type breakdown
        all_entities = await doc_repo.get_entity_counts(limit=10_000)
        by_type: dict[str, int] = {}
        for etype, _, count in all_entities:
            by_type[etype] = by_type.get(etype, 0) + count
        total_entities = len(all_entities)

        # Documents with entities
        docs_with = await doc_repo._db.fetchval(
            "SELECT COUNT(*) FROM documents WHERE jsonb_array_length(entities_mentioned) > 0"
        )

        latency_ms = (time.perf_counter() - start) * 1000
        return EntityStatsResponse(
            total_entities=total_entities,
            documents_with_entities=docs_with or 0,
            by_type=by_type,
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_entity_stats_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get entity stats")


@router.get(
    "/entities/trending",
    response_model=TrendingEntitiesResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Trending entities with mention spikes",
)
@limiter.limit(lambda: _get_settings().rate_limit_entities)
async def get_trending_entities(
    request: Request,
    hours_recent: int = Query(default=24, ge=1, le=168),
    hours_baseline: int = Query(default=168, ge=24, le=720),
    limit: int = Query(default=20, ge=1, le=100),
    api_key: str = Depends(verify_api_key),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> TrendingEntitiesResponse:
    _require_ner_enabled()
    start = time.perf_counter()

    try:
        rows = await doc_repo.get_trending_entities(
            hours_recent=hours_recent,
            hours_baseline=hours_baseline,
            limit=limit,
        )

        items = [
            TrendingEntityItem(
                type=r["type"],
                normalized=r["normalized"],
                recent_count=r["recent_count"],
                baseline_count=r["baseline_count"],
                spike_ratio=r["spike_ratio"],
            )
            for r in rows
        ]

        latency_ms = (time.perf_counter() - start) * 1000
        return TrendingEntitiesResponse(
            trending=items,
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_trending_entities_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get trending entities")


# ── Parametric entity routes ──────────────────────────


@router.get(
    "/entities",
    response_model=EntityListResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="List entities with search and filtering",
)
@limiter.limit(lambda: _get_settings().rate_limit_entities)
async def list_entities(
    request: Request,
    entity_type: str | None = Query(default=None, description="Filter by entity type"),
    search: str | None = Query(default=None, description="Search normalized name"),
    sort: str = Query(default="count", description="Sort by: count or recent"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    api_key: str = Depends(verify_api_key),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> EntityListResponse:
    _require_ner_enabled()
    start = time.perf_counter()

    try:
        if entity_type and entity_type not in VALID_ENTITY_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid entity_type {entity_type!r}. Must be one of: {sorted(VALID_ENTITY_TYPES)}",
            )

        entities, total = await doc_repo.list_entities(
            entity_type=entity_type,
            search=search,
            sort=sort,
            limit=limit,
            offset=offset,
        )

        items = [
            EntitySummaryItem(
                type=e["type"],
                normalized=e["normalized"],
                mention_count=e["mention_count"],
                first_seen=e["first_seen"].isoformat() if e["first_seen"] else None,
                last_seen=e["last_seen"].isoformat() if e["last_seen"] else None,
            )
            for e in entities
        ]

        latency_ms = (time.perf_counter() - start) * 1000
        return EntityListResponse(
            entities=items,
            total=total,
            has_more=(offset + limit) < total,
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("list_entities_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list entities")


@router.get(
    "/entities/{entity_type}/{normalized}",
    response_model=EntityDetailResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Entity detail with stats and platform breakdown",
)
@limiter.limit(lambda: _get_settings().rate_limit_entities)
async def get_entity_detail(
    request: Request,
    entity_type: str,
    normalized: str,
    api_key: str = Depends(verify_api_key),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    graph_repo: GraphRepository = Depends(get_graph_repository),
) -> EntityDetailResponse:
    _require_ner_enabled()
    start = time.perf_counter()

    try:
        detail = await doc_repo.get_entity_detail(entity_type, normalized)
        if not detail:
            raise HTTPException(
                status_code=404,
                detail=f"Entity {entity_type}:{normalized} not found",
            )

        # Try to find linked graph node
        graph_node_id = None
        try:
            settings = _get_settings()
            if settings.graph_enabled:
                node = await graph_repo.get_node(normalized.lower().replace(" ", "_"))
                if node:
                    graph_node_id = node["node_id"]
        except Exception:
            pass  # Graph lookup is best-effort

        latency_ms = (time.perf_counter() - start) * 1000
        return EntityDetailResponse(
            type=detail["type"],
            normalized=detail["normalized"],
            mention_count=detail["mention_count"],
            first_seen=detail["first_seen"].isoformat() if detail["first_seen"] else None,
            last_seen=detail["last_seen"].isoformat() if detail["last_seen"] else None,
            platforms=detail["platforms"],
            graph_node_id=graph_node_id,
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_entity_detail_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get entity detail")


@router.get(
    "/entities/{entity_type}/{normalized}/documents",
    response_model=ThemeDocumentsResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Documents mentioning an entity",
)
@limiter.limit(lambda: _get_settings().rate_limit_entities)
async def get_entity_documents(
    request: Request,
    entity_type: str,
    normalized: str,
    platform: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    api_key: str = Depends(verify_api_key),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> ThemeDocumentsResponse:
    _require_ner_enabled()
    start = time.perf_counter()

    try:
        docs = await doc_repo.get_documents_by_entity(
            entity_type=entity_type,
            entity_normalized=normalized,
            limit=limit,
        )

        # Apply platform filter and offset in Python (containment query doesn't support these)
        if platform:
            docs = [d for d in docs if d.platform == platform]
        page = docs[offset : offset + limit]

        items = [
            ThemeDocumentItem(
                document_id=d.id,
                platform=d.platform if d.platform else None,
                title=d.title,
                content_preview=d.content[:300] if d.content else None,
                url=d.url,
                author_name=d.author_name,
                tickers=d.tickers_mentioned,
                authority_score=d.authority_score,
                sentiment_label=(
                    d.sentiment.get("overall_label") if d.sentiment else None
                ),
                sentiment_confidence=(
                    d.sentiment.get("overall_confidence") if d.sentiment else None
                ),
                timestamp=d.timestamp.isoformat() if d.timestamp else None,
            )
            for d in page
        ]

        latency_ms = (time.perf_counter() - start) * 1000
        return ThemeDocumentsResponse(
            documents=items,
            total=len(docs),
            theme_id=f"{entity_type}:{normalized}",
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_entity_documents_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get entity documents")


@router.get(
    "/entities/{entity_type}/{normalized}/cooccurrence",
    response_model=CooccurrenceResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Entities co-occurring with target entity",
)
@limiter.limit(lambda: _get_settings().rate_limit_entities)
async def get_entity_cooccurrence(
    request: Request,
    entity_type: str,
    normalized: str,
    limit: int = Query(default=20, ge=1, le=100),
    min_count: int = Query(default=2, ge=1),
    api_key: str = Depends(verify_api_key),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> CooccurrenceResponse:
    _require_ner_enabled()
    start = time.perf_counter()

    try:
        rows = await doc_repo.get_cooccurring_entities(
            entity_type=entity_type,
            normalized=normalized,
            limit=limit,
            min_count=min_count,
        )

        items = [
            CooccurringEntityItem(
                type=r["type"],
                normalized=r["normalized"],
                cooccurrence_count=r["cooccurrence_count"],
                jaccard=r["jaccard"],
            )
            for r in rows
        ]

        latency_ms = (time.perf_counter() - start) * 1000
        return CooccurrenceResponse(
            entities=items,
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_cooccurrence_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get entity co-occurrence")


@router.get(
    "/entities/{entity_type}/{normalized}/sentiment",
    response_model=EntitySentimentResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Aggregate sentiment for an entity",
)
@limiter.limit(lambda: _get_settings().rate_limit_entities)
async def get_entity_sentiment(
    request: Request,
    entity_type: str,
    normalized: str,
    api_key: str = Depends(verify_api_key),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> EntitySentimentResponse:
    _require_ner_enabled()
    start = time.perf_counter()

    try:
        result = await doc_repo.get_entity_sentiment(entity_type, normalized)

        latency_ms = (time.perf_counter() - start) * 1000

        if not result:
            return EntitySentimentResponse(
                avg_score=None,
                pos_count=0,
                neg_count=0,
                neu_count=0,
                trend="stable",
                latency_ms=round(latency_ms, 2),
            )

        return EntitySentimentResponse(
            avg_score=round(result["avg_score"], 4) if result["avg_score"] else None,
            pos_count=result["pos_count"],
            neg_count=result["neg_count"],
            neu_count=result["neu_count"],
            trend=result["trend"],
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_entity_sentiment_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get entity sentiment")


@router.post(
    "/entities/{entity_type}/{normalized}/merge",
    response_model=EntityMergeResponse,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Merge an entity into another",
)
@limiter.limit(lambda: _get_settings().rate_limit_admin)
async def merge_entity(
    request: Request,
    entity_type: str,
    normalized: str,
    body: EntityMergeRequest,
    api_key: str = Depends(verify_api_key),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> EntityMergeResponse:
    _require_ner_enabled()
    start = time.perf_counter()

    try:
        if entity_type == body.to_type and normalized == body.to_normalized:
            raise HTTPException(
                status_code=422,
                detail="Cannot merge entity into itself",
            )

        affected = await doc_repo.merge_entity(
            from_type=entity_type,
            from_normalized=normalized,
            to_type=body.to_type,
            to_normalized=body.to_normalized,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Entity merged",
            from_entity=f"{entity_type}:{normalized}",
            to_entity=f"{body.to_type}:{body.to_normalized}",
            affected_documents=affected,
        )

        return EntityMergeResponse(
            affected_documents=affected,
            merged_from=f"{entity_type}:{normalized}",
            merged_to=f"{body.to_type}:{body.to_normalized}",
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("merge_entity_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to merge entity")
