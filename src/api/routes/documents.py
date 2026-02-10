"""
Document explorer endpoints for browsing, filtering, and inspecting documents.
"""

import asyncio
import json
import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.requests import Request

from src.api.auth import verify_api_key
from src.api.dependencies import get_document_repository
from src.api.rate_limit import limiter
from src.config.settings import get_settings as _get_settings
from src.api.models import (
    DocumentDetailResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentStatsResponse,
    EmbeddingCoverage,
    ErrorResponse,
    PlatformCount,
)
from src.storage.repository import DocumentRepository

logger = structlog.get_logger(__name__)
router = APIRouter()

_SORT_WHITELIST = {"timestamp", "authority_score", "spam_score", "fetched_at"}
_ORDER_WHITELIST = {"asc", "desc"}
_PLATFORM_WHITELIST = {"twitter", "reddit", "news", "substack"}
_CONTENT_TYPE_WHITELIST = {"post", "article", "comment", "thread"}


def _extract_sentiment_fields(
    sentiment: dict | str | None,
) -> tuple[str | None, float | None]:
    """Extract label and confidence from a sentiment JSONB value."""
    if sentiment is None:
        return None, None
    if isinstance(sentiment, str):
        sentiment = json.loads(sentiment)
    return sentiment.get("label"), sentiment.get("confidence")


def _parse_engagement(engagement: dict | str | None) -> dict:
    """Parse engagement JSONB to a plain dict."""
    if engagement is None:
        return {}
    if isinstance(engagement, str):
        return json.loads(engagement)
    return dict(engagement)


def _record_to_list_item(row) -> DocumentListItem:
    """Map a lightweight DB record to a DocumentListItem."""
    sentiment_label, sentiment_confidence = _extract_sentiment_fields(
        row.get("sentiment")
    )
    ts = row.get("timestamp")
    fetched = row.get("fetched_at")
    return DocumentListItem(
        document_id=row["id"],
        platform=row.get("platform"),
        content_type=row.get("content_type"),
        title=row.get("title"),
        content_preview=row.get("content_preview"),
        url=row.get("url"),
        author_name=row.get("author_name"),
        author_verified=row.get("author_verified", False),
        author_followers=row.get("author_followers"),
        tickers=list(row.get("tickers", [])),
        spam_score=row.get("spam_score"),
        authority_score=row.get("authority_score"),
        sentiment_label=sentiment_label,
        sentiment_confidence=sentiment_confidence,
        engagement=_parse_engagement(row.get("engagement")),
        theme_ids=list(row.get("theme_ids", [])),
        timestamp=ts.isoformat() if ts else None,
        fetched_at=fetched.isoformat() if fetched else None,
    )


# ── Stats endpoint (MUST be before /{document_id}) ──────────────────


@router.get(
    "/documents/stats",
    response_model=DocumentStatsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Document statistics",
    description="Aggregate statistics for the document explorer dashboard.",
)
@limiter.limit(lambda: _get_settings().rate_limit_search)
async def get_document_stats(
    request: Request,
    api_key: str = Depends(verify_api_key),
    repo: DocumentRepository = Depends(get_document_repository),
) -> DocumentStatsResponse:
    start = time.perf_counter()
    stats = await repo.get_document_stats()
    latency = (time.perf_counter() - start) * 1000

    logger.info("document_stats", total=stats["total_count"], latency_ms=round(latency, 2))

    return DocumentStatsResponse(
        total_count=stats["total_count"],
        platform_counts=[PlatformCount(**pc) for pc in stats["platform_counts"]],
        embedding_coverage=EmbeddingCoverage(**stats["embedding_coverage"]),
        sentiment_coverage=stats["sentiment_coverage"],
        earliest_document=stats["earliest_document"],
        latest_document=stats["latest_document"],
        latency_ms=round(latency, 2),
    )


# ── Detail endpoint ──────────────────────────────────────────────────


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get document detail",
    description="Full document with content, entities, keywords, and events.",
)
async def get_document(
    document_id: str,
    api_key: str = Depends(verify_api_key),
    repo: DocumentRepository = Depends(get_document_repository),
) -> DocumentDetailResponse:
    start = time.perf_counter()
    doc = await repo.get_by_id(document_id)

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )

    latency = (time.perf_counter() - start) * 1000

    # Extract sentiment fields
    sentiment_label = None
    sentiment_confidence = None
    if doc.sentiment:
        sentiment_label = doc.sentiment.get("label")
        sentiment_confidence = doc.sentiment.get("confidence")

    # Normalize entities: {type, normalized} → {type, name}
    entities = [
        {"type": e.get("type", ""), "name": e.get("normalized", e.get("name", ""))}
        for e in (doc.entities_mentioned or [])
    ]

    # Normalize keywords: {text, score} → {word, score}
    keywords = [
        {"word": k.get("text", k.get("word", "")), "score": k.get("score", 0.0)}
        for k in (doc.keywords_extracted or [])
    ]

    # Normalize events
    events = [
        {
            "type": ev.get("event_type", ev.get("type", "")),
            "actor": ev.get("actor", ""),
            "action": ev.get("action", ""),
            "object": ev.get("object", ""),
            "time_ref": ev.get("time_ref", ""),
        }
        for ev in (doc.events_extracted or [])
    ]

    logger.info("document_detail", document_id=document_id, latency_ms=round(latency, 2))

    return DocumentDetailResponse(
        document_id=doc.id,
        platform=doc.platform.value if hasattr(doc.platform, "value") else doc.platform,
        content_type=doc.content_type,
        title=doc.title,
        content_preview=doc.content[:300] if doc.content else None,
        content=doc.content,
        url=doc.url,
        author_id=doc.author_id,
        author_name=doc.author_name,
        author_verified=doc.author_verified,
        author_followers=doc.author_followers,
        tickers=doc.tickers_mentioned,
        spam_score=doc.spam_score,
        bot_probability=doc.bot_probability,
        authority_score=doc.authority_score,
        sentiment_label=sentiment_label,
        sentiment_confidence=sentiment_confidence,
        sentiment=doc.sentiment,
        engagement=doc.engagement.model_dump(),
        entities=entities,
        keywords=keywords,
        events=events,
        urls_mentioned=doc.urls_mentioned,
        theme_ids=doc.theme_ids,
        has_embedding=doc.embedding is not None,
        has_embedding_minilm=doc.embedding_minilm is not None,
        timestamp=doc.timestamp.isoformat() if doc.timestamp else None,
        fetched_at=doc.fetched_at.isoformat() if doc.fetched_at else None,
        latency_ms=round(latency, 2),
    )


# ── List endpoint ────────────────────────────────────────────────────


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="List documents",
    description="Browse and filter documents with pagination.",
)
async def list_documents(
    platform: str | None = Query(default=None, description="Filter by platform"),
    content_type: str | None = Query(default=None, description="Filter by content type"),
    ticker: str | None = Query(default=None, description="Filter by ticker symbol"),
    q: str | None = Query(default=None, description="Full-text search query"),
    since: str | None = Query(default=None, description="Start date (ISO format)"),
    until: str | None = Query(default=None, description="End date (ISO format)"),
    max_spam: float | None = Query(default=None, ge=0.0, le=1.0, description="Maximum spam score"),
    min_authority: float | None = Query(
        default=None, ge=0.0, le=1.0, description="Minimum authority score"
    ),
    sort: str = Query(default="timestamp", description="Sort field"),
    order: str = Query(default="desc", description="Sort order: asc or desc"),
    limit: int = Query(default=50, ge=1, le=200, description="Page size"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    api_key: str = Depends(verify_api_key),
    repo: DocumentRepository = Depends(get_document_repository),
) -> DocumentListResponse:
    # Validate whitelist params
    if sort not in _SORT_WHITELIST:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid sort field '{sort}'. Allowed: {sorted(_SORT_WHITELIST)}",
        )
    if order not in _ORDER_WHITELIST:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid order '{order}'. Allowed: asc, desc",
        )
    if platform is not None and platform not in _PLATFORM_WHITELIST:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid platform '{platform}'. Allowed: {sorted(_PLATFORM_WHITELIST)}",
        )
    if content_type is not None and content_type not in _CONTENT_TYPE_WHITELIST:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid content_type '{content_type}'. Allowed: {sorted(_CONTENT_TYPE_WHITELIST)}",
        )

    # Parse date strings to datetime if provided
    since_dt = None
    until_dt = None
    if since is not None:
        try:
            from datetime import datetime, timezone

            since_dt = datetime.fromisoformat(since)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid 'since' date format: '{since}'. Use ISO 8601.",
            )
    if until is not None:
        try:
            from datetime import datetime, timezone

            until_dt = datetime.fromisoformat(until)
            if until_dt.tzinfo is None:
                until_dt = until_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid 'until' date format: '{until}'. Use ISO 8601.",
            )

    filter_kwargs = dict(
        platform=platform,
        content_type=content_type,
        ticker=ticker,
        q=q,
        since=since_dt,
        until=until_dt,
        max_spam=max_spam,
        min_authority=min_authority,
    )

    start = time.perf_counter()
    total, rows = await asyncio.gather(
        repo.list_documents_count(**filter_kwargs),
        repo.list_documents(**filter_kwargs, sort=sort, order=order, limit=limit, offset=offset),
    )
    latency = (time.perf_counter() - start) * 1000

    documents = [_record_to_list_item(row) for row in rows]

    logger.info(
        "document_list",
        total=total,
        returned=len(documents),
        latency_ms=round(latency, 2),
    )

    return DocumentListResponse(
        documents=documents,
        total=total,
        page_size=limit,
        offset=offset,
        latency_ms=round(latency, 2),
    )
