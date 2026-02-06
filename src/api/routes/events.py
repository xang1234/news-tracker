"""Event endpoints for theme-linked event retrieval."""

import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog

from src.api.auth import verify_api_key
from src.api.dependencies import get_document_repository, get_theme_repository
from src.api.models import (
    ErrorResponse,
    ThemeEventItem,
    ThemeEventsResponse,
)
from src.event_extraction.schemas import VALID_EVENT_TYPES
from src.event_extraction.theme_integration import EventThemeLinker, ThemeWithEvents
from src.storage.repository import DocumentRepository
from src.themes.repository import ThemeRepository

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/themes/{theme_id}/events",
    response_model=ThemeEventsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Theme not found"},
        422: {"model": ErrorResponse, "description": "Invalid event_type"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get theme events",
    description=(
        "Get events linked to a theme via ticker overlap. "
        "Events are deduplicated across documents and include "
        "an investment signal derived from event distribution."
    ),
)
async def get_theme_events(
    theme_id: str,
    event_type: str | None = Query(
        default=None,
        description="Filter by event type (e.g., capacity_expansion, product_launch)",
    ),
    days: int = Query(
        default=7, ge=1, le=90, description="Lookback window in days"
    ),
    limit: int = Query(
        default=20, ge=1, le=200, description="Maximum events to return"
    ),
    api_key: str = Depends(verify_api_key),
    theme_repo: ThemeRepository = Depends(get_theme_repository),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> ThemeEventsResponse:
    start_time = time.perf_counter()

    try:
        # Validate event_type if provided
        if event_type and event_type not in VALID_EVENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid event_type {event_type!r}. "
                    f"Must be one of: {sorted(VALID_EVENT_TYPES)}"
                ),
            )

        # Verify theme exists
        theme = await theme_repo.get_by_id(theme_id)
        if theme is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Theme {theme_id!r} not found",
            )

        # Early return if theme has no tickers
        if not theme.top_tickers:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ThemeEventsResponse(
                events=[],
                total=0,
                theme_id=theme_id,
                event_counts={},
                investment_signal=None,
                latency_ms=round(latency_ms, 2),
            )

        now = datetime.now(timezone.utc)
        since = now - timedelta(days=days)

        # Over-fetch to account for dedup losses
        fetch_limit = limit * 3

        raw_events = await doc_repo.get_events_by_tickers(
            tickers=theme.top_tickers,
            since=since,
            until=now,
            event_type=event_type,
            limit=fetch_limit,
        )

        # Link events to theme (filter by ticker overlap)
        linked = EventThemeLinker.link_events_to_theme(raw_events, theme)

        # Deduplicate cross-document events
        deduped = EventThemeLinker.deduplicate_events(linked)

        # Truncate to requested limit
        deduped = deduped[:limit]

        # Build summary
        summary = ThemeWithEvents.from_events(theme_id, deduped)

        # Convert to response items
        items = [
            ThemeEventItem(
                event_id=e["event_id"],
                doc_id=e["doc_id"],
                event_type=e["event_type"],
                actor=e.get("actor"),
                action=e["action"],
                object=e.get("object"),
                time_ref=e.get("time_ref"),
                quantity=e.get("quantity"),
                tickers=e.get("tickers", []),
                confidence=e.get("confidence", 0.7),
                source_doc_ids=e.get("source_doc_ids", [e["doc_id"]]),
                created_at=(
                    e["created_at"].isoformat()
                    if hasattr(e.get("created_at"), "isoformat")
                    else e.get("created_at")
                ),
            )
            for e in deduped
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Theme events retrieved",
            theme_id=theme_id,
            total=len(items),
            raw_count=len(raw_events),
            linked_count=len(linked),
            event_type=event_type,
            days=days,
            latency_ms=round(latency_ms, 2),
        )

        return ThemeEventsResponse(
            events=items,
            total=len(items),
            theme_id=theme_id,
            event_counts=summary.event_counts,
            investment_signal=summary.investment_signal(),
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get theme events: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get theme events: {str(e)}",
        )
