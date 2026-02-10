"""Event extraction endpoint for playground."""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import verify_api_key
from src.api.dependencies import get_pattern_extractor
from src.api.models import (
    ErrorResponse,
    EventsExtractRequest,
    EventsExtractResponse,
    ExtractedEventItem,
)
from src.config.settings import get_settings
from src.event_extraction.patterns import PatternExtractor
from src.ingestion.schemas import NormalizedDocument, Platform

router = APIRouter()


@router.post(
    "/events/extract",
    response_model=EventsExtractResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Event extraction error"},
        503: {"model": ErrorResponse, "description": "Events service disabled"},
    },
    summary="Extract events from text",
    description="Extract structured SVO events (capacity, product, price, guidance) from financial text.",
)
async def extract_events(
    request: EventsExtractRequest,
    api_key: str = Depends(verify_api_key),
    extractor: PatternExtractor = Depends(get_pattern_extractor),
) -> EventsExtractResponse:
    settings = get_settings()
    if not settings.events_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Events service is disabled. Set EVENTS_ENABLED=true to enable.",
        )

    start_time = time.perf_counter()

    try:
        doc = NormalizedDocument(
            id="playground_0",
            platform=Platform.NEWS,
            timestamp=datetime.now(timezone.utc),
            author_id="playground",
            author_name="Playground",
            content=request.text,
            tickers_mentioned=request.tickers or [],
        )

        event_records = extractor.extract(doc)

        events = [
            ExtractedEventItem(
                event_type=er.event_type,
                actor=er.actor,
                action=er.action,
                object=er.object,
                time_ref=er.time_ref,
                quantity=er.quantity,
                tickers=er.tickers,
                confidence=er.confidence,
                span_start=er.span_start,
                span_end=er.span_end,
            )
            for er in event_records
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        return EventsExtractResponse(
            events=events,
            total=len(events),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Event extraction failed: {str(e)}",
        )
