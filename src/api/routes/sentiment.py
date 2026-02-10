"""
Sentiment analysis endpoint.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.requests import Request

from src.api.auth import verify_api_key
from src.api.dependencies import get_sentiment_service
from src.api.models import (
    EntitySentimentItem,
    ErrorResponse,
    SentimentRequest,
    SentimentResponse,
    SentimentResultItem,
    SentimentScores,
)
from src.api.rate_limit import limiter
from src.config.settings import get_settings
from src.sentiment.service import SentimentService

router = APIRouter()


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Sentiment analysis error"},
    },
    summary="Analyze sentiment of texts",
    description="""
    Analyze financial sentiment using FinBERT.

    Returns document-level sentiment (positive/neutral/negative) with confidence scores.
    For entity-level sentiment, use the processing pipeline with NER enabled.

    Model: ProsusAI/finbert (fine-tuned for financial text)
    Classes: positive (bullish), neutral, negative (bearish)

    Caching is enabled by default to speed up repeated requests.
    """,
)
@limiter.limit(lambda: get_settings().rate_limit_sentiment)
async def analyze_sentiment(
    request: Request,
    body: SentimentRequest,
    api_key: str = Depends(verify_api_key),
    service: SentimentService = Depends(get_sentiment_service),
) -> SentimentResponse:
    """
    Analyze sentiment for a batch of texts.

    Args:
        request: Starlette request (for rate limiting)
        body: Sentiment request with texts and options
        api_key: Validated API key
        service: Sentiment service

    Returns:
        Sentiment results with metadata
    """
    start_time = time.perf_counter()

    try:
        # Analyze batch (document-level)
        sentiment_results = await service.analyze_batch(
            body.texts,
            show_progress=False,
        )

        # Convert to API response format
        results = []
        for sr in sentiment_results:
            # Convert entity sentiments if present
            entity_items = []
            for e in sr.get("entity_sentiments", []):
                entity_items.append(
                    EntitySentimentItem(
                        entity=e["entity"],
                        type=e["type"],
                        label=e["label"],
                        confidence=e["confidence"],
                        scores=SentimentScores(
                            positive=e["scores"]["positive"],
                            neutral=e["scores"]["neutral"],
                            negative=e["scores"]["negative"],
                        ),
                        context=e.get("context"),
                    )
                )

            results.append(
                SentimentResultItem(
                    label=sr["label"],
                    confidence=sr["confidence"],
                    scores=SentimentScores(
                        positive=sr["scores"]["positive"],
                        neutral=sr["scores"]["neutral"],
                        negative=sr["scores"]["negative"],
                    ),
                    entity_sentiments=entity_items,
                )
            )

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        return SentimentResponse(
            results=results,
            total=len(results),
            model=sr.get("model", "ProsusAI/finbert") if sentiment_results else "ProsusAI/finbert",
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}",
        )
