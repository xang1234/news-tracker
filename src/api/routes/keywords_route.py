"""Keywords extraction endpoint for playground."""

import time

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import verify_api_key
from src.api.dependencies import get_keywords_service
from src.api.models import (
    ErrorResponse,
    KeywordItem,
    KeywordsRequest,
    KeywordsResponse,
    KeywordsResultItem,
)
from src.config.settings import get_settings
from src.keywords.service import KeywordsService

router = APIRouter()


@router.post(
    "/keywords",
    response_model=KeywordsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Keywords extraction error"},
        503: {"model": ErrorResponse, "description": "Keywords service disabled"},
    },
    summary="Extract keywords from texts",
    description="Extract important keywords using TextRank algorithm via rapid-textrank.",
)
async def extract_keywords(
    request: KeywordsRequest,
    api_key: str = Depends(verify_api_key),
    service: KeywordsService = Depends(get_keywords_service),
) -> KeywordsResponse:
    settings = get_settings()
    if not settings.keywords_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Keywords service is disabled. Set KEYWORDS_ENABLED=true to enable.",
        )

    start_time = time.perf_counter()

    try:
        batch_results = await service.extract_batch(request.texts)

        results = []
        for text, keywords in zip(request.texts, batch_results):
            items = [
                KeywordItem(
                    text=kw.text,
                    score=kw.score,
                    rank=kw.rank,
                    lemma=kw.lemma,
                    count=kw.count,
                )
                for kw in keywords[: request.top_n]
            ]
            results.append(KeywordsResultItem(keywords=items, text_length=len(text)))

        latency_ms = (time.perf_counter() - start_time) * 1000

        return KeywordsResponse(
            results=results,
            total=len(results),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Keywords extraction failed: {str(e)}",
        )
