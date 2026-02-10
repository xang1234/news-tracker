"""NER extraction endpoint for playground."""

import time

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import verify_api_key
from src.api.dependencies import get_ner_service
from src.api.models import (
    ErrorResponse,
    NEREntityItem,
    NERRequest,
    NERResponse,
    NERResultItem,
)
from src.config.settings import get_settings
from src.ner.service import NERService

router = APIRouter()


@router.post(
    "/ner",
    response_model=NERResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "NER extraction error"},
        503: {"model": ErrorResponse, "description": "NER service disabled"},
    },
    summary="Extract named entities from texts",
    description="Extract financial entities (tickers, companies, products, technologies, metrics) using spaCy NER.",
)
async def extract_entities(
    request: NERRequest,
    api_key: str = Depends(verify_api_key),
    service: NERService = Depends(get_ner_service),
) -> NERResponse:
    settings = get_settings()
    if not settings.ner_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NER service is disabled. Set NER_ENABLED=true to enable.",
        )

    start_time = time.perf_counter()

    try:
        batch_results = await service.extract_batch(request.texts)

        results = []
        for text, entities in zip(request.texts, batch_results):
            items = [
                NEREntityItem(
                    text=e.text,
                    type=e.type,
                    normalized=e.normalized,
                    start=e.start,
                    end=e.end,
                    confidence=e.confidence,
                    metadata=e.metadata,
                )
                for e in entities
            ]
            results.append(NERResultItem(entities=items, text_length=len(text)))

        latency_ms = (time.perf_counter() - start_time) * 1000

        return NERResponse(
            results=results,
            total=len(results),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NER extraction failed: {str(e)}",
        )
