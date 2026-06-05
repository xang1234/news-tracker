"""Cited Q&A endpoint: ask-anything over the structured claim corpus."""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from starlette.requests import Request

from src.api.auth import verify_api_key
from src.api.dependencies import get_qa_service
from src.api.models import (
    AnswerSegmentModel,
    CitedAnswerResponse,
    ErrorResponse,
    QARequest,
)
from src.api.rate_limit import limiter
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()


def _require_cited_qa_enabled() -> None:
    """Gate the cited-Q&A feature flag.

    Declared as a dependency *before* the Q&A service so FastAPI's in-order
    dependency resolution raises this 404 before constructing the (heavy)
    service — a disabled endpoint never opens DB/embedding resources.
    """
    if not get_settings().cited_qa_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cited Q&A is not enabled",
        )


@router.post(
    "/qa",
    response_model=CitedAnswerResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Feature disabled"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Ask a cited question over the claim corpus",
    description=(
        "Answer a free-text question with sentences each citing the evidence claims "
        "they are grounded in. Returns confidence='insufficient' (a refusal) when the "
        "retrieved grounding is too thin to answer."
    ),
)
@limiter.limit(lambda: get_settings().rate_limit_default)
async def ask_question(
    request: Request,
    body: QARequest,
    api_key: str = Depends(verify_api_key),
    _gate: None = Depends(_require_cited_qa_enabled),
    qa_service: Any = Depends(get_qa_service),
) -> CitedAnswerResponse:
    start_time = time.perf_counter()
    try:
        answer = await qa_service.answer(body.question)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return CitedAnswerResponse(
            question=answer.question,
            segments=[
                AnswerSegmentModel(text=s.text, claim_ids=s.claim_ids) for s in answer.segments
            ],
            confidence=answer.confidence,
            claim_count=answer.claim_count,
            generated_by=answer.generated_by,
            model=answer.model,
            generated_at=answer.generated_at,
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ask_question_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to answer question",
        )
