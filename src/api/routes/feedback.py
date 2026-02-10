"""Feedback endpoints for submitting and querying user quality ratings."""

import time

from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog

from src.api.auth import verify_api_key
from src.api.dependencies import get_feedback_repository
from src.api.models import (
    ErrorResponse,
    FeedbackItem,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsItem,
    FeedbackStatsResponse,
)
from src.feedback.config import FeedbackConfig
from src.feedback.repository import FeedbackRepository
from src.feedback.schemas import VALID_ENTITY_TYPES, VALID_QUALITY_LABELS, Feedback

logger = structlog.get_logger(__name__)
router = APIRouter()

_config = FeedbackConfig()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Submit feedback",
    description=(
        "Submit a quality rating for a theme, alert, or document. "
        "The authenticated API key is recorded as user_id."
    ),
)
async def create_feedback(
    request: FeedbackRequest,
    api_key: str = Depends(verify_api_key),
    feedback_repo: FeedbackRepository = Depends(get_feedback_repository),
) -> FeedbackResponse:
    start_time = time.perf_counter()

    try:
        # Validate entity_type
        if request.entity_type not in VALID_ENTITY_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid entity_type {request.entity_type!r}. "
                    f"Must be one of: {sorted(VALID_ENTITY_TYPES)}"
                ),
            )

        # Validate quality_label if provided
        if request.quality_label is not None and request.quality_label not in VALID_QUALITY_LABELS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid quality_label {request.quality_label!r}. "
                    f"Must be one of: {sorted(VALID_QUALITY_LABELS)}"
                ),
            )

        # Truncate comment if too long
        comment = request.comment
        if comment and len(comment) > _config.max_comment_length:
            comment = comment[: _config.max_comment_length]

        feedback = Feedback(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            rating=request.rating,
            quality_label=request.quality_label,
            comment=comment,
            user_id=api_key,
        )

        created = await feedback_repo.create(feedback)

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Feedback created",
            feedback_id=created.feedback_id,
            entity_type=created.entity_type,
            entity_id=created.entity_id,
            rating=created.rating,
            latency_ms=round(latency_ms, 2),
        )

        return FeedbackResponse(
            feedback=FeedbackItem(
                feedback_id=created.feedback_id,
                entity_type=created.entity_type,
                entity_id=created.entity_id,
                rating=created.rating,
                quality_label=created.quality_label,
                comment=created.comment,
                user_id=created.user_id,
                created_at=created.created_at.isoformat(),
            ),
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("create_feedback_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create feedback",
        )


@router.get(
    "/feedback/stats",
    response_model=FeedbackStatsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Invalid filter parameter"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get feedback statistics",
    description=(
        "Get aggregated feedback statistics grouped by entity type, "
        "including average rating and quality label distribution."
    ),
)
async def get_feedback_stats(
    entity_type: str | None = Query(
        default=None,
        description="Filter by entity type: theme, alert, document",
    ),
    api_key: str = Depends(verify_api_key),
    feedback_repo: FeedbackRepository = Depends(get_feedback_repository),
) -> FeedbackStatsResponse:
    start_time = time.perf_counter()

    try:
        # Validate entity_type if provided
        if entity_type is not None and entity_type not in VALID_ENTITY_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid entity_type {entity_type!r}. "
                    f"Must be one of: {sorted(VALID_ENTITY_TYPES)}"
                ),
            )

        stats = await feedback_repo.get_stats(entity_type=entity_type)

        items = [
            FeedbackStatsItem(
                entity_type=s["entity_type"],
                total_count=s["total_count"],
                avg_rating=s["avg_rating"],
                label_distribution=s["label_distribution"],
            )
            for s in stats
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Feedback stats retrieved",
            entity_type=entity_type,
            groups=len(items),
            latency_ms=round(latency_ms, 2),
        )

        return FeedbackStatsResponse(
            stats=items,
            total=len(items),
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_feedback_stats_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get feedback stats",
        )
