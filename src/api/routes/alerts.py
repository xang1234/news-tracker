"""Alert endpoints for retrieving and acknowledging alerts."""

import time

from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog

from src.alerts.repository import AlertRepository
from src.alerts.schemas import VALID_SEVERITIES, VALID_TRIGGER_TYPES
from src.api.auth import verify_api_key
from src.api.dependencies import get_alert_repository
from src.api.models import AlertItem, AlertsResponse, ErrorResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/alerts",
    response_model=AlertsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Invalid filter parameter"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="List alerts",
    description=(
        "List alerts with optional filtering by severity, trigger type, "
        "theme, and acknowledgement status. Ordered by most recent first."
    ),
)
async def list_alerts(
    severity: str | None = Query(
        default=None,
        description="Filter by severity: critical, warning, info",
    ),
    trigger_type: str | None = Query(
        default=None,
        description="Filter by trigger type: sentiment_velocity, extreme_sentiment, volume_surge, lifecycle_change, new_theme",
    ),
    theme_id: str | None = Query(
        default=None,
        description="Filter by theme identifier",
    ),
    acknowledged: bool | None = Query(
        default=None,
        description="Filter by acknowledgement status",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum alerts to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    api_key: str = Depends(verify_api_key),
    alert_repo: AlertRepository = Depends(get_alert_repository),
) -> AlertsResponse:
    start_time = time.perf_counter()

    try:
        # Validate enum params
        if severity and severity not in VALID_SEVERITIES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid severity {severity!r}. "
                    f"Must be one of: {sorted(VALID_SEVERITIES)}"
                ),
            )

        if trigger_type and trigger_type not in VALID_TRIGGER_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid trigger_type {trigger_type!r}. "
                    f"Must be one of: {sorted(VALID_TRIGGER_TYPES)}"
                ),
            )

        alerts = await alert_repo.get_recent(
            severity=severity,
            trigger_type=trigger_type,
            theme_id=theme_id,
            acknowledged=acknowledged,
            limit=limit,
            offset=offset,
        )

        items = [
            AlertItem(
                alert_id=a.alert_id,
                theme_id=a.theme_id,
                trigger_type=a.trigger_type,
                severity=a.severity,
                title=a.title,
                message=a.message,
                trigger_data=a.trigger_data,
                acknowledged=a.acknowledged,
                created_at=a.created_at.isoformat(),
            )
            for a in alerts
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Alerts listed",
            total=len(items),
            severity=severity,
            trigger_type=trigger_type,
            latency_ms=round(latency_ms, 2),
        )

        return AlertsResponse(
            alerts=items,
            total=len(items),
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list alerts: {str(e)}",
        )
