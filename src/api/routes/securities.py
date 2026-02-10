"""Securities Admin endpoints — CRUD for the security master table."""

import time

from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog

from src.api.auth import verify_api_key
from src.api.dependencies import get_security_master_repository
from src.api.models import (
    CreateSecurityRequest,
    ErrorResponse,
    SecuritiesListResponse,
    SecurityItem,
    UpdateSecurityRequest,
)
from src.config.settings import get_settings
from src.security_master.repository import SecurityMasterRepository
from src.security_master.schemas import Security

logger = structlog.get_logger(__name__)
router = APIRouter()


def _require_security_master_enabled() -> None:
    settings = get_settings()
    if not settings.security_master_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Securities endpoints require security_master_enabled=true",
        )


def _security_to_item(s: Security) -> SecurityItem:
    return SecurityItem(
        ticker=s.ticker,
        exchange=s.exchange,
        name=s.name,
        aliases=s.aliases,
        sector=s.sector,
        country=s.country,
        currency=s.currency,
        is_active=s.is_active,
        created_at=s.created_at.isoformat() if s.created_at else None,
        updated_at=s.updated_at.isoformat() if s.updated_at else None,
    )


@router.get(
    "/securities",
    response_model=SecuritiesListResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="List securities with filters",
)
async def list_securities(
    search: str | None = Query(default=None, description="Search ticker/name/aliases"),
    active_only: bool = Query(default=False, description="Only active securities"),
    exchange: str | None = Query(default=None, description="Filter by exchange"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    api_key: str = Depends(verify_api_key),
    repo: SecurityMasterRepository = Depends(get_security_master_repository),
) -> SecuritiesListResponse:
    _require_security_master_enabled()
    start = time.perf_counter()

    try:
        securities, total = await repo.list_securities(
            search=search,
            active_only=active_only,
            exchange=exchange,
            limit=limit,
            offset=offset,
        )

        items = [_security_to_item(s) for s in securities]

        latency_ms = (time.perf_counter() - start) * 1000
        return SecuritiesListResponse(
            securities=items,
            total=total,
            has_more=(offset + limit) < total,
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list securities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/securities",
    response_model=SecurityItem,
    status_code=status.HTTP_201_CREATED,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Create a new security",
)
async def create_security(
    body: CreateSecurityRequest,
    api_key: str = Depends(verify_api_key),
    repo: SecurityMasterRepository = Depends(get_security_master_repository),
) -> SecurityItem:
    _require_security_master_enabled()

    try:
        security = Security(
            ticker=body.ticker.upper(),
            exchange=body.exchange.upper(),
            name=body.name,
            aliases=body.aliases,
            sector=body.sector,
            country=body.country,
            currency=body.currency,
        )
        await repo.upsert(security)

        # Re-fetch to get timestamps
        created = await repo.get_by_ticker(security.ticker, security.exchange)
        if not created:
            raise HTTPException(status_code=500, detail="Failed to create security")

        logger.info("Security created", ticker=security.ticker, exchange=security.exchange)
        return _security_to_item(created)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create security: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/securities/{ticker}/{exchange}",
    response_model=SecurityItem,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Update a security",
)
async def update_security(
    ticker: str,
    exchange: str,
    body: UpdateSecurityRequest,
    api_key: str = Depends(verify_api_key),
    repo: SecurityMasterRepository = Depends(get_security_master_repository),
) -> SecurityItem:
    _require_security_master_enabled()

    try:
        existing = await repo.get_by_ticker(ticker.upper(), exchange.upper())
        if not existing:
            raise HTTPException(
                status_code=404,
                detail=f"Security {ticker}:{exchange} not found",
            )

        # Apply updates
        updated = Security(
            ticker=existing.ticker,
            exchange=existing.exchange,
            name=body.name if body.name is not None else existing.name,
            aliases=body.aliases if body.aliases is not None else existing.aliases,
            sector=body.sector if body.sector is not None else existing.sector,
            country=body.country if body.country is not None else existing.country,
            currency=body.currency if body.currency is not None else existing.currency,
            is_active=existing.is_active,
        )
        await repo.upsert(updated)

        result = await repo.get_by_ticker(updated.ticker, updated.exchange)
        logger.info("Security updated", ticker=ticker, exchange=exchange)
        return _security_to_item(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update security: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/securities/{ticker}/{exchange}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Deactivate a security (soft delete)",
)
async def deactivate_security(
    ticker: str,
    exchange: str,
    api_key: str = Depends(verify_api_key),
    repo: SecurityMasterRepository = Depends(get_security_master_repository),
) -> None:
    _require_security_master_enabled()

    try:
        deactivated = await repo.deactivate(ticker.upper(), exchange.upper())
        if not deactivated:
            raise HTTPException(
                status_code=404,
                detail=f"Security {ticker}:{exchange} not found or already inactive",
            )

        logger.info("Security deactivated", ticker=ticker, exchange=exchange)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deactivate security: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
