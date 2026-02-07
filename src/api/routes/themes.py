"""
Theme endpoints for dashboard, trading system, and alert consumption.
"""

import time
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog

from src.api.auth import verify_api_key
from src.api.dependencies import (
    get_document_repository,
    get_ranking_service,
    get_sentiment_aggregator,
    get_theme_repository,
)
from src.api.models import (
    ErrorResponse,
    RankedThemeItem,
    RankedThemesResponse,
    ThemeDetailResponse,
    ThemeDocumentItem,
    ThemeDocumentsResponse,
    ThemeItem,
    ThemeListResponse,
    ThemeMetricsItem,
    ThemeMetricsResponse,
    ThemeSentimentResponse,
)
from src.sentiment.aggregation import DocumentSentiment, SentimentAggregator
from src.storage.repository import DocumentRepository
from src.themes.ranking import ThemeRankingService
from src.themes.repository import ThemeRepository
from src.themes.schemas import Theme

logger = structlog.get_logger(__name__)
router = APIRouter()


def _theme_to_item(theme: Theme, include_centroid: bool = False) -> ThemeItem:
    """Convert a Theme dataclass to a ThemeItem response model."""
    return ThemeItem(
        theme_id=theme.theme_id,
        name=theme.name,
        top_keywords=theme.top_keywords,
        top_tickers=theme.top_tickers,
        top_entities=theme.top_entities,
        lifecycle_stage=theme.lifecycle_stage,
        document_count=theme.document_count,
        description=theme.description,
        created_at=theme.created_at.isoformat(),
        updated_at=theme.updated_at.isoformat(),
        metadata=theme.metadata,
        centroid=theme.centroid.tolist() if include_centroid else None,
    )


@router.get(
    "/themes",
    response_model=ThemeListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="List themes",
    description="List all themes with optional lifecycle stage filtering and pagination.",
)
async def list_themes(
    lifecycle_stage: str | None = Query(
        default=None,
        description="Filter by lifecycle stage: emerging, accelerating, mature, fading",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum themes to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    include_centroid: bool = Query(
        default=False, description="Include 768-dim centroid vectors (adds ~6KB per theme)"
    ),
    api_key: str = Depends(verify_api_key),
    repo: ThemeRepository = Depends(get_theme_repository),
) -> ThemeListResponse:
    start_time = time.perf_counter()

    try:
        stages = [lifecycle_stage] if lifecycle_stage else None
        # Over-fetch to handle offset in Python (theme count is low, <1000)
        themes = await repo.get_all(lifecycle_stages=stages, limit=limit + offset)
        page = themes[offset : offset + limit]

        items = [_theme_to_item(t, include_centroid) for t in page]
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Themes listed",
            total=len(items),
            lifecycle_stage=lifecycle_stage,
            latency_ms=round(latency_ms, 2),
        )

        return ThemeListResponse(
            themes=items,
            total=len(items),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Failed to list themes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list themes: {str(e)}",
        )


@router.get(
    "/themes/ranked",
    response_model=RankedThemesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get ranked themes",
    description="""
    Rank themes by trading actionability using strategy-specific scoring.

    The scoring formula combines volume momentum (z-score), narrative
    compellingness, and lifecycle stage into a single composite score.
    Themes are assigned to tiers: Tier 1 (top 5%), Tier 2 (top 20%),
    Tier 3 (rest).

    Strategies:
    - **swing**: Emphasizes volume momentum (alpha=0.6) for short-term trades
    - **position**: Emphasizes compellingness (beta=0.6) for longer-term positions
    """,
)
async def get_ranked_themes(
    strategy: str = Query(
        default="swing",
        description="Ranking strategy: swing (volume-biased) or position (compellingness-biased)",
    ),
    max_tier: int = Query(
        default=3, ge=1, le=3, description="Maximum tier to include (1=top 5%, 2=top 20%, 3=all)"
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum themes to return"),
    api_key: str = Depends(verify_api_key),
    ranking_service: ThemeRankingService = Depends(get_ranking_service),
) -> RankedThemesResponse:
    start_time = time.perf_counter()

    try:
        if strategy not in ("swing", "position"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy {strategy!r}. Must be 'swing' or 'position'.",
            )

        ranked = await ranking_service.get_actionable(
            strategy=strategy,
            max_tier=max_tier,
        )

        # Apply limit
        ranked = ranked[:limit]

        items = [
            RankedThemeItem(
                theme=_theme_to_item(r.theme),
                score=round(r.score, 4),
                tier=r.tier,
                components=r.components,
            )
            for r in ranked
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Themes ranked",
            strategy=strategy,
            max_tier=max_tier,
            total=len(items),
            latency_ms=round(latency_ms, 2),
        )

        return RankedThemesResponse(
            themes=items,
            total=len(items),
            strategy=strategy,
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rank themes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rank themes: {str(e)}",
        )


@router.get(
    "/themes/{theme_id}",
    response_model=ThemeDetailResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Theme not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get theme details",
    description="Get detailed information about a specific theme.",
)
async def get_theme(
    theme_id: str,
    include_centroid: bool = Query(
        default=False, description="Include 768-dim centroid vector"
    ),
    api_key: str = Depends(verify_api_key),
    repo: ThemeRepository = Depends(get_theme_repository),
) -> ThemeDetailResponse:
    start_time = time.perf_counter()

    try:
        theme = await repo.get_by_id(theme_id)
        if theme is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Theme {theme_id!r} not found",
            )

        item = _theme_to_item(theme, include_centroid)
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Theme retrieved",
            theme_id=theme_id,
            latency_ms=round(latency_ms, 2),
        )

        return ThemeDetailResponse(
            theme=item,
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get theme: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get theme: {str(e)}",
        )


@router.get(
    "/themes/{theme_id}/documents",
    response_model=ThemeDocumentsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Theme not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get theme documents",
    description="Get documents assigned to a theme with optional filtering.",
)
async def get_theme_documents(
    theme_id: str,
    limit: int = Query(default=50, ge=1, le=200, description="Maximum documents to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    platform: str | None = Query(default=None, description="Filter by platform"),
    min_authority: float | None = Query(
        default=None, ge=0.0, le=1.0, description="Minimum authority score"
    ),
    api_key: str = Depends(verify_api_key),
    theme_repo: ThemeRepository = Depends(get_theme_repository),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> ThemeDocumentsResponse:
    start_time = time.perf_counter()

    try:
        # Verify theme exists
        theme = await theme_repo.get_by_id(theme_id)
        if theme is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Theme {theme_id!r} not found",
            )

        docs = await doc_repo.get_documents_by_theme(
            theme_id=theme_id,
            limit=limit,
            offset=offset,
            platform=platform,
            min_authority=min_authority,
        )

        items = []
        for doc in docs:
            sentiment_label = None
            sentiment_confidence = None
            if doc.sentiment and isinstance(doc.sentiment, dict):
                sentiment_label = doc.sentiment.get("label")
                sentiment_confidence = doc.sentiment.get("confidence")

            items.append(
                ThemeDocumentItem(
                    document_id=doc.id,
                    platform=doc.platform.value if hasattr(doc.platform, "value") else doc.platform,
                    title=doc.title,
                    content_preview=doc.content[:300] if doc.content else None,
                    url=doc.url,
                    author_name=doc.author_name,
                    tickers=doc.tickers_mentioned,
                    authority_score=doc.authority_score,
                    sentiment_label=sentiment_label,
                    sentiment_confidence=sentiment_confidence,
                    timestamp=doc.timestamp.isoformat() if doc.timestamp else None,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Theme documents retrieved",
            theme_id=theme_id,
            total=len(items),
            latency_ms=round(latency_ms, 2),
        )

        return ThemeDocumentsResponse(
            documents=items,
            total=len(items),
            theme_id=theme_id,
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get theme documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get theme documents: {str(e)}",
        )


@router.get(
    "/themes/{theme_id}/sentiment",
    response_model=ThemeSentimentResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Theme not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get theme sentiment",
    description="""
    Aggregate sentiment for a theme over a time window.

    Uses exponential decay weighting (recent documents weighted more heavily),
    authority and confidence weighting, and detects sentiment velocity
    (rate of change) and extreme crowded positions.
    """,
)
async def get_theme_sentiment(
    theme_id: str,
    window_days: int = Query(default=7, ge=1, le=90, description="Lookback window in days"),
    api_key: str = Depends(verify_api_key),
    theme_repo: ThemeRepository = Depends(get_theme_repository),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    aggregator: SentimentAggregator = Depends(get_sentiment_aggregator),
) -> ThemeSentimentResponse:
    start_time = time.perf_counter()

    try:
        # Verify theme exists
        theme = await theme_repo.get_by_id(theme_id)
        if theme is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Theme {theme_id!r} not found",
            )

        now = datetime.now(timezone.utc)
        since = now - timedelta(days=window_days)

        # Fetch lightweight sentiment rows
        rows = await doc_repo.get_sentiments_for_theme(theme_id, since=since, until=now)

        # Convert to DocumentSentiment objects, skipping invalid rows
        doc_sentiments: list[DocumentSentiment] = []
        for row in rows:
            sentiment = row.get("sentiment")
            if not isinstance(sentiment, dict):
                continue
            label = sentiment.get("label")
            if label not in ("positive", "negative", "neutral"):
                continue
            confidence = sentiment.get("confidence", 0.5)
            scores = sentiment.get("scores", {})

            doc_sentiments.append(
                DocumentSentiment(
                    document_id=row["document_id"],
                    timestamp=row["timestamp"],
                    label=label,
                    confidence=confidence,
                    scores=scores,
                    authority_score=row.get("authority_score"),
                    platform=row.get("platform"),
                )
            )

        # Aggregate (sync, CPU-only)
        result = aggregator.aggregate_theme_sentiment(
            theme_id=theme_id,
            ticker=None,
            document_sentiments=doc_sentiments,
            window_days=window_days,
            reference_time=now,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Theme sentiment aggregated",
            theme_id=theme_id,
            document_count=result.document_count,
            window_days=window_days,
            latency_ms=round(latency_ms, 2),
        )

        return ThemeSentimentResponse(
            theme_id=theme_id,
            bullish_ratio=result.bullish_ratio,
            bearish_ratio=result.bearish_ratio,
            neutral_ratio=result.neutral_ratio,
            avg_confidence=result.avg_confidence,
            avg_authority=result.avg_authority,
            sentiment_velocity=result.sentiment_velocity,
            extreme_sentiment=result.extreme_sentiment,
            document_count=result.document_count,
            window_start=result.window_start.isoformat(),
            window_end=result.window_end.isoformat(),
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to aggregate theme sentiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to aggregate theme sentiment: {str(e)}",
        )


@router.get(
    "/themes/{theme_id}/metrics",
    response_model=ThemeMetricsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Theme not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get theme metrics",
    description="Get daily metrics time series for a theme within a date range.",
)
async def get_theme_metrics(
    theme_id: str,
    start_date: date | None = Query(
        default=None, description="Start date (inclusive, default: 30 days ago)"
    ),
    end_date: date | None = Query(
        default=None, description="End date (inclusive, default: today)"
    ),
    api_key: str = Depends(verify_api_key),
    theme_repo: ThemeRepository = Depends(get_theme_repository),
) -> ThemeMetricsResponse:
    start_time = time.perf_counter()

    try:
        # Verify theme exists
        theme = await theme_repo.get_by_id(theme_id)
        if theme is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Theme {theme_id!r} not found",
            )

        today = date.today()
        effective_end = end_date or today
        effective_start = start_date or (today - timedelta(days=30))

        metrics = await theme_repo.get_metrics_range(
            theme_id=theme_id,
            start=effective_start,
            end=effective_end,
        )

        items = [
            ThemeMetricsItem(
                date=m.date,
                document_count=m.document_count,
                sentiment_score=m.sentiment_score,
                volume_zscore=m.volume_zscore,
                velocity=m.velocity,
                acceleration=m.acceleration,
                avg_authority=m.avg_authority,
                bullish_ratio=m.bullish_ratio,
            )
            for m in metrics
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Theme metrics retrieved",
            theme_id=theme_id,
            total=len(items),
            start_date=str(effective_start),
            end_date=str(effective_end),
            latency_ms=round(latency_ms, 2),
        )

        return ThemeMetricsResponse(
            metrics=items,
            total=len(items),
            theme_id=theme_id,
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get theme metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get theme metrics: {str(e)}",
        )


