"""
Theme endpoints for dashboard, trading system, and alert consumption.
"""

import asyncio
import time
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.requests import Request

from src.api.auth import verify_api_key
from src.api.dependencies import (
    get_document_repository,
    get_graph_repository,
    get_narrative_repository,
    get_propagation_service,
    get_ranking_service,
    get_sentiment_aggregator,
    get_theme_repository,
)
from src.api.models import (
    ErrorResponse,
    MarketCatalystEvidenceItem,
    MarketCatalystItem,
    MarketCatalystRelatedTickerItem,
    MarketCatalystsResponse,
    MarketCatalystTickerItem,
    NarrativeAlertSummary,
    NarrativeBucketPoint,
    NarrativeDocumentItem,
    NarrativeMomentumResponse,
    NarrativeRunDetailResponse,
    NarrativeRunDocumentsResponse,
    NarrativeRunItem,
    NarrativeTickerCount,
    RankedThemeItem,
    RankedThemesResponse,
    ThemeDetailResponse,
    ThemeDocumentItem,
    ThemeDocumentsResponse,
    ThemeItem,
    ThemeListResponse,
    ThemeMetricsItem,
    ThemeMetricsResponse,
    ThemeNarrativesResponse,
    ThemeSentimentResponse,
)
from src.api.rate_limit import limiter
from src.config.settings import get_settings as _get_settings
from src.event_extraction.theme_integration import EventThemeLinker, ThemeWithEvents
from src.graph.propagation import SentimentPropagation
from src.graph.storage import GraphRepository
from src.narrative.repository import NarrativeRepository
from src.sentiment.aggregation import DocumentSentiment, SentimentAggregator
from src.storage.repository import DocumentRepository
from src.themes.catalysts import (
    compute_market_impact_score,
    dominant_event_types,
    humanize_identifier,
    infer_market_bias,
    propagation_delta,
    summarize_market_catalyst,
)
from src.themes.ranking import ThemeRankingService
from src.themes.repository import ThemeRepository
from src.themes.schemas import Theme

logger = structlog.get_logger(__name__)
router = APIRouter()

CATALYST_BUILD_CONCURRENCY = 10


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


async def _run_to_item(
    narrative_repo: NarrativeRepository,
    run,
    theme_name: str | None = None,
) -> NarrativeRunItem:
    recent_alerts = await narrative_repo.get_recent_alerts(run.run_id, limit=3)
    buckets = await narrative_repo.get_recent_buckets(run.run_id, limit=12)
    ordered_tickers = sorted(
        run.ticker_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return NarrativeRunItem(
        run_id=run.run_id,
        theme_id=run.theme_id,
        theme_name=theme_name,
        status=run.status,
        label=run.label,
        conviction_score=round(run.conviction_score, 2),
        current_rate_per_hour=round(run.current_rate_per_hour, 3),
        current_acceleration=round(run.current_acceleration, 3),
        platform_count=run.platform_count,
        top_tickers=[
            NarrativeTickerCount(ticker=ticker, count=count)
            for ticker, count in ordered_tickers[:5]
        ],
        last_document_at=run.last_document_at.isoformat(),
        started_at=run.started_at.isoformat(),
        recent_alerts=[
            NarrativeAlertSummary(
                alert_id=alert["alert_id"],
                trigger_type=alert["trigger_type"],
                severity=alert["severity"],
                conviction_score=alert.get("conviction_score"),
                title=alert["title"],
                created_at=alert["created_at"].isoformat(),
            )
            for alert in recent_alerts
        ],
        sparkline=[
            NarrativeBucketPoint(
                bucket_start=bucket.bucket_start.isoformat(),
                doc_count=bucket.doc_count,
            )
            for bucket in buckets
        ],
    )


def _narrative_document_item(doc: dict) -> NarrativeDocumentItem:
    sentiment = doc.get("sentiment") or {}
    return NarrativeDocumentItem(
        document_id=doc["document_id"],
        platform=doc.get("platform"),
        title=doc.get("title"),
        content_preview=doc.get("content_preview"),
        url=doc.get("url"),
        author_name=doc.get("author_name"),
        tickers=doc.get("tickers") or [],
        authority_score=doc.get("authority_score"),
        sentiment_label=sentiment.get("label"),
        sentiment_confidence=sentiment.get("confidence"),
        similarity=round(float(doc.get("similarity") or 0.0), 4),
        timestamp=doc["timestamp"].isoformat() if doc.get("timestamp") else None,
    )


def _catalyst_evidence_item(doc: dict) -> MarketCatalystEvidenceItem:
    sentiment = doc.get("sentiment") or {}
    return MarketCatalystEvidenceItem(
        document_id=doc["document_id"],
        platform=doc.get("platform"),
        title=doc.get("title"),
        url=doc.get("url"),
        author_name=doc.get("author_name"),
        authority_score=doc.get("authority_score"),
        sentiment_label=sentiment.get("label"),
        sentiment_confidence=sentiment.get("confidence"),
        timestamp=doc["timestamp"].isoformat() if doc.get("timestamp") else None,
    )


def _graph_node_id(node: object) -> str | None:
    value = node.get("node_id") if isinstance(node, dict) else getattr(node, "node_id", None)
    return value if isinstance(value, str) else None


def _graph_node_type(node: object) -> str | None:
    value = node.get("node_type") if isinstance(node, dict) else getattr(node, "node_type", None)
    return value if isinstance(value, str) else None


def _descending_timestamp_sort_key(timestamp: datetime | None) -> float:
    """Return a sort key that prefers newer timestamps and pushes missing values last."""
    if timestamp is None:
        return float("inf")
    return -timestamp.timestamp()


class _TickerNodeLookup:
    """Resolve candidate ticker nodes on demand and cache the results for the request."""

    def __init__(self, graph_repo: GraphRepository) -> None:
        self._graph_repo = graph_repo
        self._cache: dict[str, bool] = {}
        self._logged_error = False

    async def filter_tickers(self, node_ids: Iterable[str]) -> set[str]:
        candidate_ids = {node_id for node_id in node_ids if node_id}
        unresolved = sorted(node_id for node_id in candidate_ids if node_id not in self._cache)
        if unresolved:
            results = await asyncio.gather(
                *[self._graph_repo.get_node(node_id) for node_id in unresolved],
                return_exceptions=True,
            )
            for node_id, result in zip(unresolved, results, strict=True):
                if isinstance(result, Exception):
                    self._cache[node_id] = False
                    if not self._logged_error:
                        logger.warning("catalyst_graph_nodes_unavailable", error=str(result))
                        self._logged_error = True
                    continue
                self._cache[node_id] = _graph_node_type(result) == "ticker"
        return {node_id for node_id in candidate_ids if self._cache.get(node_id, False)}


async def _build_market_catalyst_bounded(
    semaphore: asyncio.Semaphore,
    **kwargs,
) -> MarketCatalystItem | None:
    """Build a catalyst while respecting the route-level concurrency ceiling."""
    async with semaphore:
        return await _build_market_catalyst(**kwargs)


async def _build_market_catalyst(
    *,
    run,
    theme_name: str | None,
    days: int,
    narrative_repo: NarrativeRepository,
    theme_repo: ThemeRepository,
    doc_repo: DocumentRepository,
    propagation: SentimentPropagation,
    ticker_node_lookup: _TickerNodeLookup,
) -> MarketCatalystItem | None:
    theme = await theme_repo.get_by_id(run.theme_id)
    if theme is None:
        return None

    now = datetime.now(UTC)
    today = now.date()
    metrics = await theme_repo.get_metrics_range(
        run.theme_id,
        start=today - timedelta(days=days),
        end=today,
    )
    latest_metric = metrics[-1] if metrics else None

    since = now - timedelta(days=days)

    run_documents = await narrative_repo.get_run_documents(run.run_id, limit=8)
    raw_events = await doc_repo.get_events_by_tickers(
        tickers=theme.top_tickers,
        since=since,
        until=now,
        limit=60,
    )

    linked_events = EventThemeLinker.link_events_to_theme(raw_events, theme)
    deduped_events = EventThemeLinker.deduplicate_events(linked_events)
    event_summary = ThemeWithEvents.from_events(theme.theme_id, deduped_events)
    investment_signal = event_summary.investment_signal()
    bias = infer_market_bias(run.avg_sentiment, investment_signal)

    ordered_tickers = sorted(
        run.ticker_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    primary_tickers = [
        MarketCatalystTickerItem(ticker=ticker, mention_count=count)
        for ticker, count in ordered_tickers[:3]
    ]
    primary_ticker_ids = {item.ticker for item in primary_tickers}

    related_tickers: list[MarketCatalystRelatedTickerItem] = []
    delta = propagation_delta(run.avg_sentiment, bias)
    if delta != 0.0 and primary_tickers:
        propagation_results = await asyncio.gather(
            *[
                propagation.propagate(source_node=item.ticker, sentiment_delta=delta)
                for item in primary_tickers
            ],
            return_exceptions=True,
        )
        candidate_node_ids = {
            impact.node_id
            for result in propagation_results
            if not isinstance(result, Exception)
            for impact in result.values()
        }
        ticker_node_ids = await ticker_node_lookup.filter_tickers(candidate_node_ids)
        best_impacts: dict[str, MarketCatalystRelatedTickerItem] = {}
        for source_item, result in zip(primary_tickers, propagation_results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    "catalyst_propagation_failed",
                    run_id=run.run_id,
                    source_ticker=source_item.ticker,
                    error=str(result),
                )
                continue

            impacts = sorted(
                result.values(),
                key=lambda impact: abs(impact.impact),
                reverse=True,
            )
            for impact in impacts:
                if impact.node_id not in ticker_node_ids:
                    continue
                if impact.node_id == source_item.ticker or impact.node_id in primary_ticker_ids:
                    continue

                candidate = MarketCatalystRelatedTickerItem(
                    ticker=impact.node_id,
                    source_ticker=source_item.ticker,
                    impact=round(impact.impact, 4),
                    depth=impact.depth,
                    relation=impact.path_relation,
                )
                existing = best_impacts.get(candidate.ticker)
                if existing is None or abs(candidate.impact) > abs(existing.impact):
                    best_impacts[candidate.ticker] = candidate

        related_tickers = sorted(
            best_impacts.values(),
            key=lambda item: abs(item.impact),
            reverse=True,
        )[:5]

    dominant_events = dominant_event_types(event_summary.event_counts)
    impact_score = compute_market_impact_score(
        conviction_score=run.conviction_score,
        volume_zscore=latest_metric.volume_zscore if latest_metric else None,
        acceleration=run.current_acceleration,
        platform_count=run.platform_count,
        avg_authority=latest_metric.avg_authority if latest_metric else run.avg_authority,
        event_count=sum(event_summary.event_counts.values()),
    )

    supporting_docs = sorted(
        run_documents,
        key=lambda doc: (
            0 if doc.get("title") else 1,
            -(doc.get("authority_score") or 0.0),
            _descending_timestamp_sort_key(doc.get("timestamp")),
        ),
    )
    evidence = [_catalyst_evidence_item(doc) for doc in supporting_docs[:3]]

    display_theme_name = humanize_identifier(theme_name or theme.name)
    summary = summarize_market_catalyst(
        theme_name=display_theme_name,
        bias=bias,
        primary_tickers=[item.ticker for item in primary_tickers],
        investment_signal=investment_signal,
        dominant_events=dominant_events,
        platform_count=run.platform_count,
        volume_zscore=latest_metric.volume_zscore if latest_metric else None,
        conviction_score=run.conviction_score,
        related_tickers=[item.ticker for item in related_tickers],
    )
    avg_authority_value = (
        latest_metric.avg_authority
        if latest_metric and latest_metric.avg_authority is not None
        else run.avg_authority
    )

    return MarketCatalystItem(
        run_id=run.run_id,
        theme_id=run.theme_id,
        theme_name=display_theme_name,
        lifecycle_stage=theme.lifecycle_stage,
        bias=bias,
        summary=summary,
        market_impact_score=impact_score,
        conviction_score=round(run.conviction_score, 2),
        current_rate_per_hour=round(run.current_rate_per_hour, 3),
        current_acceleration=round(run.current_acceleration, 3),
        platform_count=run.platform_count,
        avg_sentiment=round(run.avg_sentiment, 4) if run.avg_sentiment is not None else None,
        avg_authority=round(avg_authority_value, 4) if avg_authority_value is not None else None,
        volume_zscore=(
            round(latest_metric.volume_zscore, 4)
            if latest_metric and latest_metric.volume_zscore is not None
            else None
        ),
        investment_signal=investment_signal,
        dominant_event_types=dominant_events,
        primary_tickers=primary_tickers,
        related_tickers=related_tickers,
        evidence=evidence,
        started_at=run.started_at.isoformat(),
        last_document_at=run.last_document_at.isoformat(),
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
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def list_themes(
    request: Request,
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
        logger.error("list_themes_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list themes",
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
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def get_ranked_themes(
    request: Request,
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
        logger.error("rank_themes_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rank themes",
        )


@router.get(
    "/themes/momentum",
    response_model=NarrativeMomentumResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get global narrative momentum feed",
    description="List the hottest narrative runs across all themes.",
)
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def get_theme_momentum(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100, description="Maximum runs to return"),
    api_key: str = Depends(verify_api_key),
    narrative_repo: NarrativeRepository = Depends(get_narrative_repository),
) -> NarrativeMomentumResponse:
    start_time = time.perf_counter()
    rows = await narrative_repo.list_global_momentum(limit=limit)
    items = [await _run_to_item(narrative_repo, row["run"], row["theme_name"]) for row in rows]
    latency_ms = (time.perf_counter() - start_time) * 1000
    return NarrativeMomentumResponse(
        runs=items,
        total=len(items),
        latency_ms=round(latency_ms, 2),
    )


@router.get(
    "/themes/catalysts",
    response_model=MarketCatalystsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get stock-market catalyst radar",
    description=(
        "Translate live narratives into tradable stock-market catalysts. "
        "Each item combines conviction, event corroboration, supporting "
        "evidence, and graph-based follow-on tickers."
    ),
)
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def get_market_catalysts(
    request: Request,
    limit: int = Query(default=10, ge=1, le=50, description="Maximum catalysts to return"),
    days: int = Query(
        default=7,
        ge=1,
        le=30,
        description="Lookback window for event corroboration",
    ),
    api_key: str = Depends(verify_api_key),
    narrative_repo: NarrativeRepository = Depends(get_narrative_repository),
    theme_repo: ThemeRepository = Depends(get_theme_repository),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    graph_repo: GraphRepository = Depends(get_graph_repository),
    propagation: SentimentPropagation = Depends(get_propagation_service),
) -> MarketCatalystsResponse:
    start_time = time.perf_counter()

    fetch_limit = min(max(limit * 3, 20), 100)
    rows = await narrative_repo.list_global_momentum(limit=fetch_limit)
    ticker_node_lookup = _TickerNodeLookup(graph_repo)

    semaphore = asyncio.Semaphore(min(CATALYST_BUILD_CONCURRENCY, max(len(rows), 1)))
    catalysts = await asyncio.gather(
        *[
            _build_market_catalyst_bounded(
                semaphore=semaphore,
                run=row["run"],
                theme_name=row.get("theme_name"),
                days=days,
                narrative_repo=narrative_repo,
                theme_repo=theme_repo,
                doc_repo=doc_repo,
                propagation=propagation,
                ticker_node_lookup=ticker_node_lookup,
            )
            for row in rows
        ],
        return_exceptions=True,
    )

    items: list[MarketCatalystItem] = []
    for result in catalysts:
        if isinstance(result, Exception):
            logger.warning("build_market_catalyst_failed", error=str(result))
            continue
        if result is not None:
            items.append(result)

    items.sort(key=lambda item: item.market_impact_score, reverse=True)
    items = items[:limit]

    latency_ms = (time.perf_counter() - start_time) * 1000
    return MarketCatalystsResponse(
        catalysts=items,
        total=len(items),
        latency_ms=round(latency_ms, 2),
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
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def get_theme(
    request: Request,
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
        logger.error("get_theme_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get theme",
        )


@router.get(
    "/themes/{theme_id}/narratives",
    response_model=ThemeNarrativesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Theme not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get narrative runs for a theme",
    description="List narrative momentum runs under a theme.",
)
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def get_theme_narratives(
    request: Request,
    theme_id: str,
    run_status: str | None = Query(default=None, description="Optional run status filter"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum runs to return"),
    api_key: str = Depends(verify_api_key),
    theme_repo: ThemeRepository = Depends(get_theme_repository),
    narrative_repo: NarrativeRepository = Depends(get_narrative_repository),
) -> ThemeNarrativesResponse:
    start_time = time.perf_counter()

    theme = await theme_repo.get_by_id(theme_id)
    if theme is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Theme {theme_id!r} not found",
        )

    runs = await narrative_repo.list_theme_runs(theme_id=theme_id, status=run_status, limit=limit)
    items = [await _run_to_item(narrative_repo, run, theme.name) for run in runs]
    latency_ms = (time.perf_counter() - start_time) * 1000
    return ThemeNarrativesResponse(
        theme_id=theme_id,
        runs=items,
        total=len(items),
        latency_ms=round(latency_ms, 2),
    )


@router.get(
    "/themes/{theme_id}/narratives/{run_id}",
    response_model=NarrativeRunDetailResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Theme or run not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get narrative run detail",
    description="Get a narrative run, its timeline, and recent documents.",
)
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def get_theme_narrative_detail(
    request: Request,
    theme_id: str,
    run_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Maximum documents to return"),
    api_key: str = Depends(verify_api_key),
    theme_repo: ThemeRepository = Depends(get_theme_repository),
    narrative_repo: NarrativeRepository = Depends(get_narrative_repository),
) -> NarrativeRunDetailResponse:
    start_time = time.perf_counter()

    theme = await theme_repo.get_by_id(theme_id)
    if theme is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Theme {theme_id!r} not found",
        )

    run = await narrative_repo.get_by_id(run_id)
    if run is None or run.theme_id != theme_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Narrative run {run_id!r} not found for theme {theme_id!r}",
        )

    run_item = await _run_to_item(narrative_repo, run, theme.name)
    docs = await narrative_repo.get_run_documents(run_id, limit=limit)
    items = [_narrative_document_item(doc) for doc in docs]

    latency_ms = (time.perf_counter() - start_time) * 1000
    return NarrativeRunDetailResponse(
        run=run_item,
        platform_timeline=run.platform_first_seen,
        ticker_counts=run.ticker_counts,
        documents=items,
        latency_ms=round(latency_ms, 2),
    )


@router.get(
    "/themes/{theme_id}/narratives/{run_id}/documents",
    response_model=NarrativeRunDocumentsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Theme or run not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get documents for a narrative run",
    description="List recent documents assigned to a narrative run.",
)
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def get_theme_narrative_documents(
    request: Request,
    theme_id: str,
    run_id: str,
    limit: int = Query(default=50, ge=1, le=200, description="Maximum documents to return"),
    api_key: str = Depends(verify_api_key),
    theme_repo: ThemeRepository = Depends(get_theme_repository),
    narrative_repo: NarrativeRepository = Depends(get_narrative_repository),
) -> NarrativeRunDocumentsResponse:
    start_time = time.perf_counter()

    theme = await theme_repo.get_by_id(theme_id)
    if theme is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Theme {theme_id!r} not found",
        )

    run = await narrative_repo.get_by_id(run_id)
    if run is None or run.theme_id != theme_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Narrative run {run_id!r} not found for theme {theme_id!r}",
        )

    docs = await narrative_repo.get_run_documents(run_id, limit=limit)
    items = [_narrative_document_item(doc) for doc in docs]
    latency_ms = (time.perf_counter() - start_time) * 1000
    return NarrativeRunDocumentsResponse(
        theme_id=theme_id,
        run_id=run_id,
        documents=items,
        total=len(items),
        latency_ms=round(latency_ms, 2),
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
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def get_theme_documents(
    request: Request,
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
        logger.error("get_theme_documents_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get theme documents",
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
@limiter.limit(lambda: _get_settings().rate_limit_sentiment)
async def get_theme_sentiment(
    request: Request,
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

        now = datetime.now(UTC)
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
        logger.error("aggregate_theme_sentiment_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to aggregate theme sentiment",
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
@limiter.limit(lambda: _get_settings().rate_limit_default)
async def get_theme_metrics(
    request: Request,
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
        logger.error("get_theme_metrics_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get theme metrics",
        )
