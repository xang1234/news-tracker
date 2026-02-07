"""Backtest simulation engine.

Iterates over historical trading days, ranks themes using point-in-time
data (no look-ahead bias), collects tickers from top themes, measures
forward returns via PriceDataFeed, and computes aggregate performance
metrics via BacktestMetrics.

In-memory results are returned to the caller; only the summary is
persisted to ``backtest_runs.results`` JSONB via BacktestRunRepository.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.backtest.audit import BacktestRun, BacktestRunRepository
from src.backtest.config import BacktestConfig
from src.backtest.data_feeds import PriceDataFeed
from src.backtest.metrics import BacktestMetrics, CalibrationBucket
from src.backtest.model_versions import (
    ModelVersionRepository,
    create_version_from_settings,
)
from src.backtest.point_in_time import PointInTimeService
from src.storage.database import Database
from src.themes.ranking import RankingStrategy, ThemeRankingService
from src.themes.repository import ThemeRepository
from src.themes.schemas import ThemeMetrics as ThemeMetricsSchema

logger = logging.getLogger(__name__)


@dataclass
class DailyBacktestResult:
    """Results for a single trading day in the backtest.

    Attributes:
        date: The trading date.
        ranked_themes: Serialised snapshots of ranked themes for this day.
        ticker_returns: Per-ticker forward returns at all horizons.
        top_n_tickers: Tickers selected from the top-N themes.
        top_n_avg_return: Average forward return of top-N tickers at the
            selected horizon. None if no price data available.
        direction_correct: Whether average return was positive. None if
            no return data.
        theme_count: Number of active themes on this day.
    """

    date: date
    ranked_themes: list[dict[str, Any]] = field(default_factory=list)
    ticker_returns: dict[str, dict[int, float | None]] = field(
        default_factory=dict
    )
    top_n_tickers: list[str] = field(default_factory=list)
    top_n_avg_return: float | None = None
    direction_correct: bool | None = None
    theme_count: int = 0


@dataclass
class BacktestResults:
    """Complete results for a backtest run.

    Contains run metadata, daily results, and aggregate metrics. The
    ``to_dict()`` method produces a JSON-serialisable representation
    for storage; ``summary_dict()`` produces a compact human-readable
    version for CLI output.
    """

    # Run metadata
    run_id: str = ""
    model_version_id: str = ""
    start_date: date | None = None
    end_date: date | None = None
    strategy: str = "swing"
    horizon: int = 5
    top_n: int = 10
    trading_days: int = 0

    # Daily results
    daily_results: list[DailyBacktestResult] = field(default_factory=list)

    # Aggregate metrics
    hit_rate: float | None = None
    mean_return: float | None = None
    total_return: float | None = None
    volatility: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    calibration: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-serialisable representation."""
        return {
            "run_id": self.run_id,
            "model_version_id": self.model_version_id,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "strategy": self.strategy,
            "horizon": self.horizon,
            "top_n": self.top_n,
            "trading_days": self.trading_days,
            "hit_rate": self.hit_rate,
            "mean_return": self.mean_return,
            "total_return": self.total_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calibration": self.calibration,
            "daily_results": [
                {
                    "date": dr.date.isoformat(),
                    "theme_count": dr.theme_count,
                    "top_n_tickers": dr.top_n_tickers,
                    "top_n_avg_return": dr.top_n_avg_return,
                    "direction_correct": dr.direction_correct,
                }
                for dr in self.daily_results
            ],
        }

    def summary_dict(self) -> dict[str, Any]:
        """Compact summary for CLI output (no daily detail)."""
        return {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "horizon": self.horizon,
            "top_n": self.top_n,
            "trading_days": self.trading_days,
            "hit_rate": self.hit_rate,
            "mean_return": self.mean_return,
            "total_return": self.total_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calibration": self.calibration,
        }


class BacktestEngine:
    """Orchestrates backtest simulation over historical trading days.

    Ties together PointInTimeService (temporal theme queries),
    ThemeRankingService (scoring), PriceDataFeed (forward returns),
    and BacktestMetrics (aggregate performance).

    Args:
        database: Async database connection.
        config: Optional BacktestConfig (defaults created if omitted).
    """

    def __init__(
        self,
        database: Database,
        config: BacktestConfig | None = None,
    ) -> None:
        self._db = database
        self._config = config or BacktestConfig()
        self._theme_repo = ThemeRepository(database)
        self._pit = PointInTimeService(database, self._theme_repo)
        self._price_feed = PriceDataFeed(database, self._config)
        self._ranking = ThemeRankingService(theme_repo=self._theme_repo)
        self._run_repo = BacktestRunRepository(database)
        self._version_repo = ModelVersionRepository(database)

    async def run_backtest(
        self,
        start_date: date,
        end_date: date,
        strategy: RankingStrategy = "swing",
        top_n: int = 10,
        horizon: int = 5,
    ) -> BacktestResults:
        """Execute a full backtest simulation.

        Algorithm:
        1. Snapshot model version and create audit record.
        2. For each trading day (Mon-Fri):
           a. Get themes as-of end-of-day UTC.
           b. Fetch metrics for each theme, build metrics map.
           c. Rank themes, take top N.
           d. Collect unique tickers from top themes.
           e. Get forward returns from PriceDataFeed.
           f. Compute average return and direction correctness.
        3. Compute aggregate metrics via BacktestMetrics.
        4. Persist summary to BacktestRunRepository.

        Args:
            start_date: First day of the backtest window (inclusive).
            end_date: Last day of the backtest window (inclusive).
            strategy: Ranking strategy ("swing" or "position").
            top_n: Number of top-ranked themes to select each day.
            horizon: Forward return horizon in trading days.

        Returns:
            BacktestResults with daily detail and aggregate metrics.
        """
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        # Step 1: Snapshot model version
        model_version = create_version_from_settings(
            embedding_model=self._config.yfinance_rate_limit.__class__.__name__,
            clustering_config={},
            description=f"Backtest run {run_id}",
        )
        # Use a simpler approach — capture current config as the version
        model_version = create_version_from_settings(
            embedding_model="ProsusAI/finbert",
            clustering_config={
                "backtest_lookback_days": self._config.backtest_lookback_days,
                "forward_horizons": self._config.default_forward_horizons,
            },
            description=f"Backtest run {run_id}",
        )
        model_version = await self._version_repo.create(model_version)

        # Create audit record
        bt_run = BacktestRun(
            run_id=run_id,
            model_version_id=model_version.version_id,
            date_range_start=start_date,
            date_range_end=end_date,
            parameters={
                "strategy": strategy,
                "top_n": top_n,
                "horizon": horizon,
                "lookback_days": self._config.backtest_lookback_days,
            },
        )
        await self._run_repo.create(bt_run)

        try:
            # Step 2: Iterate trading days
            trading_days = self._trading_days(start_date, end_date)
            daily_results: list[DailyBacktestResult] = []

            for i, day in enumerate(trading_days):
                if i > 0 and i % 50 == 0:
                    logger.info(
                        "Backtest progress: day %d/%d (%s)",
                        i, len(trading_days), day,
                    )

                result = await self._process_day(
                    day, strategy, top_n, horizon,
                )
                daily_results.append(result)

            # Step 3: Compute metrics
            metrics = BacktestMetrics.compute(
                daily_results,
                horizon=horizon,
                n_calibration_buckets=self._config.backtest_calibration_buckets,
            )

            # Step 4: Build results
            results = BacktestResults(
                run_id=run_id,
                model_version_id=model_version.version_id,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                horizon=horizon,
                top_n=top_n,
                trading_days=len(trading_days),
                daily_results=daily_results,
                hit_rate=metrics.get("hit_rate"),
                mean_return=metrics.get("mean_return"),
                total_return=metrics.get("total_return"),
                volatility=metrics.get("volatility"),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                sortino_ratio=metrics.get("sortino_ratio"),
                max_drawdown=metrics.get("max_drawdown"),
                win_rate=metrics.get("win_rate"),
                profit_factor=metrics.get("profit_factor"),
                calibration=metrics.get("calibration", []),
            )

            # Step 5: Persist summary
            await self._run_repo.mark_completed(
                run_id, results.summary_dict()
            )

            logger.info(
                "Backtest complete: %d trading days, mean return %.4f",
                len(trading_days),
                results.mean_return or 0.0,
            )

            return results

        except Exception:
            # Mark run as failed on any error
            try:
                import traceback
                error_msg = traceback.format_exc()
                await self._run_repo.mark_failed(run_id, error_msg[:2000])
            except Exception:
                logger.exception("Failed to mark backtest run as failed")
            raise

    async def _process_day(
        self,
        day: date,
        strategy: RankingStrategy,
        top_n: int,
        horizon: int,
    ) -> DailyBacktestResult:
        """Process a single trading day.

        Args:
            day: The trading date.
            strategy: Ranking strategy.
            top_n: Number of top themes to select.
            horizon: Forward return horizon.

        Returns:
            DailyBacktestResult for this day.
        """
        # End of day UTC as the "as-of" time
        as_of = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)

        # Get themes active at this point in time
        themes = await self._pit.get_themes_as_of(as_of)

        if not themes:
            return DailyBacktestResult(date=day, theme_count=0)

        # Build metrics map for ranking
        metrics_map = await self._build_metrics_map(themes, as_of)

        # Rank themes
        ranked = self._ranking.rank_themes(themes, metrics_map, strategy)
        top_ranked = ranked[:top_n]

        if not top_ranked:
            return DailyBacktestResult(
                date=day,
                theme_count=len(themes),
            )

        # Serialise ranked themes for storage
        ranked_snapshots = [
            {
                "theme_id": rt.theme_id,
                "score": round(rt.score, 6),
                "tier": rt.tier,
                "name": rt.theme.name,
                "lifecycle_stage": rt.theme.lifecycle_stage,
                "top_tickers": rt.theme.top_tickers[:3],
            }
            for rt in top_ranked
        ]

        # Collect unique tickers from top themes (top 3 per theme, max 20)
        tickers: list[str] = []
        seen: set[str] = set()
        for rt in top_ranked:
            for ticker in rt.theme.top_tickers[:3]:
                if ticker not in seen and len(tickers) < 20:
                    tickers.append(ticker)
                    seen.add(ticker)

        if not tickers:
            return DailyBacktestResult(
                date=day,
                ranked_themes=ranked_snapshots,
                theme_count=len(themes),
            )

        # Get forward returns
        fwd_returns = await self._price_feed.get_forward_returns(
            tickers, day,
        )

        # Compute average return at selected horizon
        horizon_returns = []
        for ticker in tickers:
            ret = fwd_returns.get(ticker, {}).get(horizon)
            if ret is not None:
                horizon_returns.append(ret)

        top_n_avg_return: float | None = None
        direction_correct: bool | None = None

        if horizon_returns:
            top_n_avg_return = sum(horizon_returns) / len(horizon_returns)
            direction_correct = top_n_avg_return > 0

        return DailyBacktestResult(
            date=day,
            ranked_themes=ranked_snapshots,
            ticker_returns=fwd_returns,
            top_n_tickers=tickers,
            top_n_avg_return=top_n_avg_return,
            direction_correct=direction_correct,
            theme_count=len(themes),
        )

    async def _build_metrics_map(
        self,
        themes: list,
        as_of: datetime,
    ) -> dict[str, ThemeMetricsSchema]:
        """Build a theme_id → latest ThemeMetrics map.

        For each theme, fetches metrics within the lookback window and
        selects the most recent entry. This uses point-in-time data to
        prevent look-ahead bias.

        Args:
            themes: Themes active at as_of.
            as_of: Point in time for the query.

        Returns:
            Dict mapping theme_id to the latest ThemeMetrics.
        """
        metrics_map: dict[str, ThemeMetricsSchema] = {}

        for theme in themes:
            metrics_list = await self._pit.get_metrics_as_of(
                theme.theme_id,
                as_of,
                lookback_days=self._config.backtest_lookback_days,
            )
            if metrics_list:
                # Most recent entry (list is ordered by date ASC)
                metrics_map[theme.theme_id] = metrics_list[-1]

        return metrics_map

    @staticmethod
    def _trading_days(start: date, end: date) -> list[date]:
        """Generate weekday dates between start and end (inclusive).

        Skips weekends (Saturday=5, Sunday=6). Does not account for
        market holidays — this is sufficient for a backtest that
        gracefully handles missing price data.

        Args:
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of weekday dates in chronological order.
        """
        if start > end:
            return []
        days: list[date] = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Mon=0 ... Fri=4
                days.append(current)
            current += timedelta(days=1)
        return days
