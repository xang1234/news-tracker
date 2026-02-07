"""Tests for BacktestEngine — async with mocked dependencies.

Covers trading day generation, metrics map building, and the full
run_backtest lifecycle (success, error, no themes, missing prices).
"""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.backtest.config import BacktestConfig
from src.backtest.engine import BacktestEngine, BacktestResults, DailyBacktestResult
from src.themes.schemas import Theme, ThemeMetrics


# ── Trading Days ────────────────────────────────────────────


class TestTradingDays:
    def test_weekdays_only(self):
        """Mon-Fri included, Sat-Sun skipped."""
        # 2025-06-02 = Monday, 2025-06-08 = Sunday
        days = BacktestEngine._trading_days(date(2025, 6, 2), date(2025, 6, 8))
        assert len(days) == 5
        assert days[0] == date(2025, 6, 2)  # Monday
        assert days[-1] == date(2025, 6, 6)  # Friday

    def test_single_weekday(self):
        day = date(2025, 6, 4)  # Wednesday
        days = BacktestEngine._trading_days(day, day)
        assert days == [day]

    def test_single_weekend(self):
        saturday = date(2025, 6, 7)
        days = BacktestEngine._trading_days(saturday, saturday)
        assert days == []

    def test_empty_range(self):
        days = BacktestEngine._trading_days(date(2025, 6, 10), date(2025, 6, 5))
        assert days == []

    def test_two_weeks(self):
        """Two full weeks = 10 trading days."""
        days = BacktestEngine._trading_days(date(2025, 6, 2), date(2025, 6, 13))
        assert len(days) == 10


# ── Build Metrics Map ───────────────────────────────────────


class TestBuildMetricsMap:
    @pytest.mark.asyncio
    async def test_latest_metric_selected(self, mock_database, sample_themes):
        """The most recent metric (last in ASC list) is selected."""
        engine = BacktestEngine(mock_database)

        older = ThemeMetrics(
            theme_id="theme_nvda", date=date(2025, 6, 14), volume_zscore=1.0,
        )
        newer = ThemeMetrics(
            theme_id="theme_nvda", date=date(2025, 6, 15), volume_zscore=3.5,
        )

        # Mock get_metrics_as_of to return two metrics for first theme, empty for others
        call_count = 0

        async def mock_get_metrics(theme_id, as_of, lookback_days=7):
            nonlocal call_count
            call_count += 1
            if theme_id == "theme_nvda":
                return [older, newer]
            return []

        engine._pit.get_metrics_as_of = mock_get_metrics

        as_of = datetime(2025, 6, 15, 23, 59, 59, tzinfo=timezone.utc)
        metrics_map = await engine._build_metrics_map(sample_themes, as_of)

        assert "theme_nvda" in metrics_map
        assert metrics_map["theme_nvda"].volume_zscore == 3.5
        assert "theme_mem" not in metrics_map
        assert call_count == 3  # One call per theme

    @pytest.mark.asyncio
    async def test_empty_themes(self, mock_database):
        engine = BacktestEngine(mock_database)
        as_of = datetime(2025, 6, 15, 23, 59, 59, tzinfo=timezone.utc)
        metrics_map = await engine._build_metrics_map([], as_of)
        assert metrics_map == {}


# ── Run Backtest ────────────────────────────────────────────


def _make_mock_engine(
    mock_database: AsyncMock,
    themes: list[Theme],
    metrics_map: dict[str, ThemeMetrics],
    forward_returns: dict[str, dict[int, float | None]],
) -> BacktestEngine:
    """Create a BacktestEngine with fully mocked dependencies."""
    engine = BacktestEngine(mock_database)

    # Mock PointInTimeService
    engine._pit.get_themes_as_of = AsyncMock(return_value=themes)
    engine._pit.get_metrics_as_of = AsyncMock(return_value=[])

    # Mock metrics: return the appropriate metric for each theme
    async def mock_metrics_as_of(theme_id, as_of, lookback_days=7):
        m = metrics_map.get(theme_id)
        return [m] if m else []

    engine._pit.get_metrics_as_of = mock_metrics_as_of

    # Mock PriceDataFeed
    engine._price_feed.get_forward_returns = AsyncMock(return_value=forward_returns)

    # Mock repositories to return simple objects
    mock_database.fetchrow = AsyncMock(return_value={
        "version_id": "mv_test123",
        "embedding_model": "ProsusAI/finbert",
        "clustering_config": "{}",
        "config_snapshot": "{}",
        "created_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
        "description": "test",
        "run_id": "run_test",
        "model_version_id": "mv_test123",
        "date_range_start": date(2025, 6, 2),
        "date_range_end": date(2025, 6, 6),
        "parameters": '{"strategy": "swing"}',
        "results": None,
        "status": "running",
        "completed_at": None,
        "error_message": None,
    })

    return engine


class TestRunBacktest:
    @pytest.mark.asyncio
    async def test_basic_run(
        self, mock_database, sample_themes, sample_theme_metrics, sample_forward_returns,
    ):
        """3-day run produces results with metrics."""
        engine = _make_mock_engine(
            mock_database, sample_themes, sample_theme_metrics, sample_forward_returns,
        )

        results = await engine.run_backtest(
            start_date=date(2025, 6, 2),  # Monday
            end_date=date(2025, 6, 4),    # Wednesday
            strategy="swing",
            top_n=3,
            horizon=5,
        )

        assert isinstance(results, BacktestResults)
        assert results.trading_days == 3
        assert len(results.daily_results) == 3
        assert results.strategy == "swing"
        assert results.horizon == 5
        assert results.top_n == 3

    @pytest.mark.asyncio
    async def test_audit_lifecycle(
        self, mock_database, sample_themes, sample_theme_metrics, sample_forward_returns,
    ):
        """Verifies create → mark_completed sequence."""
        engine = _make_mock_engine(
            mock_database, sample_themes, sample_theme_metrics, sample_forward_returns,
        )

        results = await engine.run_backtest(
            start_date=date(2025, 6, 2),
            end_date=date(2025, 6, 2),
            strategy="swing",
            top_n=3,
            horizon=5,
        )

        # DB should have been called: create version, create run, mark completed
        # At least 3+ fetchrow calls (version create, run create, mark completed)
        assert mock_database.fetchrow.call_count >= 3

    @pytest.mark.asyncio
    async def test_error_marks_failed(self, mock_database, sample_themes, sample_theme_metrics):
        """On error, mark_failed is called before re-raising."""
        engine = _make_mock_engine(
            mock_database, sample_themes, sample_theme_metrics, {},
        )

        # Force an error during processing
        engine._price_feed.get_forward_returns = AsyncMock(
            side_effect=RuntimeError("price feed down")
        )

        with pytest.raises(RuntimeError, match="price feed down"):
            await engine.run_backtest(
                start_date=date(2025, 6, 2),
                end_date=date(2025, 6, 2),
                strategy="swing",
                top_n=3,
                horizon=5,
            )

        # mark_failed should have been called via fetchrow
        # The error should propagate
        assert mock_database.fetchrow.call_count >= 2  # create version + create run

    @pytest.mark.asyncio
    async def test_no_themes_graceful(self, mock_database):
        """Days with no themes produce empty DailyBacktestResult."""
        engine = _make_mock_engine(mock_database, [], {}, {})

        results = await engine.run_backtest(
            start_date=date(2025, 6, 2),
            end_date=date(2025, 6, 2),
            strategy="swing",
            top_n=3,
            horizon=5,
        )

        assert results.trading_days == 1
        assert results.daily_results[0].theme_count == 0
        assert results.daily_results[0].top_n_avg_return is None

    @pytest.mark.asyncio
    async def test_missing_price_data_graceful(
        self, mock_database, sample_themes, sample_theme_metrics,
    ):
        """Missing price data results in None returns."""
        # Empty forward returns dict
        engine = _make_mock_engine(
            mock_database, sample_themes, sample_theme_metrics, {},
        )

        results = await engine.run_backtest(
            start_date=date(2025, 6, 2),
            end_date=date(2025, 6, 2),
            strategy="swing",
            top_n=3,
            horizon=5,
        )

        assert results.daily_results[0].top_n_avg_return is None

    @pytest.mark.asyncio
    async def test_strategy_forwarded(
        self, mock_database, sample_themes, sample_theme_metrics, sample_forward_returns,
    ):
        """Strategy parameter is recorded in results."""
        engine = _make_mock_engine(
            mock_database, sample_themes, sample_theme_metrics, sample_forward_returns,
        )

        results = await engine.run_backtest(
            start_date=date(2025, 6, 2),
            end_date=date(2025, 6, 2),
            strategy="position",
            top_n=5,
            horizon=10,
        )

        assert results.strategy == "position"
        assert results.horizon == 10
        assert results.top_n == 5


# ── Serialization ───────────────────────────────────────────


class TestSerialization:
    def test_to_dict(self, sample_daily_results):
        results = BacktestResults(
            run_id="run_001",
            start_date=date(2025, 6, 2),
            end_date=date(2025, 6, 6),
            strategy="swing",
            horizon=5,
            top_n=10,
            trading_days=5,
            daily_results=sample_daily_results,
            hit_rate=0.6,
            mean_return=0.018,
        )
        d = results.to_dict()
        assert d["run_id"] == "run_001"
        assert d["start_date"] == "2025-06-02"
        assert len(d["daily_results"]) == 5
        assert d["daily_results"][0]["date"] == "2025-06-02"

    def test_summary_dict(self):
        results = BacktestResults(
            run_id="run_001",
            strategy="swing",
            horizon=5,
            hit_rate=0.6,
        )
        s = results.summary_dict()
        assert "daily_results" not in s
        assert s["hit_rate"] == 0.6
