"""Tests for BacktestVisualizer — file-based output validation.

Each test verifies that plot methods produce valid PNG files without
raising exceptions. Uses pytest tmp_path for isolated output.
"""

from datetime import date
from pathlib import Path

import pytest

from src.backtest.engine import BacktestResults, DailyBacktestResult
from src.backtest.visualization import BacktestVisualizer


@pytest.fixture
def results_with_data() -> BacktestResults:
    """BacktestResults with 10 trading days of synthetic data."""
    daily = []
    base_returns = [0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.04, 0.01, -0.03, 0.025]
    base_scores = [5.0, 4.0, 6.5, 3.5, 5.5, 2.0, 7.0, 4.5, 3.0, 6.0]

    for i, (ret, score) in enumerate(zip(base_returns, base_scores)):
        daily.append(DailyBacktestResult(
            date=date(2025, 6, 2 + i),
            ranked_themes=[{"score": score, "theme_id": f"t{i}"}],
            top_n_tickers=["NVDA", "AMD"],
            top_n_avg_return=ret,
            direction_correct=ret > 0,
            theme_count=10,
        ))

    return BacktestResults(
        run_id="run_test_viz",
        strategy="swing",
        horizon=5,
        top_n=10,
        trading_days=10,
        daily_results=daily,
        hit_rate=0.6,
        mean_return=0.01,
        total_return=0.05,
        max_drawdown=-0.03,
    )


@pytest.fixture
def empty_results() -> BacktestResults:
    """BacktestResults with no daily data."""
    return BacktestResults(
        run_id="run_empty",
        strategy="swing",
        horizon=5,
        top_n=10,
        trading_days=0,
        daily_results=[],
    )


@pytest.fixture
def single_day_results() -> BacktestResults:
    """BacktestResults with a single trading day."""
    return BacktestResults(
        run_id="run_single",
        strategy="position",
        horizon=20,
        top_n=5,
        trading_days=1,
        daily_results=[
            DailyBacktestResult(
                date=date(2025, 6, 2),
                ranked_themes=[{"score": 5.0, "theme_id": "t1"}],
                top_n_tickers=["NVDA"],
                top_n_avg_return=0.03,
                direction_correct=True,
                theme_count=5,
            ),
        ],
    )


@pytest.fixture
def all_none_returns() -> BacktestResults:
    """BacktestResults where every day has None returns."""
    return BacktestResults(
        run_id="run_none",
        strategy="swing",
        horizon=5,
        top_n=10,
        trading_days=3,
        daily_results=[
            DailyBacktestResult(
                date=date(2025, 6, 2 + i),
                theme_count=5,
            )
            for i in range(3)
        ],
    )


@pytest.fixture
def multi_month_results() -> BacktestResults:
    """BacktestResults spanning multiple months for heatmap testing."""
    daily = []
    # 60 trading days across Jan-Mar 2025
    day_num = 0
    for month in range(1, 4):
        for day in range(1, 22):  # ~20 trading days per month
            d = date(2025, month, day)
            if d.weekday() < 5:
                ret = 0.01 * ((-1) ** day_num) + 0.002
                daily.append(DailyBacktestResult(
                    date=d,
                    ranked_themes=[{"score": 4.0 + day_num * 0.1, "theme_id": f"t{day_num}"}],
                    top_n_tickers=["NVDA"],
                    top_n_avg_return=ret,
                    direction_correct=ret > 0,
                    theme_count=10,
                ))
                day_num += 1

    return BacktestResults(
        run_id="run_multi_month",
        strategy="swing",
        horizon=5,
        top_n=10,
        trading_days=len(daily),
        daily_results=daily,
    )


# ── plot_cumulative_returns ────────────────────────────────


class TestPlotCumulativeReturns:
    def test_creates_file(self, results_with_data, tmp_path):
        path = BacktestVisualizer.plot_cumulative_returns(results_with_data, tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_filename(self, results_with_data, tmp_path):
        path = BacktestVisualizer.plot_cumulative_returns(results_with_data, tmp_path)
        assert path.name == "cumulative_returns.png"

    def test_empty_results(self, empty_results, tmp_path):
        path = BacktestVisualizer.plot_cumulative_returns(empty_results, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_single_day(self, single_day_results, tmp_path):
        path = BacktestVisualizer.plot_cumulative_returns(single_day_results, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_all_none_returns(self, all_none_returns, tmp_path):
        path = BacktestVisualizer.plot_cumulative_returns(all_none_returns, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0


# ── plot_drawdown ──────────────────────────────────────────


class TestPlotDrawdown:
    def test_creates_file(self, results_with_data, tmp_path):
        path = BacktestVisualizer.plot_drawdown(results_with_data, tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_filename(self, results_with_data, tmp_path):
        path = BacktestVisualizer.plot_drawdown(results_with_data, tmp_path)
        assert path.name == "drawdown.png"

    def test_empty_results(self, empty_results, tmp_path):
        path = BacktestVisualizer.plot_drawdown(empty_results, tmp_path)
        assert path.exists()

    def test_single_day(self, single_day_results, tmp_path):
        path = BacktestVisualizer.plot_drawdown(single_day_results, tmp_path)
        assert path.exists()

    def test_all_none_returns(self, all_none_returns, tmp_path):
        path = BacktestVisualizer.plot_drawdown(all_none_returns, tmp_path)
        assert path.exists()


# ── plot_score_vs_return ───────────────────────────────────


class TestPlotScoreVsReturn:
    def test_creates_file(self, results_with_data, tmp_path):
        path = BacktestVisualizer.plot_score_vs_return(results_with_data, tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_filename(self, results_with_data, tmp_path):
        path = BacktestVisualizer.plot_score_vs_return(results_with_data, tmp_path)
        assert path.name == "score_vs_return.png"

    def test_empty_results(self, empty_results, tmp_path):
        path = BacktestVisualizer.plot_score_vs_return(empty_results, tmp_path)
        assert path.exists()

    def test_all_none_returns(self, all_none_returns, tmp_path):
        path = BacktestVisualizer.plot_score_vs_return(all_none_returns, tmp_path)
        assert path.exists()

    def test_color_coding(self, results_with_data, tmp_path):
        """Verifies no exception with mixed direction_correct values."""
        path = BacktestVisualizer.plot_score_vs_return(results_with_data, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0


# ── plot_monthly_heatmap ───────────────────────────────────


class TestPlotMonthlyHeatmap:
    def test_creates_file(self, results_with_data, tmp_path):
        path = BacktestVisualizer.plot_monthly_heatmap(results_with_data, tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_filename(self, results_with_data, tmp_path):
        path = BacktestVisualizer.plot_monthly_heatmap(results_with_data, tmp_path)
        assert path.name == "monthly_heatmap.png"

    def test_empty_results(self, empty_results, tmp_path):
        path = BacktestVisualizer.plot_monthly_heatmap(empty_results, tmp_path)
        assert path.exists()

    def test_multi_month_data(self, multi_month_results, tmp_path):
        """Multiple months should produce a proper heatmap grid."""
        path = BacktestVisualizer.plot_monthly_heatmap(multi_month_results, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_all_none_returns(self, all_none_returns, tmp_path):
        path = BacktestVisualizer.plot_monthly_heatmap(all_none_returns, tmp_path)
        assert path.exists()


# ── generate_all ───────────────────────────────────────────


class TestGenerateAll:
    def test_generates_four_charts(self, results_with_data, tmp_path):
        paths = BacktestVisualizer.generate_all(results_with_data, tmp_path)
        assert len(paths) == 4
        expected_names = {
            "cumulative_returns.png",
            "drawdown.png",
            "score_vs_return.png",
            "monthly_heatmap.png",
        }
        assert {p.name for p in paths} == expected_names
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_creates_output_dir(self, results_with_data, tmp_path):
        nested = tmp_path / "nested" / "output"
        paths = BacktestVisualizer.generate_all(results_with_data, nested)
        assert nested.exists()
        assert len(paths) == 4

    def test_empty_results_still_generates(self, empty_results, tmp_path):
        paths = BacktestVisualizer.generate_all(empty_results, tmp_path)
        assert len(paths) == 4
        for p in paths:
            assert p.exists()

    def test_returns_path_objects(self, results_with_data, tmp_path):
        paths = BacktestVisualizer.generate_all(results_with_data, tmp_path)
        for p in paths:
            assert isinstance(p, Path)
