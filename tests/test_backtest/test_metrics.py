"""Tests for BacktestMetrics — all synchronous, no mocks needed.

Covers return extraction, directional accuracy, return statistics,
volatility, Sharpe, Sortino, max drawdown, win rate, profit factor,
and calibration buckets.
"""

from datetime import date

import pytest

from src.backtest.engine import DailyBacktestResult
from src.backtest.metrics import BacktestMetrics, CalibrationBucket


# ── _extract_returns ────────────────────────────────────────


class TestExtractReturns:
    def test_filters_none(self, sample_daily_results):
        """Non-None returns are extracted; None is skipped."""
        returns = BacktestMetrics._extract_returns(sample_daily_results)
        assert len(returns) == 5
        assert all(isinstance(r, float) for r in returns)

    def test_all_none(self):
        """Returns empty list when all returns are None."""
        results = [
            DailyBacktestResult(date=date(2025, 1, 1), top_n_avg_return=None),
            DailyBacktestResult(date=date(2025, 1, 2), top_n_avg_return=None),
        ]
        assert BacktestMetrics._extract_returns(results) == []

    def test_empty_input(self):
        assert BacktestMetrics._extract_returns([]) == []


# ── _directional_accuracy ───────────────────────────────────


class TestDirectionalAccuracy:
    def test_mixed_directions(self, sample_daily_results):
        """3 correct out of 5 = 0.6."""
        accuracy = BacktestMetrics._directional_accuracy(sample_daily_results)
        assert accuracy == pytest.approx(0.6)

    def test_all_correct(self):
        results = [
            DailyBacktestResult(date=date(2025, 1, i), direction_correct=True)
            for i in range(1, 4)
        ]
        assert BacktestMetrics._directional_accuracy(results) == 1.0

    def test_all_none(self):
        results = [
            DailyBacktestResult(date=date(2025, 1, i), direction_correct=None)
            for i in range(1, 4)
        ]
        assert BacktestMetrics._directional_accuracy(results) is None

    def test_empty(self):
        assert BacktestMetrics._directional_accuracy([]) is None


# ── _return_metrics ─────────────────────────────────────────


class TestReturnMetrics:
    def test_basic_stats(self):
        returns = [0.01, 0.02, -0.01, 0.03, -0.005]
        result = BacktestMetrics._return_metrics(returns)
        assert result["mean"] == pytest.approx(0.009, abs=1e-6)
        assert result["max"] == pytest.approx(0.03)
        assert result["min"] == pytest.approx(-0.01)
        # Cumulative: (1.01)*(1.02)*(0.99)*(1.03)*(0.995) - 1
        assert "total" in result

    def test_cumulative_return(self):
        """Product of (1+r) - 1."""
        returns = [0.10, -0.05, 0.02]
        result = BacktestMetrics._return_metrics(returns)
        expected = (1.10 * 0.95 * 1.02) - 1.0
        assert result["total"] == pytest.approx(expected, abs=1e-10)

    def test_empty_returns(self):
        assert BacktestMetrics._return_metrics([]) == {}

    def test_single_return(self):
        result = BacktestMetrics._return_metrics([0.05])
        assert result["mean"] == pytest.approx(0.05)
        assert result["total"] == pytest.approx(0.05)
        assert result["max"] == pytest.approx(0.05)
        assert result["min"] == pytest.approx(0.05)


# ── _volatility ─────────────────────────────────────────────


class TestVolatility:
    def test_basic(self):
        returns = [0.01, 0.02, -0.01, 0.03, -0.005]
        vol = BacktestMetrics._volatility(returns)
        assert vol is not None
        assert vol > 0

    def test_single_value(self):
        """Need at least 2 points for std."""
        assert BacktestMetrics._volatility([0.05]) is None

    def test_empty(self):
        assert BacktestMetrics._volatility([]) is None

    def test_constant_returns(self):
        """Constant returns have near-zero volatility."""
        vol = BacktestMetrics._volatility([0.01, 0.01, 0.01, 0.01])
        assert vol == pytest.approx(0.0, abs=1e-10)


# ── _sharpe_ratio ───────────────────────────────────────────


class TestSharpeRatio:
    def test_positive_sharpe(self):
        returns = [0.01, 0.02, 0.015, 0.005, 0.03]
        sharpe = BacktestMetrics._sharpe_ratio(returns, horizon=5)
        assert sharpe is not None
        assert sharpe > 0  # All positive returns → positive Sharpe

    def test_zero_std(self):
        """Constant returns mean zero std → None."""
        sharpe = BacktestMetrics._sharpe_ratio([0.01, 0.01, 0.01], horizon=5)
        assert sharpe is None

    def test_annualisation_factor(self):
        """Horizon affects the annualisation: 252/horizon."""
        returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        sharpe_h5 = BacktestMetrics._sharpe_ratio(returns, horizon=5)
        sharpe_h20 = BacktestMetrics._sharpe_ratio(returns, horizon=20)
        # Both should be non-None
        assert sharpe_h5 is not None
        assert sharpe_h20 is not None
        # Same underlying returns but different annualisation
        # sqrt(252/5) vs sqrt(252/20) → h5 has higher annualisation
        # so |sharpe_h5| should differ from |sharpe_h20|
        assert sharpe_h5 != sharpe_h20

    def test_insufficient_data(self):
        assert BacktestMetrics._sharpe_ratio([0.01], horizon=5) is None
        assert BacktestMetrics._sharpe_ratio([], horizon=5) is None


# ── _sortino_ratio ──────────────────────────────────────────


class TestSortinoRatio:
    def test_no_downside(self):
        """All positive returns → None (no downside deviation)."""
        sortino = BacktestMetrics._sortino_ratio([0.01, 0.02, 0.03], horizon=5)
        assert sortino is None

    def test_with_downside(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.005]
        sortino = BacktestMetrics._sortino_ratio(returns, horizon=5)
        assert sortino is not None

    def test_insufficient_data(self):
        assert BacktestMetrics._sortino_ratio([0.01], horizon=5) is None
        assert BacktestMetrics._sortino_ratio([], horizon=5) is None


# ── _max_drawdown ───────────────────────────────────────────


class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing wealth → drawdown is 0."""
        returns = [0.01, 0.02, 0.03]
        dd = BacktestMetrics._max_drawdown(returns)
        assert dd == pytest.approx(0.0)

    def test_simple_drawdown(self):
        """Known drawdown calculation."""
        returns = [0.10, -0.20, 0.05]
        dd = BacktestMetrics._max_drawdown(returns)
        assert dd is not None
        assert dd < 0  # Drawdown is negative
        # Wealth: 1.10, 0.88, 0.924
        # Peak:   1.10, 1.10, 1.10
        # DD:     0.0, (0.88-1.10)/1.10=-0.2, (0.924-1.10)/1.10=-0.16
        assert dd == pytest.approx(-0.2, abs=1e-10)

    def test_empty(self):
        assert BacktestMetrics._max_drawdown([]) is None


# ── _win_rate ───────────────────────────────────────────────


class TestWinRate:
    def test_basic(self):
        returns = [0.01, -0.01, 0.02, 0.03, -0.005]
        wr = BacktestMetrics._win_rate(returns)
        assert wr == pytest.approx(3 / 5)

    def test_all_positive(self):
        assert BacktestMetrics._win_rate([0.01, 0.02]) == 1.0

    def test_all_negative(self):
        assert BacktestMetrics._win_rate([-0.01, -0.02]) == 0.0

    def test_zero_return_not_win(self):
        """Zero return is not counted as a win."""
        assert BacktestMetrics._win_rate([0.0, 0.01]) == pytest.approx(0.5)

    def test_empty(self):
        assert BacktestMetrics._win_rate([]) is None


# ── _profit_factor ──────────────────────────────────────────


class TestProfitFactor:
    def test_basic(self):
        returns = [0.03, -0.01, 0.02, -0.005]
        pf = BacktestMetrics._profit_factor(returns)
        assert pf is not None
        # gains = 0.03 + 0.02 = 0.05, losses = -0.01 + -0.005 = -0.015
        assert pf == pytest.approx(0.05 / 0.015)

    def test_no_losses(self):
        """All gains → None (division by zero edge case)."""
        assert BacktestMetrics._profit_factor([0.01, 0.02]) is None

    def test_no_gains(self):
        assert BacktestMetrics._profit_factor([-0.01, -0.02]) is None

    def test_empty(self):
        assert BacktestMetrics._profit_factor([]) is None


# ── _calibration_buckets ────────────────────────────────────


class TestCalibrationBuckets:
    def test_basic_buckets(self, sample_daily_results):
        """5 data points with 5 buckets → 5 buckets."""
        buckets = BacktestMetrics._calibration_buckets(sample_daily_results, 5)
        assert len(buckets) == 5
        for b in buckets:
            assert isinstance(b, CalibrationBucket)
            assert b.count >= 1
            assert 0.0 <= b.hit_rate <= 1.0

    def test_fewer_than_buckets(self):
        """Fewer observations than buckets → empty list."""
        results = [
            DailyBacktestResult(
                date=date(2025, 1, 1),
                ranked_themes=[{"score": 1.0}],
                top_n_avg_return=0.01,
            ),
        ]
        buckets = BacktestMetrics._calibration_buckets(results, 5)
        assert buckets == []

    def test_skip_none_returns(self):
        """None returns are excluded from calibration."""
        results = [
            DailyBacktestResult(
                date=date(2025, 1, i),
                ranked_themes=[{"score": float(i)}],
                top_n_avg_return=None,
            )
            for i in range(1, 10)
        ]
        buckets = BacktestMetrics._calibration_buckets(results, 3)
        assert buckets == []

    def test_bucket_label_format(self, sample_daily_results):
        buckets = BacktestMetrics._calibration_buckets(sample_daily_results, 2)
        assert len(buckets) == 2
        assert buckets[0].bucket_label.startswith("Q1")
        assert buckets[1].bucket_label.startswith("Q2")


# ── compute (integration) ──────────────────────────────────


class TestCompute:
    def test_full_compute(self, sample_daily_results):
        """Smoke test for the full compute pipeline."""
        result = BacktestMetrics.compute(sample_daily_results, horizon=5, n_calibration_buckets=2)
        assert "hit_rate" in result
        assert "mean_return" in result
        assert "total_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert "max_drawdown" in result
        assert "win_rate" in result
        assert "profit_factor" in result
        assert "calibration" in result
        assert result["trading_days_with_returns"] == 5

    def test_empty_results(self):
        result = BacktestMetrics.compute([], horizon=5)
        assert result["hit_rate"] is None
        assert result["mean_return"] is None
        assert result["trading_days_with_returns"] == 0
