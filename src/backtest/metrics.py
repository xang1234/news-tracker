"""Backtest performance metrics computation.

Stateless service following the LifecycleClassifier pattern — all methods
are static, no instance state, trivially testable. Computes standard quant
metrics (Sharpe, Sortino, max drawdown, calibration) from daily backtest
results.

The ``CalibrationBucket`` dataclass lives here to avoid circular imports
(engine.py imports from metrics.py; metrics.py uses TYPE_CHECKING for
DailyBacktestResult).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.backtest.engine import DailyBacktestResult


@dataclass
class CalibrationBucket:
    """One quantile bucket comparing ranking scores to realised returns.

    Used for calibration analysis: do high-scored themes actually produce
    higher forward returns?

    Attributes:
        bucket_label: Human-readable label (e.g., "Q1 (0.00-0.20)").
        score_min: Lower bound of ranking scores in this bucket.
        score_max: Upper bound of ranking scores in this bucket.
        count: Number of observations in the bucket.
        avg_score: Mean ranking score.
        avg_return: Mean forward return.
        hit_rate: Fraction of observations with positive return.
    """

    bucket_label: str
    score_min: float
    score_max: float
    count: int
    avg_score: float
    avg_return: float
    hit_rate: float


class BacktestMetrics:
    """Stateless computation of backtest performance metrics.

    All methods are static — call them directly on the class without
    instantiation. This matches the LifecycleClassifier and
    VolumeMetricsService patterns used elsewhere in the project.
    """

    @staticmethod
    def compute(
        daily_results: list[DailyBacktestResult],
        horizon: int,
        n_calibration_buckets: int = 5,
    ) -> dict:
        """Orchestrate all metric computations.

        Args:
            daily_results: Per-day backtest outputs.
            horizon: Forward horizon used for annualisation.
            n_calibration_buckets: Quantile buckets for calibration.

        Returns:
            Dict with all aggregate metrics, suitable for JSON serialisation
            and storage in ``backtest_runs.results``.
        """
        returns = BacktestMetrics._extract_returns(daily_results)
        ret_metrics = BacktestMetrics._return_metrics(returns)

        return {
            "hit_rate": BacktestMetrics._directional_accuracy(daily_results),
            "mean_return": ret_metrics.get("mean"),
            "total_return": ret_metrics.get("total"),
            "max_return": ret_metrics.get("max"),
            "min_return": ret_metrics.get("min"),
            "volatility": BacktestMetrics._volatility(returns),
            "sharpe_ratio": BacktestMetrics._sharpe_ratio(returns, horizon),
            "sortino_ratio": BacktestMetrics._sortino_ratio(returns, horizon),
            "max_drawdown": BacktestMetrics._max_drawdown(returns),
            "win_rate": BacktestMetrics._win_rate(returns),
            "profit_factor": BacktestMetrics._profit_factor(returns),
            "calibration": [
                {
                    "bucket_label": b.bucket_label,
                    "score_min": b.score_min,
                    "score_max": b.score_max,
                    "count": b.count,
                    "avg_score": b.avg_score,
                    "avg_return": b.avg_return,
                    "hit_rate": b.hit_rate,
                }
                for b in BacktestMetrics._calibration_buckets(
                    daily_results, n_calibration_buckets
                )
            ],
            "trading_days_with_returns": len(returns),
        }

    @staticmethod
    def _extract_returns(
        daily_results: list[DailyBacktestResult],
    ) -> list[float]:
        """Extract non-None top_n_avg_return values from daily results."""
        return [
            r.top_n_avg_return
            for r in daily_results
            if r.top_n_avg_return is not None
        ]

    @staticmethod
    def _directional_accuracy(
        daily_results: list[DailyBacktestResult],
    ) -> float | None:
        """Fraction of days where direction was correctly predicted.

        Returns None if no days have direction data.
        """
        directions = [
            r.direction_correct
            for r in daily_results
            if r.direction_correct is not None
        ]
        if not directions:
            return None
        return sum(1 for d in directions if d) / len(directions)

    @staticmethod
    def _return_metrics(returns: list[float]) -> dict:
        """Compute basic return statistics.

        Returns:
            Dict with mean, total (cumulative product), max, min.
            Empty dict if no returns.
        """
        if not returns:
            return {}
        arr = np.array(returns)
        # Cumulative return: product of (1 + r_i) - 1
        total = float(np.prod(1.0 + arr) - 1.0)
        return {
            "mean": float(np.mean(arr)),
            "total": total,
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
        }

    @staticmethod
    def _volatility(returns: list[float]) -> float | None:
        """Standard deviation of returns. None if < 2 data points."""
        if len(returns) < 2:
            return None
        return float(np.std(returns, ddof=1))

    @staticmethod
    def _sharpe_ratio(
        returns: list[float],
        horizon: int,
    ) -> float | None:
        """Annualised Sharpe ratio (assuming zero risk-free rate).

        Formula: (mean * annualisation) / (std * sqrt(annualisation))
        where annualisation = 252 / horizon.

        Returns None if < 2 data points or zero standard deviation.
        """
        if len(returns) < 2:
            return None
        arr = np.array(returns)
        std = float(np.std(arr, ddof=1))
        if std == 0.0:
            return None
        mean = float(np.mean(arr))
        ann_factor = 252.0 / horizon
        return (mean * ann_factor) / (std * math.sqrt(ann_factor))

    @staticmethod
    def _sortino_ratio(
        returns: list[float],
        horizon: int,
    ) -> float | None:
        """Annualised Sortino ratio (downside deviation only).

        Returns None if < 2 data points or zero downside deviation.
        """
        if len(returns) < 2:
            return None
        arr = np.array(returns)
        downside = arr[arr < 0]
        if len(downside) == 0:
            return None
        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0.0:
            return None
        mean = float(np.mean(arr))
        ann_factor = 252.0 / horizon
        return (mean * ann_factor) / (downside_std * math.sqrt(ann_factor))

    @staticmethod
    def _max_drawdown(returns: list[float]) -> float | None:
        """Maximum peak-to-trough drawdown from the cumulative wealth curve.

        Returns None if no returns. Returns 0.0 if wealth never declines.
        """
        if not returns:
            return None
        wealth = np.cumprod(1.0 + np.array(returns))
        peak = np.maximum.accumulate(wealth)
        drawdowns = (wealth - peak) / peak
        return float(np.min(drawdowns))

    @staticmethod
    def _win_rate(returns: list[float]) -> float | None:
        """Fraction of returns that are positive. None if empty."""
        if not returns:
            return None
        wins = sum(1 for r in returns if r > 0)
        return wins / len(returns)

    @staticmethod
    def _profit_factor(returns: list[float]) -> float | None:
        """Sum of positive returns divided by abs(sum of negative returns).

        Returns None if no positive or no negative returns.
        """
        if not returns:
            return None
        gains = sum(r for r in returns if r > 0)
        losses = sum(r for r in returns if r < 0)
        if gains == 0 or losses == 0:
            return None
        return gains / abs(losses)

    @staticmethod
    def _calibration_buckets(
        daily_results: list[DailyBacktestResult],
        n_buckets: int,
    ) -> list[CalibrationBucket]:
        """Group daily results into score quantile buckets.

        For each day, uses the mean score of top-N ranked themes and the
        realised return. Groups into ``n_buckets`` quantile bins, then
        computes average score, average return, and hit rate per bucket.

        Returns empty list if insufficient data (< n_buckets observations).
        """
        # Build (score, return) pairs
        pairs: list[tuple[float, float]] = []
        for r in daily_results:
            if r.top_n_avg_return is None or not r.ranked_themes:
                continue
            # Mean score of ranked themes on that day
            scores = [t.get("score", 0.0) for t in r.ranked_themes]
            if scores:
                avg_score = sum(scores) / len(scores)
                pairs.append((avg_score, r.top_n_avg_return))

        if len(pairs) < n_buckets:
            return []

        # Sort by score ascending
        pairs.sort(key=lambda p: p[0])

        # Split into quantile buckets
        bucket_size = len(pairs) / n_buckets
        buckets: list[CalibrationBucket] = []

        for i in range(n_buckets):
            start_idx = int(round(i * bucket_size))
            end_idx = int(round((i + 1) * bucket_size))
            bucket_pairs = pairs[start_idx:end_idx]

            if not bucket_pairs:
                continue

            scores_in = [p[0] for p in bucket_pairs]
            returns_in = [p[1] for p in bucket_pairs]
            hits = sum(1 for r in returns_in if r > 0)

            buckets.append(
                CalibrationBucket(
                    bucket_label=f"Q{i + 1} ({min(scores_in):.2f}-{max(scores_in):.2f})",
                    score_min=min(scores_in),
                    score_max=max(scores_in),
                    count=len(bucket_pairs),
                    avg_score=sum(scores_in) / len(scores_in),
                    avg_return=sum(returns_in) / len(returns_in),
                    hit_rate=hits / len(bucket_pairs),
                )
            )

        return buckets
