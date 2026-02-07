"""Backtest result visualization.

Stateless service following the BacktestMetrics pattern — all methods
are static, no instance state, trivially testable. Generates matplotlib
charts saved to disk as PNG files.

Plot types:
- Cumulative returns curve (wealth multiplier over time)
- Drawdown chart (peak-to-trough decline, shaded)
- Score vs return scatter (calibration visual with trend line)
- Monthly performance heatmap (diverging colormap)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.backtest.engine import BacktestResults

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """Stateless chart generation for backtest results.

    All methods are static — call them directly on the class. Each
    method lazy-imports matplotlib to avoid the import cost when the
    module is loaded but not used.
    """

    @staticmethod
    def generate_all(
        results: BacktestResults,
        output_dir: str | Path,
    ) -> list[Path]:
        """Generate all chart types and return paths to saved PNGs.

        Args:
            results: Complete backtest results with daily detail.
            output_dir: Directory to save chart files.

        Returns:
            List of Path objects pointing to generated PNG files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []

        for plot_fn in (
            BacktestVisualizer.plot_cumulative_returns,
            BacktestVisualizer.plot_drawdown,
            BacktestVisualizer.plot_score_vs_return,
            BacktestVisualizer.plot_monthly_heatmap,
        ):
            try:
                path = plot_fn(results, output_dir)
                paths.append(path)
            except Exception:
                logger.exception("Failed to generate %s", plot_fn.__name__)

        return paths

    @staticmethod
    def plot_cumulative_returns(
        results: BacktestResults,
        output_dir: str | Path,
    ) -> Path:
        """Plot cumulative wealth curve from daily returns.

        Computes ``np.cumprod(1 + returns)`` and draws a line chart
        with a horizontal breakeven line at 1.0.

        Args:
            results: Backtest results with daily_results populated.
            output_dir: Directory to save the chart.

        Returns:
            Path to the saved PNG file.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        output_dir = Path(output_dir)

        dates, returns = _extract_date_returns(results)

        fig, ax = plt.subplots(figsize=(12, 6))

        if dates and returns:
            wealth = np.cumprod(1.0 + np.array(returns))
            ax.plot(dates, wealth, linewidth=1.5, color="#2196F3")
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.fill_between(
                dates, 1.0, wealth,
                where=wealth >= 1.0, alpha=0.1, color="#4CAF50",
            )
            ax.fill_between(
                dates, 1.0, wealth,
                where=wealth < 1.0, alpha=0.1, color="#F44336",
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()

        ax.set_xlabel("Date")
        ax.set_ylabel("Wealth Multiplier")
        ax.set_title(
            f"Cumulative Returns — {results.strategy} strategy, "
            f"{results.horizon}d horizon"
        )
        ax.grid(True, alpha=0.3)

        path = output_dir / "cumulative_returns.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    @staticmethod
    def plot_drawdown(
        results: BacktestResults,
        output_dir: str | Path,
    ) -> Path:
        """Plot drawdown chart with shaded decline regions.

        Computes drawdown as ``(wealth - peak) / peak`` and fills the
        area below zero in red. Annotates the maximum drawdown point.

        Args:
            results: Backtest results with daily_results populated.
            output_dir: Directory to save the chart.

        Returns:
            Path to the saved PNG file.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        output_dir = Path(output_dir)

        dates, returns = _extract_date_returns(results)

        fig, ax = plt.subplots(figsize=(12, 5))

        if dates and returns:
            arr = np.array(returns)
            wealth = np.cumprod(1.0 + arr)
            peak = np.maximum.accumulate(wealth)
            drawdown = (wealth - peak) / peak

            ax.fill_between(dates, 0, drawdown, color="#F44336", alpha=0.4)
            ax.plot(dates, drawdown, color="#D32F2F", linewidth=1.0)

            # Annotate max drawdown
            min_idx = int(np.argmin(drawdown))
            ax.annotate(
                f"{drawdown[min_idx]:.1%}",
                xy=(dates[min_idx], drawdown[min_idx]),
                xytext=(10, -20),
                textcoords="offset points",
                fontsize=9,
                color="#D32F2F",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#D32F2F"),
            )

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.set_title(
            f"Drawdown — {results.strategy} strategy, "
            f"{results.horizon}d horizon"
        )
        ax.grid(True, alpha=0.3)

        path = output_dir / "drawdown.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    @staticmethod
    def plot_score_vs_return(
        results: BacktestResults,
        output_dir: str | Path,
    ) -> Path:
        """Scatter plot of mean daily theme score vs realised return.

        Each point is one trading day. Color-coded by whether the
        direction was predicted correctly (green) or not (red). Includes
        a linear trend line via numpy polyfit.

        Args:
            results: Backtest results with daily_results populated.
            output_dir: Directory to save the chart.

        Returns:
            Path to the saved PNG file.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)

        scores: list[float] = []
        returns: list[float] = []
        colors: list[str] = []

        for dr in results.daily_results:
            if dr.top_n_avg_return is None or not dr.ranked_themes:
                continue
            day_scores = [t.get("score", 0.0) for t in dr.ranked_themes]
            if not day_scores:
                continue
            avg_score = sum(day_scores) / len(day_scores)
            scores.append(avg_score)
            returns.append(dr.top_n_avg_return)
            if dr.direction_correct is True:
                colors.append("#4CAF50")
            elif dr.direction_correct is False:
                colors.append("#F44336")
            else:
                colors.append("#9E9E9E")

        fig, ax = plt.subplots(figsize=(10, 7))

        if scores and returns:
            ax.scatter(scores, returns, c=colors, alpha=0.6, edgecolors="white", s=50)

            # Trend line
            if len(scores) >= 2:
                z = np.polyfit(scores, returns, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(scores), max(scores), 100)
                ax.plot(x_range, p(x_range), "--", color="#FF9800", linewidth=1.5,
                        label=f"Trend (slope={z[0]:.4f})")
                ax.legend()

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Mean Theme Score")
        ax.set_ylabel("Forward Return")
        ax.set_title(
            f"Score vs Return — {results.strategy} strategy, "
            f"{results.horizon}d horizon"
        )
        ax.grid(True, alpha=0.3)

        path = output_dir / "score_vs_return.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    @staticmethod
    def plot_monthly_heatmap(
        results: BacktestResults,
        output_dir: str | Path,
    ) -> Path:
        """Monthly performance heatmap with diverging colormap.

        Groups daily returns by (year, month), computes cumulative
        monthly return per cell, and renders as a heatmap with values
        annotated inside each cell.

        Args:
            results: Backtest results with daily_results populated.
            output_dir: Directory to save the chart.

        Returns:
            Path to the saved PNG file.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        output_dir = Path(output_dir)

        # Group returns by (year, month)
        monthly: dict[tuple[int, int], list[float]] = {}
        for dr in results.daily_results:
            if dr.top_n_avg_return is None:
                continue
            key = (dr.date.year, dr.date.month)
            monthly.setdefault(key, []).append(dr.top_n_avg_return)

        fig, ax = plt.subplots(figsize=(12, 6))

        if monthly:
            # Compute cumulative return per month: product(1+r) - 1
            monthly_returns: dict[tuple[int, int], float] = {}
            for key, rets in monthly.items():
                monthly_returns[key] = float(np.prod(1.0 + np.array(rets)) - 1.0)

            years = sorted(set(k[0] for k in monthly_returns))
            months = list(range(1, 13))
            month_labels = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ]

            # Build data matrix (years × months)
            data = np.full((len(years), 12), np.nan)
            for (yr, mo), ret in monthly_returns.items():
                row = years.index(yr)
                col = mo - 1
                data[row, col] = ret

            # Determine colormap bounds
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                abs_max = max(abs(valid.min()), abs(valid.max()), 1e-6)
                norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
            else:
                norm = TwoSlopeNorm(vmin=-0.01, vcenter=0.0, vmax=0.01)

            im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

            # Annotate cells
            for i in range(len(years)):
                for j in range(12):
                    val = data[i, j]
                    if not np.isnan(val):
                        text_color = "white" if abs(val) > abs_max * 0.6 else "black"
                        ax.text(
                            j, i, f"{val:.1%}",
                            ha="center", va="center",
                            fontsize=8, color=text_color,
                        )

            ax.set_xticks(range(12))
            ax.set_xticklabels(month_labels)
            ax.set_yticks(range(len(years)))
            ax.set_yticklabels([str(y) for y in years])
            fig.colorbar(im, ax=ax, label="Monthly Return", shrink=0.8)
        else:
            ax.text(
                0.5, 0.5, "No monthly data available",
                ha="center", va="center", transform=ax.transAxes,
            )

        ax.set_title(
            f"Monthly Returns — {results.strategy} strategy, "
            f"{results.horizon}d horizon"
        )

        path = output_dir / "monthly_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path


def _extract_date_returns(
    results: BacktestResults,
) -> tuple[list, list[float]]:
    """Extract (date, return) pairs where return is not None.

    Returns matplotlib-compatible date objects and float returns.
    """
    import matplotlib.dates as mdates

    dates = []
    returns: list[float] = []
    for dr in results.daily_results:
        if dr.top_n_avg_return is not None:
            dates.append(dr.date)
            returns.append(dr.top_n_avg_return)
    return dates, returns
