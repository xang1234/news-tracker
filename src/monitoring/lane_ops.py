"""Per-lane operational metrics: failure rates, freshness budgets, and trace context.

Provides lane-level operational visibility over the intelligence
pipeline. Separate from publish-gating lane health — these metrics
answer "where is the pipeline failing or aging?" for operators,
not "can we publish?"

Three metric types:
    - Failure rate: fraction of recent lane runs that failed
    - Freshness budget: is the lane within its operational SLA?
    - Trace context: lane-attributed span attributes for OTLP filtering

All check functions are stateless — the caller provides pre-
aggregated run counts and timestamps, the checker classifies them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.contracts.intelligence.lanes import (
    ALL_LANES,
    LANE_BACKTEST,
    LANE_FILING,
    LANE_NARRATIVE,
    LANE_STRUCTURAL,
)
from src.monitoring.quality_metrics import (
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
    QualityMetric,
    QualityReport,
    _classify,
)


# -- Default thresholds -------------------------------------------------------

DEFAULT_FAILURE_WARNING = 0.10
DEFAULT_FAILURE_CRITICAL = 0.25

# Per-lane operational SLAs, separate from publish-gating thresholds
# in lane_health.py. A lane can be within budget but aging
# (operationally fine), or exceeding budget but not yet stale
# enough to block publication.
DEFAULT_FRESHNESS_BUDGETS: dict[str, float] = {
    LANE_NARRATIVE: 6.0,
    LANE_FILING: 72.0,
    LANE_STRUCTURAL: 24.0,
    LANE_BACKTEST: 168.0,
}

DEFAULT_BUDGET_WARNING_FRACTION = 0.75
DEFAULT_BUDGET_CRITICAL_FRACTION = 1.0


# -- Trace context helpers -----------------------------------------------------


def build_lane_trace_attributes(
    lane: str,
    run_id: str | None = None,
    *,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build span attributes for lane-attributed tracing.

    Returns a dict suitable for OpenTelemetry span.set_attributes().
    Enables Jaeger/OTLP filtering by lane name.

    Args:
        lane: Canonical lane name.
        run_id: Optional lane run ID for correlation.
        extra: Additional attributes to include.

    Returns:
        Dict of span attribute key → value.
    """
    attrs = {"intel.lane": lane}
    if run_id is not None:
        attrs["intel.run_id"] = run_id
    if extra:
        attrs.update(extra)
    return attrs


# -- Lane run summary ----------------------------------------------------------


@dataclass(frozen=True)
class LaneRunSummary:
    """Aggregated lane run statistics for a recent window.

    Attributes:
        lane: Which lane.
        window_hours: How many hours back the window covers.
        total_runs: Runs in the window.
        completed: Runs that completed successfully.
        failed: Runs that failed.
        cancelled: Runs that were cancelled.
        last_completed_at: Most recent successful completion.
        last_failed_at: Most recent failure (None if no failures).
    """

    lane: str
    window_hours: float
    total_runs: int
    completed: int
    failed: int
    cancelled: int
    last_completed_at: datetime | None = None
    last_failed_at: datetime | None = None

    @property
    def failure_rate(self) -> float:
        """Fraction of runs that failed (0.0 if no runs)."""
        if self.total_runs == 0:
            return 0.0
        return self.failed / self.total_runs


# -- Check functions (stateless) -----------------------------------------------


def check_lane_failure_rate(
    summary: LaneRunSummary,
    *,
    warning_threshold: float = DEFAULT_FAILURE_WARNING,
    critical_threshold: float = DEFAULT_FAILURE_CRITICAL,
    now: datetime | None = None,
) -> QualityMetric:
    """Check whether a lane's failure rate exceeds thresholds.

    Args:
        summary: Aggregated run stats for the lane.
        warning_threshold: Failure rate for warning.
        critical_threshold: Failure rate for critical.
        now: Measurement timestamp.

    Returns:
        QualityMetric with lane_failure_rate type.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    rate = summary.failure_rate
    severity = _classify(
        rate, warning_threshold, critical_threshold, higher_is_better=False,
    )

    return QualityMetric(
        metric_type="lane_failure_rate",
        lane=summary.lane,
        value=rate,
        severity=severity,
        thresholds={"warning": warning_threshold, "critical": critical_threshold},
        message=(
            f"{summary.lane}: {rate:.1%} failure rate "
            f"({summary.failed}/{summary.total_runs} in "
            f"{summary.window_hours:.0f}h window)"
        ),
        details={
            "total_runs": summary.total_runs,
            "completed": summary.completed,
            "failed": summary.failed,
            "cancelled": summary.cancelled,
            "window_hours": summary.window_hours,
            "last_failed_at": (
                summary.last_failed_at.isoformat()
                if summary.last_failed_at
                else None
            ),
        },
        measured_at=now,
    )


def check_lane_freshness_budget(
    lane: str,
    last_completed_at: datetime | None,
    *,
    budget_hours: float | None = None,
    warning_fraction: float = DEFAULT_BUDGET_WARNING_FRACTION,
    critical_fraction: float = DEFAULT_BUDGET_CRITICAL_FRACTION,
    now: datetime | None = None,
) -> QualityMetric:
    """Check whether a lane is within its freshness budget.

    The freshness budget is a per-lane operational SLA: the filing
    lane has a longer budget (72h) because filings are quarterly,
    while the narrative lane has a shorter budget (6h) because
    news is real-time.

    Args:
        lane: Canonical lane name.
        last_completed_at: When the lane last completed successfully.
        budget_hours: Operational SLA in hours (default: per-lane).
        warning_fraction: Fraction of budget for warning (default 75%).
        critical_fraction: Fraction of budget for critical (default 100%).
        now: Current time.

    Returns:
        QualityMetric with lane_freshness_budget type.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if budget_hours is None:
        budget_hours = DEFAULT_FRESHNESS_BUDGETS.get(lane, 24.0)

    if last_completed_at is None:
        hours_since = float("inf")
        utilization = float("inf")
    else:
        hours_since = max(0.0, (now - last_completed_at).total_seconds() / 3600)
        utilization = hours_since / budget_hours if budget_hours > 0 else float("inf")

    warning_at = budget_hours * warning_fraction
    critical_at = budget_hours * critical_fraction

    severity = _classify(
        hours_since, warning_at, critical_at, higher_is_better=False,
    )

    return QualityMetric(
        metric_type="lane_freshness_budget",
        lane=lane,
        value=round(utilization, 4) if utilization != float("inf") else 999.0,
        severity=severity,
        thresholds={
            "warning": round(warning_at, 2),
            "critical": round(critical_at, 2),
        },
        message=(
            f"{lane}: {hours_since:.1f}h since last completion "
            f"(budget={budget_hours:.0f}h, "
            f"utilization={utilization:.0%})"
            if last_completed_at is not None
            else f"{lane}: no completed runs (budget={budget_hours:.0f}h)"
        ),
        details={
            "hours_since_completion": round(hours_since, 2) if hours_since != float("inf") else None,
            "budget_hours": budget_hours,
            "utilization": round(utilization, 4) if utilization != float("inf") else None,
            "last_completed_at": (
                last_completed_at.isoformat()
                if last_completed_at
                else None
            ),
        },
        measured_at=now,
    )


# -- Multi-lane report builder -------------------------------------------------


def check_all_lanes(
    summaries: dict[str, LaneRunSummary],
    last_completions: dict[str, datetime | None],
    *,
    failure_warning: float = DEFAULT_FAILURE_WARNING,
    failure_critical: float = DEFAULT_FAILURE_CRITICAL,
    budget_warning_fraction: float = DEFAULT_BUDGET_WARNING_FRACTION,
    budget_critical_fraction: float = DEFAULT_BUDGET_CRITICAL_FRACTION,
    now: datetime | None = None,
) -> QualityReport:
    """Run failure rate and freshness budget checks across all lanes.

    Args:
        summaries: Per-lane run summaries (lane → summary).
        last_completions: Per-lane last completion times.
        failure_warning: Failure rate warning threshold.
        failure_critical: Failure rate critical threshold.
        budget_warning_fraction: Fraction of budget for warning.
        budget_critical_fraction: Fraction of budget for critical.
        now: Current time.

    Returns:
        QualityReport with per-lane failure and freshness metrics.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    metrics: list[QualityMetric] = []

    for lane in sorted(summaries.keys()):
        summary = summaries[lane]
        metrics.append(
            check_lane_failure_rate(
                summary,
                warning_threshold=failure_warning,
                critical_threshold=failure_critical,
                now=now,
            )
        )

    for lane in ALL_LANES:
        last = last_completions.get(lane)
        metrics.append(
            check_lane_freshness_budget(
                lane, last,
                warning_fraction=budget_warning_fraction,
                critical_fraction=budget_critical_fraction,
                now=now,
            )
        )

    return QualityReport(metrics=metrics, measured_at=now)
