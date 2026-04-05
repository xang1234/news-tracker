"""Tests for per-lane operational metrics.

Verifies failure rate classification, freshness budget checks,
trace context building, and multi-lane report aggregation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.monitoring.lane_ops import (
    DEFAULT_BUDGET_CRITICAL_FRACTION,
    DEFAULT_BUDGET_WARNING_FRACTION,
    DEFAULT_FAILURE_CRITICAL,
    DEFAULT_FAILURE_WARNING,
    DEFAULT_FRESHNESS_BUDGETS,
    LaneRunSummary,
    build_lane_trace_attributes,
    check_all_lanes,
    check_lane_failure_rate,
    check_lane_freshness_budget,
)
from src.monitoring.quality_metrics import (
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
)

NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)


# -- Helpers ---------------------------------------------------------------


def _summary(
    lane: str = "narrative",
    total: int = 20,
    completed: int = 18,
    failed: int = 2,
    cancelled: int = 0,
    window_hours: float = 24.0,
    last_completed_at: datetime | None = None,
    last_failed_at: datetime | None = None,
) -> LaneRunSummary:
    return LaneRunSummary(
        lane=lane,
        window_hours=window_hours,
        total_runs=total,
        completed=completed,
        failed=failed,
        cancelled=cancelled,
        last_completed_at=last_completed_at or NOW - timedelta(hours=1),
        last_failed_at=last_failed_at,
    )


# -- Trace context tests ---------------------------------------------------


class TestTraceContext:
    """Lane-attributed span attributes."""

    def test_basic_attributes(self) -> None:
        attrs = build_lane_trace_attributes("narrative")
        assert attrs["intel.lane"] == "narrative"

    def test_with_run_id(self) -> None:
        attrs = build_lane_trace_attributes("filing", run_id="run_001")
        assert attrs["intel.lane"] == "filing"
        assert attrs["intel.run_id"] == "run_001"

    def test_without_run_id(self) -> None:
        attrs = build_lane_trace_attributes("structural")
        assert "intel.run_id" not in attrs

    def test_extra_attributes(self) -> None:
        attrs = build_lane_trace_attributes(
            "narrative",
            extra={"intel.phase": "scoring"},
        )
        assert attrs["intel.phase"] == "scoring"
        assert attrs["intel.lane"] == "narrative"

    def test_empty_extra(self) -> None:
        attrs = build_lane_trace_attributes("narrative", extra=None)
        assert len(attrs) == 1


# -- LaneRunSummary tests --------------------------------------------------


class TestLaneRunSummary:
    """Aggregated run statistics."""

    def test_failure_rate(self) -> None:
        s = _summary(total=20, failed=4)
        assert s.failure_rate == 0.2

    def test_zero_runs(self) -> None:
        s = _summary(total=0, completed=0, failed=0)
        assert s.failure_rate == 0.0

    def test_no_failures(self) -> None:
        s = _summary(total=10, completed=10, failed=0)
        assert s.failure_rate == 0.0

    def test_all_failed(self) -> None:
        s = _summary(total=5, completed=0, failed=5)
        assert s.failure_rate == 1.0

    def test_frozen(self) -> None:
        s = _summary()
        try:
            s.failed = 99  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


# -- Failure rate tests ----------------------------------------------------


class TestFailureRate:
    """Per-lane failure rate classification."""

    def test_ok(self) -> None:
        m = check_lane_failure_rate(
            _summary(total=20, failed=1), now=NOW,
        )
        assert m.severity == SEVERITY_OK
        assert m.metric_type == "lane_failure_rate"
        assert m.lane == "narrative"

    def test_warning(self) -> None:
        m = check_lane_failure_rate(
            _summary(total=20, failed=3), now=NOW,
        )
        assert m.severity == SEVERITY_WARNING

    def test_critical(self) -> None:
        m = check_lane_failure_rate(
            _summary(total=20, failed=6), now=NOW,
        )
        assert m.severity == SEVERITY_CRITICAL

    def test_zero_runs_ok(self) -> None:
        m = check_lane_failure_rate(
            _summary(total=0, completed=0, failed=0), now=NOW,
        )
        assert m.severity == SEVERITY_OK
        assert m.value == 0.0

    def test_boundary_warning(self) -> None:
        m = check_lane_failure_rate(
            _summary(total=100, failed=10), now=NOW,
        )
        assert m.severity == SEVERITY_OK  # 10% == threshold, <=

    def test_boundary_critical(self) -> None:
        m = check_lane_failure_rate(
            _summary(total=100, failed=25), now=NOW,
        )
        assert m.severity == SEVERITY_WARNING  # 25% == threshold, <=

    def test_details(self) -> None:
        m = check_lane_failure_rate(
            _summary(total=20, failed=3, cancelled=1), now=NOW,
        )
        assert m.details["total_runs"] == 20
        assert m.details["failed"] == 3
        assert m.details["cancelled"] == 1

    def test_custom_thresholds(self) -> None:
        m = check_lane_failure_rate(
            _summary(total=20, failed=2),
            warning_threshold=0.01,
            critical_threshold=0.05,
            now=NOW,
        )
        assert m.severity == SEVERITY_CRITICAL


# -- Freshness budget tests ------------------------------------------------


class TestFreshnessBudget:
    """Per-lane freshness budget classification."""

    def test_narrative_fresh(self) -> None:
        m = check_lane_freshness_budget(
            "narrative",
            NOW - timedelta(hours=2),
            now=NOW,
        )
        assert m.severity == SEVERITY_OK
        assert m.metric_type == "lane_freshness_budget"
        assert m.lane == "narrative"

    def test_narrative_warning(self) -> None:
        """Narrative budget=6h, warning at 75%=4.5h."""
        m = check_lane_freshness_budget(
            "narrative",
            NOW - timedelta(hours=5),
            now=NOW,
        )
        assert m.severity == SEVERITY_WARNING

    def test_narrative_critical(self) -> None:
        """Narrative budget=6h, critical at 100%=6h."""
        m = check_lane_freshness_budget(
            "narrative",
            NOW - timedelta(hours=7),
            now=NOW,
        )
        assert m.severity == SEVERITY_CRITICAL

    def test_filing_longer_budget(self) -> None:
        """Filing budget=72h. 48h old should be ok."""
        m = check_lane_freshness_budget(
            "filing",
            NOW - timedelta(hours=48),
            now=NOW,
        )
        assert m.severity == SEVERITY_OK

    def test_no_completed_runs(self) -> None:
        m = check_lane_freshness_budget("narrative", None, now=NOW)
        assert m.severity == SEVERITY_CRITICAL
        assert "no completed runs" in m.message

    def test_custom_budget(self) -> None:
        m = check_lane_freshness_budget(
            "narrative",
            NOW - timedelta(hours=3),
            budget_hours=2.0,
            now=NOW,
        )
        assert m.severity == SEVERITY_CRITICAL

    def test_utilization_in_value(self) -> None:
        """Value is utilization: hours_since / budget_hours."""
        m = check_lane_freshness_budget(
            "narrative",
            NOW - timedelta(hours=3),
            budget_hours=6.0,
            now=NOW,
        )
        assert abs(m.value - 0.5) < 0.01

    def test_details(self) -> None:
        m = check_lane_freshness_budget(
            "filing",
            NOW - timedelta(hours=24),
            now=NOW,
        )
        assert m.details["budget_hours"] == 72.0
        assert abs(m.details["hours_since_completion"] - 24.0) < 0.01
        assert m.details["last_completed_at"] is not None

    def test_unknown_lane_default_budget(self) -> None:
        """Unknown lanes get 24h default budget."""
        m = check_lane_freshness_budget(
            "unknown_lane",
            NOW - timedelta(hours=20),
            now=NOW,
        )
        assert m.severity == SEVERITY_WARNING  # 20h > 75% of 24h


# -- Multi-lane report tests -----------------------------------------------


class TestCheckAllLanes:
    """Cross-lane report aggregation."""

    def test_basic_report(self) -> None:
        summaries = {
            "narrative": _summary(lane="narrative"),
            "filing": _summary(lane="filing"),
        }
        completions = {
            "narrative": NOW - timedelta(hours=1),
            "filing": NOW - timedelta(hours=12),
        }
        report = check_all_lanes(summaries, completions, now=NOW)
        types = {m.metric_type for m in report.metrics}
        assert "lane_failure_rate" in types
        assert "lane_freshness_budget" in types

    def test_freshness_for_all_lanes(self) -> None:
        """Freshness budget checked for ALL canonical lanes."""
        report = check_all_lanes({}, {}, now=NOW)
        budget_metrics = [
            m for m in report.metrics
            if m.metric_type == "lane_freshness_budget"
        ]
        lanes = {m.lane for m in budget_metrics}
        from src.contracts.intelligence.lanes import ALL_LANES
        assert lanes == set(ALL_LANES)

    def test_failure_only_for_provided_summaries(self) -> None:
        """Failure rate only checked for lanes with summaries."""
        summaries = {"narrative": _summary(lane="narrative")}
        report = check_all_lanes(summaries, {}, now=NOW)
        failure_metrics = [
            m for m in report.metrics
            if m.metric_type == "lane_failure_rate"
        ]
        assert len(failure_metrics) == 1
        assert failure_metrics[0].lane == "narrative"

    def test_overall_severity_worst_of(self) -> None:
        summaries = {
            "narrative": _summary(lane="narrative", total=10, failed=5),
        }
        completions = {"narrative": NOW - timedelta(hours=1)}
        report = check_all_lanes(summaries, completions, now=NOW)
        assert report.overall_severity == SEVERITY_CRITICAL

    def test_empty_inputs(self) -> None:
        report = check_all_lanes({}, {}, now=NOW)
        assert len(report.metrics) >= 4  # freshness budgets for all lanes
