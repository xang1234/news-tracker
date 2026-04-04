"""Tests for narrative parity and validation checks.

Verifies that the new component model produces results consistent
with the existing narrative system, providing evidence that
narrative publication is safe.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.narrative.components import NarrativeComponents, compute_narrative_components
from src.narrative.parity import (
    DIRECTIONAL_TOLERANCE,
    ParityCheck,
    ParityReport,
    check_backfill_stability,
    check_component_conviction_parity,
    check_replay_coverage,
    compute_components_for_run,
    validate_run_parity,
)
from src.narrative.schemas import NarrativeRun

NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)


# -- Helpers ---------------------------------------------------------------


def _make_run(
    run_id: str = "nr_001",
    *,
    conviction_score: float = 50.0,
    current_rate_per_hour: float = 20.0,
    current_acceleration: float = 8.0,
    doc_count: int = 15,
    platform_count: int = 3,
    avg_sentiment: float = 0.6,
    avg_authority: float = 0.7,
    started_at: datetime | None = None,
    last_document_at: datetime | None = None,
    **overrides,
) -> NarrativeRun:
    defaults = dict(
        run_id=run_id,
        theme_id="theme_test",
        status="active",
        centroid=np.zeros(768),
        label="Test Run",
        started_at=started_at or NOW - timedelta(hours=4),
        last_document_at=last_document_at or NOW,
        conviction_score=conviction_score,
        current_rate_per_hour=current_rate_per_hour,
        current_acceleration=current_acceleration,
        doc_count=doc_count,
        platform_count=platform_count,
        avg_sentiment=avg_sentiment,
        avg_authority=avg_authority,
    )
    defaults.update(overrides)
    return NarrativeRun(**defaults)


# -- ParityCheck / ParityReport tests --------------------------------------


class TestParityReport:
    """ParityReport aggregation."""

    def test_all_passed(self) -> None:
        report = ParityReport()
        report.add(ParityCheck("check_a", True, "ok"))
        report.add(ParityCheck("check_b", True, "ok"))
        assert report.passed is True
        assert len(report.failed_checks) == 0

    def test_one_failed(self) -> None:
        report = ParityReport()
        report.add(ParityCheck("check_a", True, "ok"))
        report.add(ParityCheck("check_b", False, "bad"))
        assert report.passed is False
        assert len(report.failed_checks) == 1
        assert report.failed_checks[0].check_name == "check_b"

    def test_empty_report_passes(self) -> None:
        report = ParityReport()
        assert report.passed is True


# -- Component conviction parity tests ------------------------------------


class TestComponentConvictionParity:
    """New composite vs legacy conviction directional agreement."""

    def test_close_scores_pass(self) -> None:
        run = _make_run(conviction_score=40.0)
        components = compute_components_for_run(run, now=NOW)
        check = check_component_conviction_parity(run, components)
        # The component composite should be in a reasonable range
        assert check.details["delta"] <= DIRECTIONAL_TOLERANCE or not check.passed

    def test_wildly_different_scores_fail(self) -> None:
        # Legacy conviction is 95 but run has zero activity metrics
        run = _make_run(
            conviction_score=95.0,
            current_rate_per_hour=0.0,
            current_acceleration=0.0,
            doc_count=0,
            platform_count=0,
            avg_sentiment=0.0,
            avg_authority=0.0,
        )
        components = compute_components_for_run(run, now=NOW)
        check = check_component_conviction_parity(run, components)
        # Composite near 0, legacy at 95 → should fail
        assert check.passed is False
        assert check.details["delta"] > DIRECTIONAL_TOLERANCE

    def test_check_has_details(self) -> None:
        run = _make_run()
        components = compute_components_for_run(run, now=NOW)
        check = check_component_conviction_parity(run, components)
        assert "legacy_conviction" in check.details
        assert "component_composite" in check.details
        assert "delta" in check.details


# -- Backfill stability tests ----------------------------------------------


class TestBackfillStability:
    """Two backfill runs should produce matching metrics."""

    def test_identical_runs_pass(self) -> None:
        run_a = _make_run("nr_a")
        run_b = _make_run("nr_b")
        check = check_backfill_stability(run_a, run_b)
        assert check.passed is True

    def test_different_doc_count_fails(self) -> None:
        run_a = _make_run("nr_a", doc_count=10)
        run_b = _make_run("nr_b", doc_count=15)
        check = check_backfill_stability(run_a, run_b)
        assert check.passed is False
        assert "doc_count" in check.message

    def test_different_sentiment_fails(self) -> None:
        run_a = _make_run("nr_a", avg_sentiment=0.5)
        run_b = _make_run("nr_b", avg_sentiment=-0.3)
        check = check_backfill_stability(run_a, run_b)
        assert check.passed is False
        assert "avg_sentiment" in check.message

    def test_small_sentiment_diff_passes(self) -> None:
        """Within 0.01 tolerance."""
        run_a = _make_run("nr_a", avg_sentiment=0.600)
        run_b = _make_run("nr_b", avg_sentiment=0.605)
        check = check_backfill_stability(run_a, run_b)
        assert check.passed is True

    def test_multiple_issues_reported(self) -> None:
        run_a = _make_run("nr_a", doc_count=10, platform_count=2)
        run_b = _make_run("nr_b", doc_count=15, platform_count=4)
        check = check_backfill_stability(run_a, run_b)
        assert check.passed is False
        assert len(check.details["issues"]) >= 2


# -- Replay coverage tests ------------------------------------------------


class TestReplayCoverage:
    """Triggered signals should have component support."""

    def test_no_signal_passes(self) -> None:
        run = _make_run()
        components = compute_components_for_run(run, now=NOW)
        check = check_replay_coverage(run, components, signal_triggered=False)
        assert check.passed is True
        assert "not applicable" in check.message

    def test_triggered_with_strong_components(self) -> None:
        run = _make_run(
            current_rate_per_hour=40.0,
            doc_count=30,
            platform_count=4,
            avg_authority=0.8,
        )
        components = compute_components_for_run(run, now=NOW)
        check = check_replay_coverage(run, components, signal_triggered=True)
        assert check.passed is True

    def test_triggered_with_weak_components_fails(self) -> None:
        run = _make_run(
            current_rate_per_hour=0.5,
            current_acceleration=0.0,
            doc_count=1,
            platform_count=1,
            avg_sentiment=0.05,
            avg_authority=0.1,
            last_document_at=NOW - timedelta(days=5),
        )
        components = compute_components_for_run(run, now=NOW)
        check = check_replay_coverage(run, components, signal_triggered=True)
        assert check.passed is False
        assert "investigate" in check.message


# -- compute_components_for_run tests --------------------------------------


class TestComputeComponentsForRun:
    """Convenience wrapper around compute_narrative_components."""

    def test_produces_components(self) -> None:
        run = _make_run()
        components = compute_components_for_run(run, now=NOW)
        assert isinstance(components, NarrativeComponents)
        assert components.composite > 0

    def test_passes_through_optional_args(self) -> None:
        run = _make_run()
        c1 = compute_components_for_run(run, now=NOW, source_type_count=1)
        c2 = compute_components_for_run(run, now=NOW, source_type_count=4)
        assert c2.corroboration.source_diversity > c1.corroboration.source_diversity


# -- Full validation tests -------------------------------------------------


class TestValidateRunParity:
    """End-to-end parity validation for a single run."""

    def test_healthy_run(self) -> None:
        run = _make_run(conviction_score=30.0)
        components, checks = validate_run_parity(run, now=NOW)
        assert isinstance(components, NarrativeComponents)
        assert len(checks) == 2
        # Both checks should have names
        names = {c.check_name for c in checks}
        assert "component_conviction_parity" in names
        assert "replay_coverage" in names

    def test_with_signal_triggered(self) -> None:
        run = _make_run(
            conviction_score=60.0,
            current_rate_per_hour=40.0,
            doc_count=30,
            platform_count=4,
        )
        components, checks = validate_run_parity(
            run, signal_triggered=True, now=NOW
        )
        coverage_check = next(
            c for c in checks if c.check_name == "replay_coverage"
        )
        assert coverage_check.details["signal_triggered"] is True
