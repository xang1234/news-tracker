"""Tests for intelligence layer quality metrics.

Verifies lineage completeness, unresolved entity rates, filing
parse quality, and stale evidence checks with correct severity
classification and report aggregation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.monitoring.quality_metrics import (
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
    VALID_METRIC_TYPES,
    QualityMetric,
    _classify,
    build_quality_report,
    check_filing_parse_quality,
    check_lineage_completeness,
    check_stale_evidence,
    check_unresolved_entities,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Classification tests -----------------------------------------------------


class TestClassify:
    """Severity classification helper."""

    def test_higher_is_better_ok(self) -> None:
        assert _classify(0.97, 0.95, 0.85) == SEVERITY_OK

    def test_higher_is_better_warning(self) -> None:
        assert _classify(0.90, 0.95, 0.85) == SEVERITY_WARNING

    def test_higher_is_better_critical(self) -> None:
        assert _classify(0.80, 0.95, 0.85) == SEVERITY_CRITICAL

    def test_higher_is_better_boundary_ok(self) -> None:
        assert _classify(0.95, 0.95, 0.85) == SEVERITY_OK

    def test_higher_is_better_boundary_warning(self) -> None:
        assert _classify(0.85, 0.95, 0.85) == SEVERITY_WARNING

    def test_lower_is_better_ok(self) -> None:
        assert _classify(0.03, 0.05, 0.15, higher_is_better=False) == SEVERITY_OK

    def test_lower_is_better_warning(self) -> None:
        assert _classify(0.10, 0.05, 0.15, higher_is_better=False) == SEVERITY_WARNING

    def test_lower_is_better_critical(self) -> None:
        assert _classify(0.20, 0.05, 0.15, higher_is_better=False) == SEVERITY_CRITICAL

    def test_lower_is_better_boundary_ok(self) -> None:
        assert _classify(0.05, 0.05, 0.15, higher_is_better=False) == SEVERITY_OK

    def test_lower_is_better_boundary_warning(self) -> None:
        assert _classify(0.15, 0.05, 0.15, higher_is_better=False) == SEVERITY_WARNING


# -- Lineage completeness tests -----------------------------------------------


class TestLineageCompleteness:
    """Fraction of objects with source lineage."""

    def test_perfect_lineage(self) -> None:
        m = check_lineage_completeness(100, 100, "narrative", now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 1.0
        assert m.metric_type == "lineage_completeness"
        assert m.lane == "narrative"

    def test_good_lineage(self) -> None:
        m = check_lineage_completeness(100, 96, "filing", now=NOW)
        assert m.severity == SEVERITY_OK

    def test_warning_lineage(self) -> None:
        m = check_lineage_completeness(100, 90, "narrative", now=NOW)
        assert m.severity == SEVERITY_WARNING

    def test_critical_lineage(self) -> None:
        m = check_lineage_completeness(100, 80, "narrative", now=NOW)
        assert m.severity == SEVERITY_CRITICAL

    def test_empty_objects(self) -> None:
        """No objects → perfect lineage (nothing to measure)."""
        m = check_lineage_completeness(0, 0, "narrative", now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 1.0

    def test_details(self) -> None:
        m = check_lineage_completeness(100, 92, "filing", now=NOW)
        assert m.details["total_objects"] == 100
        assert m.details["objects_with_lineage"] == 92
        assert m.details["missing_lineage"] == 8

    def test_custom_thresholds(self) -> None:
        m = check_lineage_completeness(
            100, 92, "narrative",
            warning_threshold=0.99, critical_threshold=0.95,
            now=NOW,
        )
        assert m.severity == SEVERITY_CRITICAL

    def test_to_dict(self) -> None:
        m = check_lineage_completeness(100, 95, "narrative", now=NOW)
        d = m.to_dict()
        assert d["metric_type"] == "lineage_completeness"
        assert d["lane"] == "narrative"
        assert isinstance(d["measured_at"], str)


# -- Unresolved entities tests -------------------------------------------------


class TestUnresolvedEntities:
    """Fraction of entities failing concept resolution."""

    def test_all_resolved(self) -> None:
        m = check_unresolved_entities(100, 0, "narrative", now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 0.0

    def test_low_unresolved(self) -> None:
        m = check_unresolved_entities(100, 3, "narrative", now=NOW)
        assert m.severity == SEVERITY_OK

    def test_warning_unresolved(self) -> None:
        m = check_unresolved_entities(100, 10, "narrative", now=NOW)
        assert m.severity == SEVERITY_WARNING

    def test_critical_unresolved(self) -> None:
        m = check_unresolved_entities(100, 20, "narrative", now=NOW)
        assert m.severity == SEVERITY_CRITICAL

    def test_empty_entities(self) -> None:
        m = check_unresolved_entities(0, 0, "filing", now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 0.0

    def test_details(self) -> None:
        m = check_unresolved_entities(200, 15, "filing", now=NOW)
        assert m.details["total_entities"] == 200
        assert m.details["unresolved_count"] == 15
        assert m.details["resolved_count"] == 185


# -- Filing parse quality tests ------------------------------------------------


class TestFilingParseQuality:
    """Fraction of filings parsed successfully."""

    def test_perfect_parsing(self) -> None:
        m = check_filing_parse_quality(50, 50, 0, now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 1.0
        assert m.lane == "filing"

    def test_good_parsing(self) -> None:
        m = check_filing_parse_quality(100, 92, 8, now=NOW)
        assert m.severity == SEVERITY_OK

    def test_warning_parsing(self) -> None:
        m = check_filing_parse_quality(100, 80, 20, now=NOW)
        assert m.severity == SEVERITY_WARNING

    def test_critical_parsing(self) -> None:
        m = check_filing_parse_quality(100, 60, 40, now=NOW)
        assert m.severity == SEVERITY_CRITICAL

    def test_no_filings(self) -> None:
        m = check_filing_parse_quality(0, 0, 0, now=NOW)
        assert m.severity == SEVERITY_OK

    def test_details_include_skipped(self) -> None:
        m = check_filing_parse_quality(100, 85, 10, now=NOW)
        assert m.details["skipped_count"] == 5  # 100 - 85 - 10

    def test_custom_lane(self) -> None:
        m = check_filing_parse_quality(50, 50, 0, lane="custom", now=NOW)
        assert m.lane == "custom"


# -- Stale evidence tests ------------------------------------------------------


class TestStaleEvidence:
    """Fraction of assertions with old evidence."""

    def test_no_stale(self) -> None:
        m = check_stale_evidence(100, 0, 90.0, "narrative", now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 0.0

    def test_low_stale(self) -> None:
        m = check_stale_evidence(100, 8, 90.0, "narrative", now=NOW)
        assert m.severity == SEVERITY_OK

    def test_warning_stale(self) -> None:
        m = check_stale_evidence(100, 15, 90.0, "filing", now=NOW)
        assert m.severity == SEVERITY_WARNING

    def test_critical_stale(self) -> None:
        m = check_stale_evidence(100, 30, 90.0, "structural", now=NOW)
        assert m.severity == SEVERITY_CRITICAL

    def test_empty_assertions(self) -> None:
        m = check_stale_evidence(0, 0, 90.0, "narrative", now=NOW)
        assert m.severity == SEVERITY_OK

    def test_details(self) -> None:
        m = check_stale_evidence(200, 40, 60.0, "filing", now=NOW)
        assert m.details["total_assertions"] == 200
        assert m.details["stale_count"] == 40
        assert m.details["fresh_count"] == 160
        assert m.details["stale_threshold_days"] == 60.0

    def test_message_includes_threshold(self) -> None:
        m = check_stale_evidence(100, 20, 90.0, "narrative", now=NOW)
        assert "90 days" in m.message


# -- Quality report tests ------------------------------------------------------


class TestQualityReport:
    """Report aggregation with worst-of severity."""

    def test_all_ok(self) -> None:
        metrics = [
            check_lineage_completeness(100, 100, "narrative", now=NOW),
            check_unresolved_entities(100, 0, "narrative", now=NOW),
        ]
        report = build_quality_report(metrics, now=NOW)
        assert report.overall_severity == SEVERITY_OK

    def test_worst_of_warning(self) -> None:
        metrics = [
            check_lineage_completeness(100, 100, "narrative", now=NOW),
            check_lineage_completeness(100, 90, "filing", now=NOW),
        ]
        report = build_quality_report(metrics, now=NOW)
        assert report.overall_severity == SEVERITY_WARNING

    def test_worst_of_critical(self) -> None:
        metrics = [
            check_lineage_completeness(100, 100, "narrative", now=NOW),
            check_stale_evidence(100, 30, 90.0, "filing", now=NOW),
        ]
        report = build_quality_report(metrics, now=NOW)
        assert report.overall_severity == SEVERITY_CRITICAL

    def test_empty_report(self) -> None:
        report = build_quality_report([], now=NOW)
        assert report.overall_severity == SEVERITY_OK

    def test_to_dict(self) -> None:
        metrics = [check_lineage_completeness(100, 95, "narrative", now=NOW)]
        report = build_quality_report(metrics, now=NOW)
        d = report.to_dict()
        assert d["overall_severity"] == SEVERITY_OK
        assert d["metric_count"] == 1
        assert len(d["metrics"]) == 1

    def test_measured_at(self) -> None:
        report = build_quality_report([], now=NOW)
        assert report.measured_at == NOW


# -- Dataclass tests -----------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass invariants."""

    def test_metric_frozen(self) -> None:
        m = check_lineage_completeness(100, 100, "narrative", now=NOW)
        try:
            m.value = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_report_frozen(self) -> None:
        report = build_quality_report([], now=NOW)
        try:
            report.measured_at = NOW  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_invalid_metric_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric_type"):
            QualityMetric(
                metric_type="bogus", lane="narrative",
                value=0.5, severity="ok",
            )

    def test_invalid_severity(self) -> None:
        with pytest.raises(ValueError, match="Invalid severity"):
            QualityMetric(
                metric_type="lineage_completeness", lane="narrative",
                value=0.5, severity="fatal",
            )

    def test_valid_metric_types_complete(self) -> None:
        assert len(VALID_METRIC_TYPES) == 10

    def test_skipped_count_clamped(self) -> None:
        """Inconsistent inputs don't produce negative skipped_count."""
        m = check_filing_parse_quality(10, 8, 5, now=NOW)
        assert m.details["skipped_count"] >= 0
