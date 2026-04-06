"""Tests for publish boundary metrics.

Verifies manifest seal rate, bundle integrity, coverage drift,
and contract compatibility checks.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.contracts.intelligence.version import ContractRegistry, ContractVersion
from src.monitoring.publish_metrics import (
    check_bundle_integrity,
    check_contract_compat,
    check_coverage_drift,
    check_manifest_seal_rate,
    check_publish_boundary,
)
from src.monitoring.quality_metrics import (
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Manifest seal rate tests --------------------------------------------------


class TestManifestSealRate:
    """Fraction of manifests successfully sealed."""

    def test_perfect_seal(self) -> None:
        m = check_manifest_seal_rate(10, 10, "narrative", now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 1.0
        assert m.metric_type == "manifest_seal_rate"
        assert m.lane == "narrative"

    def test_good_seal(self) -> None:
        m = check_manifest_seal_rate(20, 19, "filing", now=NOW)
        assert m.severity == SEVERITY_OK

    def test_warning_seal(self) -> None:
        m = check_manifest_seal_rate(20, 17, "narrative", now=NOW)
        assert m.severity == SEVERITY_WARNING

    def test_critical_seal(self) -> None:
        m = check_manifest_seal_rate(20, 14, "narrative", now=NOW)
        assert m.severity == SEVERITY_CRITICAL

    def test_no_manifests(self) -> None:
        m = check_manifest_seal_rate(0, 0, "narrative", now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 1.0

    def test_details(self) -> None:
        m = check_manifest_seal_rate(10, 8, "filing", now=NOW)
        assert m.details["total_manifests"] == 10
        assert m.details["sealed_count"] == 8
        assert m.details["unsealed_count"] == 2


# -- Bundle integrity tests ----------------------------------------------------


class TestBundleIntegrity:
    """Fraction of bundles passing checksum verification."""

    def test_all_valid(self) -> None:
        m = check_bundle_integrity(5, 5, now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 1.0
        assert m.metric_type == "bundle_integrity"
        assert m.lane == "all"

    def test_warning(self) -> None:
        m = check_bundle_integrity(100, 97, now=NOW)
        assert m.severity == SEVERITY_WARNING

    def test_critical(self) -> None:
        m = check_bundle_integrity(100, 90, now=NOW)
        assert m.severity == SEVERITY_CRITICAL

    def test_no_bundles(self) -> None:
        m = check_bundle_integrity(0, 0, now=NOW)
        assert m.severity == SEVERITY_OK

    def test_details(self) -> None:
        m = check_bundle_integrity(10, 9, now=NOW)
        assert m.details["invalid_count"] == 1


# -- Coverage drift tests ------------------------------------------------------


class TestCoverageDrift:
    """Lane representation in composite manifests."""

    def test_full_coverage(self) -> None:
        counts = {
            "narrative": 10,
            "filing": 5,
            "structural": 3,
            "backtest": 2,
        }
        m = check_coverage_drift(counts, now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 1.0
        assert m.metric_type == "coverage_drift"

    def test_missing_one_lane(self) -> None:
        counts = {
            "narrative": 10,
            "filing": 5,
            "structural": 3,
        }
        m = check_coverage_drift(counts, now=NOW)
        assert m.severity == SEVERITY_OK  # 3/4 = 75% >= warning
        assert "backtest" in m.details["missing_lanes"]

    def test_missing_two_lanes(self) -> None:
        counts = {"narrative": 10, "filing": 5}
        m = check_coverage_drift(counts, now=NOW)
        assert m.severity == SEVERITY_WARNING  # 2/4 = 50%

    def test_missing_three_lanes(self) -> None:
        counts = {"narrative": 10}
        m = check_coverage_drift(counts, now=NOW)
        assert m.severity == SEVERITY_CRITICAL  # 1/4 = 25%

    def test_empty_counts(self) -> None:
        m = check_coverage_drift({}, now=NOW)
        assert m.severity == SEVERITY_CRITICAL  # 0/4

    def test_zero_objects_not_covered(self) -> None:
        counts = {"narrative": 10, "filing": 0, "structural": 3, "backtest": 2}
        m = check_coverage_drift(counts, now=NOW)
        assert "filing" in m.details["missing_lanes"]

    def test_custom_expected_lanes(self) -> None:
        counts = {"narrative": 10}
        m = check_coverage_drift(
            counts, expected_lanes=("narrative",), now=NOW,
        )
        assert m.severity == SEVERITY_OK
        assert m.value == 1.0

    def test_custom_min_objects(self) -> None:
        counts = {"narrative": 2, "filing": 1, "structural": 1, "backtest": 1}
        m = check_coverage_drift(counts, min_objects=3, now=NOW)
        # only narrative has >= 3, so 1/4 = 25%
        assert m.severity == SEVERITY_CRITICAL

    def test_message_includes_missing(self) -> None:
        counts = {"narrative": 10}
        m = check_coverage_drift(counts, now=NOW)
        assert "missing" in m.message

    def test_no_missing_message(self) -> None:
        counts = {"narrative": 1, "filing": 1, "structural": 1, "backtest": 1}
        m = check_coverage_drift(counts, now=NOW)
        assert "missing" not in m.message


# -- Contract compat tests -----------------------------------------------------


class TestContractCompat:
    """Contract version compatibility checks."""

    def test_current_version_ok(self) -> None:
        m = check_contract_compat(str(ContractRegistry.CURRENT), now=NOW)
        assert m.severity == SEVERITY_OK
        assert m.value == 1.0
        assert m.metric_type == "contract_compat"

    def test_compatible_version_ok(self) -> None:
        # Same major, different minor
        current = ContractRegistry.CURRENT
        compat = ContractVersion(current.major, current.minor, current.patch)
        m = check_contract_compat(str(compat), now=NOW)
        assert m.severity == SEVERITY_OK

    def test_incompatible_major_critical(self) -> None:
        current = ContractRegistry.CURRENT
        incompat = ContractVersion(current.major + 1, 0, 0)
        m = check_contract_compat(str(incompat), now=NOW)
        assert m.severity == SEVERITY_CRITICAL
        assert m.value == 0.0

    def test_details(self) -> None:
        m = check_contract_compat(str(ContractRegistry.CURRENT), now=NOW)
        assert m.details["compatible"] is True
        assert "published_version" in m.details
        assert "current_version" in m.details


# -- Publish boundary report tests ---------------------------------------------


class TestPublishBoundary:
    """Full publish boundary report."""

    def test_healthy_boundary(self) -> None:
        report = check_publish_boundary(
            seal_rates={"narrative": (10, 10), "filing": (5, 5)},
            bundle_stats=(3, 3),
            lane_object_counts={"narrative": 10, "filing": 5, "structural": 3, "backtest": 2},
            published_version=str(ContractRegistry.CURRENT),
            now=NOW,
        )
        assert report.overall_severity == SEVERITY_OK

    def test_includes_all_metric_types(self) -> None:
        report = check_publish_boundary(
            seal_rates={"narrative": (10, 10)},
            bundle_stats=(3, 3),
            lane_object_counts={"narrative": 10, "filing": 5, "structural": 3, "backtest": 2},
            published_version=str(ContractRegistry.CURRENT),
            now=NOW,
        )
        types = {m.metric_type for m in report.metrics}
        assert "manifest_seal_rate" in types
        assert "bundle_integrity" in types
        assert "coverage_drift" in types
        assert "contract_compat" in types

    def test_critical_propagates(self) -> None:
        report = check_publish_boundary(
            seal_rates={"narrative": (10, 5)},
            bundle_stats=(3, 3),
            lane_object_counts={"narrative": 10, "filing": 5, "structural": 3, "backtest": 2},
            published_version=str(ContractRegistry.CURRENT),
            now=NOW,
        )
        assert report.overall_severity == SEVERITY_CRITICAL

    def test_empty_seal_rates(self) -> None:
        report = check_publish_boundary(
            seal_rates={},
            bundle_stats=(0, 0),
            lane_object_counts={},
            published_version=str(ContractRegistry.CURRENT),
            now=NOW,
        )
        # coverage drift will be critical (0/4 lanes), rest ok
        assert report.overall_severity == SEVERITY_CRITICAL


# -- Dataclass tests -----------------------------------------------------------


class TestDataclasses:
    """Frozen metric invariants."""

    def test_metric_frozen(self) -> None:
        m = check_manifest_seal_rate(10, 10, "narrative", now=NOW)
        try:
            m.value = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass
