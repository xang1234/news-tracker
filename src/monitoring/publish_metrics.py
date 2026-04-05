"""Publish boundary metrics: seal rates, bundle integrity, coverage drift, and contract compat.

Instruments the publish boundary where producer mistakes become
consumer outages. Four metric types:

    - Manifest seal rate: fraction of manifests successfully sealed
    - Bundle integrity: are exported bundles valid (checksums match)?
    - Coverage drift: are all expected lanes represented in composites?
    - Contract compatibility: is the published version supported?

All functions are stateless — the caller provides pre-aggregated
counts and state, the checker classifies them.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.contracts.intelligence.lanes import ALL_LANES
from src.contracts.intelligence.ownership import check_compatibility
from src.contracts.intelligence.version import ContractRegistry
from src.monitoring.quality_metrics import (
    SEVERITY_CRITICAL,
    SEVERITY_OK,
    SEVERITY_WARNING,
    QualityMetric,
    QualityReport,
    _classify,
)


# -- Default thresholds -------------------------------------------------------

# Manifest seal rate: fraction of created manifests that get sealed
DEFAULT_SEAL_WARNING = 0.90
DEFAULT_SEAL_CRITICAL = 0.75

# Bundle integrity: fraction of bundles passing checksum verification
DEFAULT_INTEGRITY_WARNING = 0.99
DEFAULT_INTEGRITY_CRITICAL = 0.95

# Coverage: minimum object count per lane to be "covered"
DEFAULT_MIN_LANE_OBJECTS = 1

# Coverage drift: fraction of expected lanes present in composite
DEFAULT_COVERAGE_WARNING = 0.75
DEFAULT_COVERAGE_CRITICAL = 0.50


# -- Check functions (stateless) -----------------------------------------------


def check_manifest_seal_rate(
    total_manifests: int,
    sealed_count: int,
    lane: str,
    *,
    warning_threshold: float = DEFAULT_SEAL_WARNING,
    critical_threshold: float = DEFAULT_SEAL_CRITICAL,
    now: datetime | None = None,
) -> QualityMetric:
    """Check the fraction of manifests successfully sealed.

    Unsealed manifests indicate publication pipeline failures —
    objects were added but the manifest was never finalized.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    rate = sealed_count / total_manifests if total_manifests > 0 else 1.0
    severity = _classify(rate, warning_threshold, critical_threshold)

    return QualityMetric(
        metric_type="manifest_seal_rate",
        lane=lane,
        value=rate,
        severity=severity,
        thresholds={"warning": warning_threshold, "critical": critical_threshold},
        message=(
            f"{lane}: {rate:.1%} seal rate "
            f"({sealed_count}/{total_manifests})"
        ),
        details={
            "total_manifests": total_manifests,
            "sealed_count": sealed_count,
            "unsealed_count": total_manifests - sealed_count,
        },
        measured_at=now,
    )


def check_bundle_integrity(
    total_bundles: int,
    valid_count: int,
    *,
    warning_threshold: float = DEFAULT_INTEGRITY_WARNING,
    critical_threshold: float = DEFAULT_INTEGRITY_CRITICAL,
    now: datetime | None = None,
) -> QualityMetric:
    """Check the fraction of bundles passing checksum verification.

    Invalid bundles indicate export pipeline corruption — the
    content doesn't match its recorded checksum.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    rate = valid_count / total_bundles if total_bundles > 0 else 1.0
    severity = _classify(rate, warning_threshold, critical_threshold)

    return QualityMetric(
        metric_type="bundle_integrity",
        lane="all",
        value=rate,
        severity=severity,
        thresholds={"warning": warning_threshold, "critical": critical_threshold},
        message=(
            f"Bundle integrity: {rate:.1%} valid "
            f"({valid_count}/{total_bundles})"
        ),
        details={
            "total_bundles": total_bundles,
            "valid_count": valid_count,
            "invalid_count": total_bundles - valid_count,
        },
        measured_at=now,
    )


def check_coverage_drift(
    lane_object_counts: dict[str, int],
    expected_lanes: tuple[str, ...] | None = None,
    *,
    min_objects: int = DEFAULT_MIN_LANE_OBJECTS,
    warning_threshold: float = DEFAULT_COVERAGE_WARNING,
    critical_threshold: float = DEFAULT_COVERAGE_CRITICAL,
    now: datetime | None = None,
) -> QualityMetric:
    """Check whether all expected lanes are represented in a composite.

    Coverage drift occurs when a lane quietly stops producing
    objects without triggering a hard failure. This check ensures
    all expected lanes have at least min_objects in the composite.

    Args:
        lane_object_counts: Objects per lane in the latest composite.
        expected_lanes: Which lanes should be present (default: ALL_LANES).
        min_objects: Minimum objects for a lane to be "covered."
        warning_threshold: Coverage fraction for warning.
        critical_threshold: Coverage fraction for critical.
        now: Measurement timestamp.

    Returns:
        QualityMetric with coverage_drift type.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if expected_lanes is None:
        expected_lanes = ALL_LANES

    covered = sum(
        1 for lane in expected_lanes
        if lane_object_counts.get(lane, 0) >= min_objects
    )
    total_expected = len(expected_lanes)
    rate = covered / total_expected if total_expected > 0 else 1.0
    severity = _classify(rate, warning_threshold, critical_threshold)

    missing = [
        lane for lane in expected_lanes
        if lane_object_counts.get(lane, 0) < min_objects
    ]

    return QualityMetric(
        metric_type="coverage_drift",
        lane="all",
        value=rate,
        severity=severity,
        thresholds={"warning": warning_threshold, "critical": critical_threshold},
        message=(
            f"Coverage: {covered}/{total_expected} lanes covered"
            + (f" (missing: {', '.join(missing)})" if missing else "")
        ),
        details={
            "covered_lanes": covered,
            "expected_lanes": total_expected,
            "missing_lanes": missing,
            "lane_object_counts": lane_object_counts,
            "min_objects": min_objects,
        },
        measured_at=now,
    )


def check_contract_compat(
    published_version: str,
    *,
    now: datetime | None = None,
) -> QualityMetric:
    """Check whether the published contract version is supported.

    Evaluates the version against the registry: unsupported or
    incompatible versions are critical, deprecated versions are
    warnings, and current/compatible versions are ok.

    Args:
        published_version: Contract version string being published.
        now: Measurement timestamp.

    Returns:
        QualityMetric with contract_compat type.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    result = check_compatibility(published_version)

    # Categorical classification — not a continuous rate, so _classify
    # doesn't apply here. Incompatible = critical, deprecated = warning.
    if not result.compatible:
        severity = SEVERITY_CRITICAL
    elif ContractRegistry.is_deprecated(result.checked):
        severity = SEVERITY_WARNING
    else:
        severity = SEVERITY_OK

    return QualityMetric(
        metric_type="contract_compat",
        lane="all",
        value=1.0 if result.compatible else 0.0,
        severity=severity,
        thresholds={},
        message=result.message,
        details={
            "published_version": str(result.checked),
            "current_version": str(result.current),
            "compatible": result.compatible,
            "deprecated": ContractRegistry.is_deprecated(result.checked),
        },
        measured_at=now,
    )


# -- Report builder -----------------------------------------------------------


def check_publish_boundary(
    seal_rates: dict[str, tuple[int, int]],
    bundle_stats: tuple[int, int],
    lane_object_counts: dict[str, int],
    published_version: str,
    *,
    now: datetime | None = None,
) -> QualityReport:
    """Run all publish boundary checks and assemble a report.

    Args:
        seal_rates: Per-lane (total_manifests, sealed_count).
        bundle_stats: (total_bundles, valid_count).
        lane_object_counts: Objects per lane in latest composite.
        published_version: Contract version being published.
        now: Measurement timestamp.

    Returns:
        QualityReport with publish boundary metrics.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    metrics: list[QualityMetric] = []

    for lane in sorted(seal_rates.keys()):
        total, sealed = seal_rates[lane]
        metrics.append(
            check_manifest_seal_rate(total, sealed, lane, now=now)
        )

    total_bundles, valid = bundle_stats
    metrics.append(
        check_bundle_integrity(total_bundles, valid, now=now)
    )

    metrics.append(
        check_coverage_drift(lane_object_counts, now=now)
    )

    metrics.append(
        check_contract_compat(published_version, now=now)
    )

    return QualityReport(metrics=metrics, measured_at=now)
