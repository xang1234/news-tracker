"""Quality metrics for the intelligence layer.

Measures lineage completeness, unresolved entity rates, filing
parse quality, and stale evidence — the core quality risks
introduced by claims, filings, and publication.

Follows the DriftCheck pattern: each check function receives
pre-aggregated counts, classifies severity using thresholds,
and returns a structured QualityMetric. Metrics compose into
a QualityReport with worst-of severity.

All functions are stateless — the caller fetches aggregate
counts from the database, the checker classifies them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# -- Severity levels ----------------------------------------------------------

SEVERITY_OK = "ok"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

SEVERITY_ORDER = {SEVERITY_OK: 0, SEVERITY_WARNING: 1, SEVERITY_CRITICAL: 2}


# -- Default thresholds -------------------------------------------------------

# Lineage completeness: fraction of objects with source lineage
DEFAULT_LINEAGE_WARNING = 0.95
DEFAULT_LINEAGE_CRITICAL = 0.85

# Unresolved entities: fraction of entities that failed resolution
DEFAULT_UNRESOLVED_WARNING = 0.05
DEFAULT_UNRESOLVED_CRITICAL = 0.15

# Filing parse quality: fraction of filings successfully parsed
DEFAULT_PARSE_WARNING = 0.90
DEFAULT_PARSE_CRITICAL = 0.75

# Stale evidence: fraction of assertions with old evidence
DEFAULT_STALE_WARNING = 0.10
DEFAULT_STALE_CRITICAL = 0.25


# -- QualityMetric dataclass ---------------------------------------------------


@dataclass(frozen=True)
class QualityMetric:
    """A single quality measurement for a lane or subsystem.

    Attributes:
        metric_type: What is being measured.
        lane: Which lane or "all" for cross-lane.
        value: The measured value (rate 0-1 or raw count).
        severity: "ok", "warning", or "critical".
        thresholds: Severity → threshold mapping for context.
        message: Human-readable description.
        details: Structured context for dashboards.
        measured_at: When this measurement was taken.
    """

    metric_type: str
    lane: str
    value: float
    severity: str
    thresholds: dict[str, float] = field(default_factory=dict)
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    measured_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_type": self.metric_type,
            "lane": self.lane,
            "value": round(self.value, 4),
            "severity": self.severity,
            "thresholds": self.thresholds,
            "message": self.message,
            "details": self.details,
            "measured_at": self.measured_at.isoformat(),
        }


# -- QualityReport dataclass ---------------------------------------------------


@dataclass(frozen=True)
class QualityReport:
    """Aggregation of quality metrics with worst-of severity.

    Attributes:
        metrics: Individual quality measurements.
        measured_at: When this report was assembled.
    """

    metrics: list[QualityMetric] = field(default_factory=list)
    measured_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def overall_severity(self) -> str:
        """Worst severity across all metrics."""
        if not self.metrics:
            return SEVERITY_OK
        return max(
            self.metrics,
            key=lambda m: SEVERITY_ORDER.get(m.severity, 0),
        ).severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_severity": self.overall_severity,
            "metric_count": len(self.metrics),
            "metrics": [m.to_dict() for m in self.metrics],
            "measured_at": self.measured_at.isoformat(),
        }


# -- Classification helper ----------------------------------------------------


def _classify(
    value: float,
    warning_threshold: float,
    critical_threshold: float,
    *,
    higher_is_better: bool = True,
) -> str:
    """Classify a value into ok/warning/critical.

    Args:
        value: The measured value.
        warning_threshold: Boundary between ok and warning.
        critical_threshold: Boundary between warning and critical.
        higher_is_better: If True, value >= warning is ok.
            If False, value <= warning is ok (for error rates).
    """
    if higher_is_better:
        if value >= warning_threshold:
            return SEVERITY_OK
        if value >= critical_threshold:
            return SEVERITY_WARNING
        return SEVERITY_CRITICAL
    else:
        if value <= warning_threshold:
            return SEVERITY_OK
        if value <= critical_threshold:
            return SEVERITY_WARNING
        return SEVERITY_CRITICAL


# -- Check functions (stateless) -----------------------------------------------


def check_lineage_completeness(
    total_objects: int,
    objects_with_lineage: int,
    lane: str,
    *,
    warning_threshold: float = DEFAULT_LINEAGE_WARNING,
    critical_threshold: float = DEFAULT_LINEAGE_CRITICAL,
    now: datetime | None = None,
) -> QualityMetric:
    """Check what fraction of published objects have source lineage.

    Lineage means the object has non-empty source_ids or run_id,
    enabling traceability from published output back to source data.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    rate = objects_with_lineage / total_objects if total_objects > 0 else 1.0
    severity = _classify(rate, warning_threshold, critical_threshold)
    missing = total_objects - objects_with_lineage

    return QualityMetric(
        metric_type="lineage_completeness",
        lane=lane,
        value=rate,
        severity=severity,
        thresholds={"warning": warning_threshold, "critical": critical_threshold},
        message=(
            f"{lane}: {rate:.1%} lineage completeness "
            f"({objects_with_lineage}/{total_objects}, {missing} missing)"
        ),
        details={
            "total_objects": total_objects,
            "objects_with_lineage": objects_with_lineage,
            "missing_lineage": missing,
        },
        measured_at=now,
    )


def check_unresolved_entities(
    total_entities: int,
    unresolved_count: int,
    lane: str,
    *,
    warning_threshold: float = DEFAULT_UNRESOLVED_WARNING,
    critical_threshold: float = DEFAULT_UNRESOLVED_CRITICAL,
    now: datetime | None = None,
) -> QualityMetric:
    """Check the fraction of entities that failed concept resolution.

    Unresolved entities are those where the resolver cascade
    (exact → alias → fuzzy → LLM) could not find a canonical concept.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    rate = unresolved_count / total_entities if total_entities > 0 else 0.0
    severity = _classify(
        rate, warning_threshold, critical_threshold, higher_is_better=False,
    )

    return QualityMetric(
        metric_type="unresolved_entities",
        lane=lane,
        value=rate,
        severity=severity,
        thresholds={"warning": warning_threshold, "critical": critical_threshold},
        message=(
            f"{lane}: {rate:.1%} unresolved entity rate "
            f"({unresolved_count}/{total_entities})"
        ),
        details={
            "total_entities": total_entities,
            "unresolved_count": unresolved_count,
            "resolved_count": total_entities - unresolved_count,
        },
        measured_at=now,
    )


def check_filing_parse_quality(
    total_filings: int,
    parsed_count: int,
    failed_count: int,
    lane: str = "filing",
    *,
    warning_threshold: float = DEFAULT_PARSE_WARNING,
    critical_threshold: float = DEFAULT_PARSE_CRITICAL,
    now: datetime | None = None,
) -> QualityMetric:
    """Check the fraction of filings that parsed successfully.

    Parse quality covers section extraction, XBRL fact parsing,
    and overall filing ingestion success rate.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    rate = parsed_count / total_filings if total_filings > 0 else 1.0
    severity = _classify(rate, warning_threshold, critical_threshold)

    return QualityMetric(
        metric_type="filing_parse_quality",
        lane=lane,
        value=rate,
        severity=severity,
        thresholds={"warning": warning_threshold, "critical": critical_threshold},
        message=(
            f"{lane}: {rate:.1%} parse success rate "
            f"({parsed_count}/{total_filings}, {failed_count} failed)"
        ),
        details={
            "total_filings": total_filings,
            "parsed_count": parsed_count,
            "failed_count": failed_count,
            "skipped_count": total_filings - parsed_count - failed_count,
        },
        measured_at=now,
    )


def check_stale_evidence(
    total_assertions: int,
    stale_count: int,
    stale_threshold_days: float,
    lane: str,
    *,
    warning_threshold: float = DEFAULT_STALE_WARNING,
    critical_threshold: float = DEFAULT_STALE_CRITICAL,
    now: datetime | None = None,
) -> QualityMetric:
    """Check the fraction of assertions with stale evidence.

    An assertion is stale when its last_evidence_at is older than
    stale_threshold_days. Stale assertions may reflect outdated
    relationships that should be revalidated or retracted.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    rate = stale_count / total_assertions if total_assertions > 0 else 0.0
    severity = _classify(
        rate, warning_threshold, critical_threshold, higher_is_better=False,
    )

    return QualityMetric(
        metric_type="stale_evidence",
        lane=lane,
        value=rate,
        severity=severity,
        thresholds={"warning": warning_threshold, "critical": critical_threshold},
        message=(
            f"{lane}: {rate:.1%} stale evidence rate "
            f"({stale_count}/{total_assertions}, "
            f"threshold={stale_threshold_days:.0f} days)"
        ),
        details={
            "total_assertions": total_assertions,
            "stale_count": stale_count,
            "fresh_count": total_assertions - stale_count,
            "stale_threshold_days": stale_threshold_days,
        },
        measured_at=now,
    )


# -- Report builder -----------------------------------------------------------


def build_quality_report(
    metrics: list[QualityMetric],
    *,
    now: datetime | None = None,
) -> QualityReport:
    """Assemble a quality report from individual metrics.

    The report's overall_severity is the worst across all metrics.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    return QualityReport(metrics=metrics, measured_at=now)
