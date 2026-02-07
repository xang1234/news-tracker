"""Schema definitions for drift detection results.

Drift checks produce DriftResult objects that are collected into a DriftReport.
The report carries an overall severity (worst-of) so callers can quickly decide
whether to alert or log.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

DriftSeverity = Literal["ok", "warning", "critical"]

VALID_DRIFT_TYPES: frozenset[str] = frozenset({
    "embedding_drift",
    "theme_fragmentation",
    "sentiment_calibration",
    "cluster_stability",
})


@dataclass
class DriftResult:
    """Result from a single drift check.

    Attributes:
        drift_type: Which check produced this result.
        severity: ok / warning / critical.
        value: The measured metric value.
        thresholds: Dict mapping severity names to their thresholds.
        message: Human-readable explanation.
        metadata: Extra context (sample sizes, dates, etc.).
        checked_at: Timestamp of the check.
    """

    drift_type: str
    severity: DriftSeverity
    value: float
    thresholds: dict[str, float]
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if self.drift_type not in VALID_DRIFT_TYPES:
            raise ValueError(
                f"Invalid drift_type {self.drift_type!r}. "
                f"Must be one of: {sorted(VALID_DRIFT_TYPES)}"
            )


@dataclass
class DriftReport:
    """Aggregated report from one or more drift checks.

    Attributes:
        results: Individual check results.
    """

    results: list[DriftResult] = field(default_factory=list)

    @property
    def overall_severity(self) -> DriftSeverity:
        """Return the worst severity across all results."""
        if not self.results:
            return "ok"
        severities = {r.severity for r in self.results}
        if "critical" in severities:
            return "critical"
        if "warning" in severities:
            return "warning"
        return "ok"

    @property
    def has_issues(self) -> bool:
        """True if any result is not ok."""
        return self.overall_severity != "ok"
