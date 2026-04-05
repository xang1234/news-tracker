"""Lane freshness, quality status, and quarantine handling.

Tracks per-lane health so manifest assembly can reason about which
lanes are publishable. A lane's health is composed from:
    - Freshness: time since last completed run
    - Quality: fraction of recent objects passing quality checks
    - Quarantine: explicit operator or automated block on publication

Health is computed, not stored — the inputs (lane runs, quality
verdicts, quarantine records) live in their own tables. This module
provides the computation and decision logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


# -- Health status types ---------------------------------------------------


class FreshnessLevel(str, Enum):
    """How fresh a lane's data is."""

    FRESH = "fresh"          # Within expected cadence
    AGING = "aging"          # Approaching staleness threshold
    STALE = "stale"          # Past staleness threshold
    UNKNOWN = "unknown"      # No completed runs


class QualityLevel(str, Enum):
    """Overall quality assessment for a lane."""

    HEALTHY = "healthy"      # Quality rate above threshold
    DEGRADED = "degraded"    # Quality rate below threshold but above critical
    CRITICAL = "critical"    # Quality rate below critical threshold
    UNKNOWN = "unknown"      # No quality data available


class QuarantineState(str, Enum):
    """Whether a lane is quarantined from publication."""

    CLEAR = "clear"          # No quarantine — publishable
    QUARANTINED = "quarantined"  # Blocked from publication
    WATCH = "watch"          # Not blocked but under observation


class PublishReadiness(str, Enum):
    """Whether a lane is ready for manifest inclusion."""

    READY = "ready"          # Fresh + healthy + not quarantined
    WARN = "warn"            # Publishable with caveats (aging/degraded)
    BLOCKED = "blocked"      # Not publishable (stale/critical/quarantined)


# -- Configuration defaults ------------------------------------------------

DEFAULT_FRESH_HOURS = 6       # Max hours for "fresh"
DEFAULT_AGING_HOURS = 24      # Max hours for "aging" (beyond = stale)
DEFAULT_QUALITY_THRESHOLD = 0.9    # Min quality rate for "healthy"
DEFAULT_QUALITY_CRITICAL = 0.7     # Min quality rate to avoid "critical"
DEFAULT_MIN_QUALITY_SAMPLE = 10    # Min objects to compute quality rate


# -- LaneHealthStatus ------------------------------------------------------


@dataclass(frozen=True)
class LaneHealthStatus:
    """Comprehensive health status for a single lane.

    Attributes:
        lane: Which lane this health applies to.
        freshness: How recent the data is.
        quality: Quality assessment.
        quarantine: Quarantine state.
        readiness: Overall publish readiness.
        last_completed_at: When the last run completed.
        hours_since_completion: Hours since last completed run.
        quality_rate: Fraction of objects passing quality (0-1).
        quality_sample_size: How many objects were evaluated.
        quarantine_reason: Why the lane is quarantined (if applicable).
        metadata: Extensible metadata.
    """

    lane: str
    freshness: FreshnessLevel
    quality: QualityLevel
    quarantine: QuarantineState
    readiness: PublishReadiness
    last_completed_at: datetime | None = None
    hours_since_completion: float | None = None
    quality_rate: float | None = None
    quality_sample_size: int = 0
    quarantine_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# -- QuarantineRecord -----------------------------------------------------


@dataclass
class QuarantineRecord:
    """An explicit quarantine on a lane.

    Attributes:
        lane: The quarantined lane.
        reason: Why it was quarantined.
        quarantined_at: When the quarantine was set.
        quarantined_by: Who/what set it (operator, automated check).
        state: Current state (quarantined or watch).
        metadata: Extra context.
    """

    lane: str
    reason: str
    quarantined_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    quarantined_by: str = "system"
    state: QuarantineState = QuarantineState.QUARANTINED
    metadata: dict[str, Any] = field(default_factory=dict)


# -- Compute functions (stateless) -----------------------------------------


def compute_freshness(
    last_completed_at: datetime | None,
    *,
    now: datetime | None = None,
    fresh_hours: float = DEFAULT_FRESH_HOURS,
    aging_hours: float = DEFAULT_AGING_HOURS,
) -> tuple[FreshnessLevel, float | None]:
    """Compute lane freshness from last completed run time.

    Returns (level, hours_since) tuple.
    """
    if last_completed_at is None:
        return FreshnessLevel.UNKNOWN, None

    if now is None:
        now = datetime.now(timezone.utc)

    hours = (now - last_completed_at).total_seconds() / 3600
    hours = max(0.0, hours)

    if hours <= fresh_hours:
        return FreshnessLevel.FRESH, hours
    elif hours <= aging_hours:
        return FreshnessLevel.AGING, hours
    else:
        return FreshnessLevel.STALE, hours


def compute_quality(
    passed: int,
    total: int,
    *,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    quality_critical: float = DEFAULT_QUALITY_CRITICAL,
    min_sample: int = DEFAULT_MIN_QUALITY_SAMPLE,
) -> tuple[QualityLevel, float | None]:
    """Compute lane quality from pass/total counts.

    Returns (level, rate) tuple. Rate is None if sample too small.
    """
    if total < min_sample:
        return QualityLevel.UNKNOWN, None

    rate = passed / total
    if rate >= quality_threshold:
        return QualityLevel.HEALTHY, rate
    elif rate >= quality_critical:
        return QualityLevel.DEGRADED, rate
    else:
        return QualityLevel.CRITICAL, rate


def determine_readiness(
    freshness: FreshnessLevel,
    quality: QualityLevel,
    quarantine: QuarantineState,
) -> PublishReadiness:
    """Determine overall publish readiness from health components.

    BLOCKED if any component is in a blocking state.
    WARN if any component is in a warning state.
    READY only if all components are healthy.
    """
    if quarantine == QuarantineState.QUARANTINED:
        return PublishReadiness.BLOCKED
    if freshness == FreshnessLevel.STALE:
        return PublishReadiness.BLOCKED
    if quality == QualityLevel.CRITICAL:
        return PublishReadiness.BLOCKED

    if freshness == FreshnessLevel.AGING:
        return PublishReadiness.WARN
    if quality == QualityLevel.DEGRADED:
        return PublishReadiness.WARN
    if quarantine == QuarantineState.WATCH:
        return PublishReadiness.WARN

    return PublishReadiness.READY


def compute_lane_health(
    lane: str,
    *,
    last_completed_at: datetime | None = None,
    passed: int = 0,
    total: int = 0,
    quarantine_record: QuarantineRecord | None = None,
    now: datetime | None = None,
    fresh_hours: float = DEFAULT_FRESH_HOURS,
    aging_hours: float = DEFAULT_AGING_HOURS,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    quality_critical: float = DEFAULT_QUALITY_CRITICAL,
    min_sample: int = DEFAULT_MIN_QUALITY_SAMPLE,
) -> LaneHealthStatus:
    """Compute comprehensive health status for a lane.

    Main entry point. Takes raw inputs and returns a complete
    LaneHealthStatus with readiness verdict.
    """
    freshness, hours = compute_freshness(
        last_completed_at, now=now,
        fresh_hours=fresh_hours, aging_hours=aging_hours,
    )
    quality, rate = compute_quality(
        passed, total,
        quality_threshold=quality_threshold,
        quality_critical=quality_critical,
        min_sample=min_sample,
    )

    if quarantine_record is not None:
        quarantine = quarantine_record.state
        quarantine_reason = quarantine_record.reason
    else:
        quarantine = QuarantineState.CLEAR
        quarantine_reason = None

    readiness = determine_readiness(freshness, quality, quarantine)

    return LaneHealthStatus(
        lane=lane,
        freshness=freshness,
        quality=quality,
        quarantine=quarantine,
        readiness=readiness,
        last_completed_at=last_completed_at,
        hours_since_completion=round(hours, 2) if hours is not None else None,
        quality_rate=round(rate, 4) if rate is not None else None,
        quality_sample_size=total,
        quarantine_reason=quarantine_reason,
    )
