"""Tests for lane freshness, quality status, and quarantine handling.

Verifies that each lane produces a health status that manifest
assembly can use to decide what is publishable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.publish.lane_health import (
    DEFAULT_FRESH_HOURS,
    FreshnessLevel,
    PublishReadiness,
    QualityLevel,
    QuarantineRecord,
    QuarantineState,
    compute_freshness,
    compute_lane_health,
    compute_quality,
    determine_readiness,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Freshness tests -------------------------------------------------------


class TestFreshness:
    """Lane freshness from last completed run."""

    def test_fresh(self) -> None:
        level, hours = compute_freshness(
            NOW - timedelta(hours=2), now=NOW
        )
        assert level == FreshnessLevel.FRESH
        assert hours is not None and hours < DEFAULT_FRESH_HOURS

    def test_aging(self) -> None:
        level, hours = compute_freshness(
            NOW - timedelta(hours=12), now=NOW
        )
        assert level == FreshnessLevel.AGING

    def test_stale(self) -> None:
        level, hours = compute_freshness(
            NOW - timedelta(hours=48), now=NOW
        )
        assert level == FreshnessLevel.STALE

    def test_unknown(self) -> None:
        level, hours = compute_freshness(None, now=NOW)
        assert level == FreshnessLevel.UNKNOWN
        assert hours is None

    def test_exactly_fresh_boundary(self) -> None:
        level, _ = compute_freshness(
            NOW - timedelta(hours=DEFAULT_FRESH_HOURS), now=NOW
        )
        assert level == FreshnessLevel.FRESH

    def test_just_past_fresh(self) -> None:
        level, _ = compute_freshness(
            NOW - timedelta(hours=DEFAULT_FRESH_HOURS + 0.1), now=NOW
        )
        assert level == FreshnessLevel.AGING

    def test_custom_thresholds(self) -> None:
        level, _ = compute_freshness(
            NOW - timedelta(hours=3), now=NOW,
            fresh_hours=1.0, aging_hours=2.0,
        )
        assert level == FreshnessLevel.STALE


# -- Quality tests ---------------------------------------------------------


class TestQuality:
    """Lane quality from pass/total counts."""

    def test_healthy(self) -> None:
        level, rate = compute_quality(95, 100)
        assert level == QualityLevel.HEALTHY
        assert rate == 0.95

    def test_degraded(self) -> None:
        level, rate = compute_quality(80, 100)
        assert level == QualityLevel.DEGRADED

    def test_critical(self) -> None:
        level, rate = compute_quality(50, 100)
        assert level == QualityLevel.CRITICAL

    def test_unknown_small_sample(self) -> None:
        level, rate = compute_quality(9, 9)
        assert level == QualityLevel.UNKNOWN
        assert rate is None

    def test_exactly_at_threshold(self) -> None:
        level, rate = compute_quality(90, 100)
        assert level == QualityLevel.HEALTHY

    def test_exactly_at_critical(self) -> None:
        level, rate = compute_quality(70, 100)
        assert level == QualityLevel.DEGRADED

    def test_custom_thresholds(self) -> None:
        level, _ = compute_quality(
            85, 100,
            quality_threshold=0.95,
            quality_critical=0.90,
        )
        assert level == QualityLevel.CRITICAL

    def test_zero_total(self) -> None:
        level, rate = compute_quality(0, 0)
        assert level == QualityLevel.UNKNOWN


# -- Readiness tests -------------------------------------------------------


class TestReadiness:
    """Publish readiness from health components."""

    def test_all_healthy(self) -> None:
        r = determine_readiness(
            FreshnessLevel.FRESH,
            QualityLevel.HEALTHY,
            QuarantineState.CLEAR,
        )
        assert r == PublishReadiness.READY

    def test_quarantined_blocks(self) -> None:
        r = determine_readiness(
            FreshnessLevel.FRESH,
            QualityLevel.HEALTHY,
            QuarantineState.QUARANTINED,
        )
        assert r == PublishReadiness.BLOCKED

    def test_stale_blocks(self) -> None:
        r = determine_readiness(
            FreshnessLevel.STALE,
            QualityLevel.HEALTHY,
            QuarantineState.CLEAR,
        )
        assert r == PublishReadiness.BLOCKED

    def test_critical_quality_blocks(self) -> None:
        r = determine_readiness(
            FreshnessLevel.FRESH,
            QualityLevel.CRITICAL,
            QuarantineState.CLEAR,
        )
        assert r == PublishReadiness.BLOCKED

    def test_aging_warns(self) -> None:
        r = determine_readiness(
            FreshnessLevel.AGING,
            QualityLevel.HEALTHY,
            QuarantineState.CLEAR,
        )
        assert r == PublishReadiness.WARN

    def test_degraded_warns(self) -> None:
        r = determine_readiness(
            FreshnessLevel.FRESH,
            QualityLevel.DEGRADED,
            QuarantineState.CLEAR,
        )
        assert r == PublishReadiness.WARN

    def test_watch_warns(self) -> None:
        r = determine_readiness(
            FreshnessLevel.FRESH,
            QualityLevel.HEALTHY,
            QuarantineState.WATCH,
        )
        assert r == PublishReadiness.WARN

    def test_unknown_freshness_with_healthy_quality(self) -> None:
        """Unknown freshness (no runs) is not stale — READY if rest ok."""
        r = determine_readiness(
            FreshnessLevel.UNKNOWN,
            QualityLevel.HEALTHY,
            QuarantineState.CLEAR,
        )
        assert r == PublishReadiness.READY

    def test_block_takes_priority_over_warn(self) -> None:
        r = determine_readiness(
            FreshnessLevel.STALE,
            QualityLevel.DEGRADED,
            QuarantineState.WATCH,
        )
        assert r == PublishReadiness.BLOCKED


# -- QuarantineRecord tests ------------------------------------------------


class TestQuarantineRecord:
    """Quarantine record construction."""

    def test_basic(self) -> None:
        q = QuarantineRecord(
            lane="narrative",
            reason="Dead-letter rate too high",
        )
        assert q.state == QuarantineState.QUARANTINED
        assert q.quarantined_by == "system"

    def test_watch_state(self) -> None:
        q = QuarantineRecord(
            lane="filing",
            reason="Provider intermittent",
            state=QuarantineState.WATCH,
            quarantined_by="operator",
        )
        assert q.state == QuarantineState.WATCH


# -- compute_lane_health integration tests ---------------------------------


class TestComputeLaneHealth:
    """End-to-end lane health computation."""

    def test_healthy_lane(self) -> None:
        status = compute_lane_health(
            "narrative",
            last_completed_at=NOW - timedelta(hours=1),
            passed=95,
            total=100,
            now=NOW,
        )
        assert status.freshness == FreshnessLevel.FRESH
        assert status.quality == QualityLevel.HEALTHY
        assert status.quarantine == QuarantineState.CLEAR
        assert status.readiness == PublishReadiness.READY

    def test_stale_lane(self) -> None:
        status = compute_lane_health(
            "filing",
            last_completed_at=NOW - timedelta(days=3),
            passed=90,
            total=100,
            now=NOW,
        )
        assert status.freshness == FreshnessLevel.STALE
        assert status.readiness == PublishReadiness.BLOCKED

    def test_quarantined_lane(self) -> None:
        q = QuarantineRecord(
            lane="structural",
            reason="Graph data corruption detected",
        )
        status = compute_lane_health(
            "structural",
            last_completed_at=NOW - timedelta(hours=1),
            passed=95,
            total=100,
            quarantine_record=q,
            now=NOW,
        )
        assert status.quarantine == QuarantineState.QUARANTINED
        assert status.readiness == PublishReadiness.BLOCKED
        assert status.quarantine_reason == "Graph data corruption detected"

    def test_degraded_quality(self) -> None:
        status = compute_lane_health(
            "narrative",
            last_completed_at=NOW - timedelta(hours=1),
            passed=75,
            total=100,
            now=NOW,
        )
        assert status.quality == QualityLevel.DEGRADED
        assert status.readiness == PublishReadiness.WARN

    def test_no_data(self) -> None:
        status = compute_lane_health("backtest", now=NOW)
        assert status.freshness == FreshnessLevel.UNKNOWN
        assert status.quality == QualityLevel.UNKNOWN
        assert status.readiness == PublishReadiness.READY

    def test_metadata_includes_hours(self) -> None:
        status = compute_lane_health(
            "narrative",
            last_completed_at=NOW - timedelta(hours=3),
            now=NOW,
        )
        assert status.hours_since_completion is not None
        assert abs(status.hours_since_completion - 3.0) < 0.01

    def test_quality_rate_present(self) -> None:
        status = compute_lane_health(
            "narrative",
            passed=85,
            total=100,
            now=NOW,
        )
        assert status.quality_rate == 0.85
        assert status.quality_sample_size == 100
