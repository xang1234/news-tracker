"""Tests for stateless alert trigger functions.

This is the most critical test file — each trigger function is tested
for boundary conditions, severity classification, and None handling.
"""

import pytest
from datetime import date, datetime, timezone

import numpy as np

from src.alerts.config import AlertConfig
from src.alerts.triggers import (
    check_all_triggers,
    check_extreme_sentiment,
    check_lifecycle_change,
    check_new_theme,
    check_sentiment_velocity,
    check_volume_surge,
)
from src.themes.schemas import Theme, ThemeMetrics
from src.themes.transitions import LifecycleTransition


@pytest.fixture
def config():
    return AlertConfig()


@pytest.fixture
def theme():
    return Theme(
        theme_id="theme_test123",
        name="test_ai_chips",
        centroid=np.zeros(768, dtype=np.float32),
        lifecycle_stage="emerging",
        document_count=25,
    )


def _make_metrics(
    theme_id: str = "theme_test123",
    target_date: date = date(2026, 2, 7),
    sentiment_score: float | None = 0.5,
    bullish_ratio: float | None = 0.6,
    volume_zscore: float | None = 1.0,
    document_count: int = 10,
) -> ThemeMetrics:
    return ThemeMetrics(
        theme_id=theme_id,
        date=target_date,
        document_count=document_count,
        sentiment_score=sentiment_score,
        bullish_ratio=bullish_ratio,
        volume_zscore=volume_zscore,
    )


# ── Sentiment Velocity ───────────────────────────────────


class TestCheckSentimentVelocity:
    """Test sentiment velocity trigger."""

    def test_no_alert_below_threshold(self, theme, config):
        today = _make_metrics(sentiment_score=0.5)
        yesterday = _make_metrics(sentiment_score=0.3)
        # delta = 0.2, threshold = 0.3
        result = check_sentiment_velocity(theme, today, yesterday, config)
        assert result is None

    def test_warning_at_threshold(self, theme, config):
        today = _make_metrics(sentiment_score=0.6)
        yesterday = _make_metrics(sentiment_score=0.2)
        # delta = 0.4, threshold = 0.3
        result = check_sentiment_velocity(theme, today, yesterday, config)
        assert result is not None
        assert result.severity == "warning"
        assert result.trigger_type == "sentiment_velocity"
        assert "bullish" in result.trigger_data["direction"]

    def test_critical_at_high_delta(self, theme, config):
        today = _make_metrics(sentiment_score=0.8)
        yesterday = _make_metrics(sentiment_score=0.1)
        # delta = 0.7, critical = 0.6
        result = check_sentiment_velocity(theme, today, yesterday, config)
        assert result is not None
        assert result.severity == "critical"

    def test_bearish_direction(self, theme, config):
        today = _make_metrics(sentiment_score=0.1)
        yesterday = _make_metrics(sentiment_score=0.5)
        # delta = -0.4
        result = check_sentiment_velocity(theme, today, yesterday, config)
        assert result is not None
        assert result.trigger_data["direction"] == "bearish"

    def test_none_today_score(self, theme, config):
        today = _make_metrics(sentiment_score=None)
        yesterday = _make_metrics(sentiment_score=0.5)
        result = check_sentiment_velocity(theme, today, yesterday, config)
        assert result is None

    def test_none_yesterday_score(self, theme, config):
        today = _make_metrics(sentiment_score=0.5)
        yesterday = _make_metrics(sentiment_score=None)
        result = check_sentiment_velocity(theme, today, yesterday, config)
        assert result is None

    def test_exact_threshold_boundary(self, theme, config):
        # Exactly at threshold: abs(0.3) < 0.3 is False → alert fires
        today = _make_metrics(sentiment_score=0.5)
        yesterday = _make_metrics(sentiment_score=0.2)
        # delta = 0.3, threshold = 0.3 → NOT less than, so alert fires
        result = check_sentiment_velocity(theme, today, yesterday, config)
        assert result is not None
        assert result.severity == "warning"

    def test_just_above_threshold(self, theme, config):
        today = _make_metrics(sentiment_score=0.501)
        yesterday = _make_metrics(sentiment_score=0.2)
        # delta = 0.301
        result = check_sentiment_velocity(theme, today, yesterday, config)
        assert result is not None

    def test_trigger_data_contents(self, theme, config):
        today = _make_metrics(sentiment_score=0.7)
        yesterday = _make_metrics(sentiment_score=0.2)
        result = check_sentiment_velocity(theme, today, yesterday, config)
        assert "delta" in result.trigger_data
        assert "today_score" in result.trigger_data
        assert "yesterday_score" in result.trigger_data
        assert "direction" in result.trigger_data


# ── Extreme Sentiment ────────────────────────────────────


class TestCheckExtremeSentiment:
    """Test extreme sentiment trigger."""

    def test_no_alert_normal_ratio(self, theme, config):
        metrics = _make_metrics(bullish_ratio=0.6)
        result = check_extreme_sentiment(theme, metrics, config)
        assert result is None

    def test_extreme_bullish(self, theme, config):
        metrics = _make_metrics(bullish_ratio=0.90)
        result = check_extreme_sentiment(theme, metrics, config)
        assert result is not None
        assert result.severity == "warning"
        assert result.trigger_data["condition"] == "extreme_bullish"

    def test_extreme_bearish(self, theme, config):
        metrics = _make_metrics(bullish_ratio=0.10)
        result = check_extreme_sentiment(theme, metrics, config)
        assert result is not None
        assert result.severity == "warning"
        assert result.trigger_data["condition"] == "extreme_bearish"

    def test_none_bullish_ratio(self, theme, config):
        metrics = _make_metrics(bullish_ratio=None)
        result = check_extreme_sentiment(theme, metrics, config)
        assert result is None

    def test_boundary_bullish(self, theme, config):
        # Exactly at threshold 0.85 should not fire (> not >=)
        metrics = _make_metrics(bullish_ratio=0.85)
        result = check_extreme_sentiment(theme, metrics, config)
        assert result is None

    def test_boundary_bearish(self, theme, config):
        # Exactly at threshold 0.15 should not fire (< not <=)
        metrics = _make_metrics(bullish_ratio=0.15)
        result = check_extreme_sentiment(theme, metrics, config)
        assert result is None

    def test_just_above_bullish_threshold(self, theme, config):
        metrics = _make_metrics(bullish_ratio=0.851)
        result = check_extreme_sentiment(theme, metrics, config)
        assert result is not None
        assert result.trigger_data["condition"] == "extreme_bullish"

    def test_just_below_bearish_threshold(self, theme, config):
        metrics = _make_metrics(bullish_ratio=0.149)
        result = check_extreme_sentiment(theme, metrics, config)
        assert result is not None
        assert result.trigger_data["condition"] == "extreme_bearish"


# ── Volume Surge ─────────────────────────────────────────


class TestCheckVolumeSurge:
    """Test volume surge trigger."""

    def test_no_alert_below_threshold(self, theme, config):
        metrics = _make_metrics(volume_zscore=2.5)
        result = check_volume_surge(theme, metrics, config)
        assert result is None

    def test_warning_at_threshold(self, theme, config):
        metrics = _make_metrics(volume_zscore=3.5)
        result = check_volume_surge(theme, metrics, config)
        assert result is not None
        assert result.severity == "warning"
        assert result.trigger_type == "volume_surge"

    def test_critical_at_high_zscore(self, theme, config):
        metrics = _make_metrics(volume_zscore=4.5)
        result = check_volume_surge(theme, metrics, config)
        assert result is not None
        assert result.severity == "critical"

    def test_none_volume_zscore(self, theme, config):
        metrics = _make_metrics(volume_zscore=None)
        result = check_volume_surge(theme, metrics, config)
        assert result is None

    def test_boundary_warning(self, theme, config):
        # Exactly at 3.0 should trigger (>=)
        metrics = _make_metrics(volume_zscore=3.0)
        result = check_volume_surge(theme, metrics, config)
        assert result is not None
        assert result.severity == "warning"

    def test_boundary_critical(self, theme, config):
        # Exactly at 4.0 should be critical (>=)
        metrics = _make_metrics(volume_zscore=4.0)
        result = check_volume_surge(theme, metrics, config)
        assert result is not None
        assert result.severity == "critical"

    def test_trigger_data_contents(self, theme, config):
        metrics = _make_metrics(volume_zscore=3.5, document_count=42)
        result = check_volume_surge(theme, metrics, config)
        assert result.trigger_data["volume_zscore"] == 3.5
        assert result.trigger_data["document_count"] == 42

    def test_negative_zscore_no_alert(self, theme, config):
        metrics = _make_metrics(volume_zscore=-2.0)
        result = check_volume_surge(theme, metrics, config)
        assert result is None


# ── Lifecycle Change ─────────────────────────────────────


class TestCheckLifecycleChange:
    """Test lifecycle change trigger."""

    def test_alertable_transition_gaining(self, config):
        transition = LifecycleTransition(
            theme_id="t1",
            from_stage="emerging",
            to_stage="accelerating",
            confidence=0.8,
        )
        result = check_lifecycle_change(transition, "AI Chips", config)
        assert result is not None
        assert result.severity == "critical"
        assert result.trigger_type == "lifecycle_change"

    def test_alertable_transition_fading(self, config):
        transition = LifecycleTransition(
            theme_id="t1",
            from_stage="mature",
            to_stage="fading",
            confidence=0.7,
        )
        result = check_lifecycle_change(transition, "Old Theme", config)
        assert result is not None
        assert result.severity == "warning"

    def test_non_alertable_transition(self, config):
        transition = LifecycleTransition(
            theme_id="t1",
            from_stage="fading",
            to_stage="emerging",
            confidence=0.5,
        )
        result = check_lifecycle_change(transition, "Revival", config)
        assert result is None

    def test_trigger_data_contents(self, config):
        transition = LifecycleTransition(
            theme_id="t1",
            from_stage="accelerating",
            to_stage="mature",
            confidence=0.9,
        )
        result = check_lifecycle_change(transition, "Peak Theme", config)
        assert result.trigger_data["from_stage"] == "accelerating"
        assert result.trigger_data["to_stage"] == "mature"
        assert result.trigger_data["confidence"] == 0.9


# ── New Theme ────────────────────────────────────────────


class TestCheckNewTheme:
    """Test new theme trigger."""

    def test_always_fires(self):
        result = check_new_theme("theme_new1", "Emerging AI Chips")
        assert result is not None
        assert result.severity == "info"
        assert result.trigger_type == "new_theme"

    def test_message_contains_name(self):
        result = check_new_theme("t1", "Quantum Computing")
        assert "Quantum Computing" in result.message

    def test_trigger_data(self):
        result = check_new_theme("theme_xyz", "Test Theme")
        assert result.trigger_data["theme_id"] == "theme_xyz"


# ── check_all_triggers ───────────────────────────────────


class TestCheckAllTriggers:
    """Test the aggregation function."""

    def test_empty_when_normal(self, theme, config):
        today = _make_metrics(
            sentiment_score=0.5, bullish_ratio=0.6, volume_zscore=1.0,
        )
        yesterday = _make_metrics(sentiment_score=0.4)
        result = check_all_triggers(theme, today, yesterday, config)
        assert result == []

    def test_multiple_triggers(self, theme, config):
        today = _make_metrics(
            sentiment_score=0.9,
            bullish_ratio=0.95,
            volume_zscore=4.5,
        )
        yesterday = _make_metrics(sentiment_score=0.1)
        result = check_all_triggers(theme, today, yesterday, config)
        types = {a.trigger_type for a in result}
        assert "sentiment_velocity" in types
        assert "extreme_sentiment" in types
        assert "volume_surge" in types

    def test_no_yesterday_skips_velocity(self, theme, config):
        today = _make_metrics(
            sentiment_score=0.9, bullish_ratio=0.6, volume_zscore=1.0,
        )
        result = check_all_triggers(theme, today, None, config)
        types = {a.trigger_type for a in result}
        assert "sentiment_velocity" not in types

    def test_single_trigger_only(self, theme, config):
        today = _make_metrics(
            sentiment_score=0.5, bullish_ratio=0.6, volume_zscore=3.5,
        )
        yesterday = _make_metrics(sentiment_score=0.4)
        result = check_all_triggers(theme, today, yesterday, config)
        assert len(result) == 1
        assert result[0].trigger_type == "volume_surge"
