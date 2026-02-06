"""Tests for theme lifecycle classifier and transition detection."""

from datetime import date, datetime, timezone

import numpy as np
import pytest

from src.themes.lifecycle import (
    ACCELERATING_VELOCITY_THRESHOLD,
    EMERGING_DOC_CEILING,
    FADING_VELOCITY_THRESHOLD,
    LifecycleClassifier,
)
from src.themes.schemas import Theme, ThemeMetrics
from src.themes.transitions import (
    ALERTABLE_TRANSITIONS,
    LifecycleTransition,
)


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def classifier() -> LifecycleClassifier:
    return LifecycleClassifier()


@pytest.fixture
def centroid() -> np.ndarray:
    rng = np.random.default_rng(99)
    vec = rng.standard_normal(768).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_theme(
    centroid: np.ndarray,
    document_count: int = 25,
    lifecycle_stage: str = "emerging",
) -> Theme:
    return Theme(
        theme_id="theme_test123456",
        name="test_theme",
        centroid=centroid,
        document_count=document_count,
        lifecycle_stage=lifecycle_stage,
    )


def _make_metrics(
    theme_id: str = "theme_test123456",
    days: int = 5,
    base_count: int = 10,
    velocity: float = 0.1,
    count_growth: int = 2,
) -> list[ThemeMetrics]:
    """Build a metrics history with configurable trends."""
    return [
        ThemeMetrics(
            theme_id=theme_id,
            date=date(2025, 6, 10 + i),
            document_count=base_count + i * count_growth,
            velocity=velocity + i * 0.05,
        )
        for i in range(days)
    ]


# ── LifecycleClassifier.classify ────────────────────────────


class TestClassifyEmerging:
    """Tests for the 'emerging' classification path."""

    def test_insufficient_history_returns_emerging(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """< 3 days of metrics → emerging with low confidence."""
        theme = _make_theme(centroid, document_count=10)
        metrics = _make_metrics(days=2)
        stage, confidence = classifier.classify(theme, metrics)
        assert stage == "emerging"
        assert confidence == 0.5

    def test_empty_history_returns_emerging(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        theme = _make_theme(centroid)
        stage, confidence = classifier.classify(theme, [])
        assert stage == "emerging"
        assert confidence == 0.5

    def test_low_doc_count_positive_velocity(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """Low doc count + positive velocity trend → emerging."""
        theme = _make_theme(centroid, document_count=30)
        metrics = _make_metrics(days=5, velocity=0.1)
        stage, confidence = classifier.classify(theme, metrics)
        assert stage == "emerging"
        assert 0.5 < confidence <= 1.0

    def test_at_doc_ceiling_not_emerging(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """At or above doc ceiling → not classified as emerging."""
        theme = _make_theme(centroid, document_count=EMERGING_DOC_CEILING)
        metrics = _make_metrics(days=5, velocity=0.1)
        stage, _conf = classifier.classify(theme, metrics)
        assert stage != "emerging"

    def test_emerging_confidence_inversely_proportional_to_doc_count(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """Lower doc count → higher confidence in 'emerging'."""
        metrics = _make_metrics(days=5, velocity=0.1)

        theme_small = _make_theme(centroid, document_count=5)
        theme_mid = _make_theme(centroid, document_count=40)

        _, conf_small = classifier.classify(theme_small, metrics)
        _, conf_mid = classifier.classify(theme_mid, metrics)

        assert conf_small > conf_mid


class TestClassifyAccelerating:
    """Tests for the 'accelerating' classification path."""

    def test_high_velocity_positive_volume(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """High velocity trend + positive volume → accelerating."""
        theme = _make_theme(centroid, document_count=100)
        # Steep velocity growth: [0.1, 0.5, 1.0, 2.0, 4.0] — last 3 give
        # normalized trend well above 0.5
        metrics = [
            ThemeMetrics(
                theme_id="theme_test123456",
                date=date(2025, 6, 10 + i),
                document_count=50 + i * 20,
                velocity=0.1 * (2 ** i),
            )
            for i in range(5)
        ]
        stage, confidence = classifier.classify(theme, metrics)
        assert stage == "accelerating"
        assert confidence > 0.6

    def test_velocity_just_above_threshold(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """Velocity trend marginally above threshold still classifies."""
        theme = _make_theme(centroid, document_count=200)
        # Velocity doubles each day — steep relative growth
        metrics = [
            ThemeMetrics(
                theme_id="theme_test123456",
                date=date(2025, 6, 10 + i),
                document_count=50 + i * 10,
                velocity=0.2 * (2 ** i),
            )
            for i in range(5)
        ]
        stage, _conf = classifier.classify(theme, metrics)
        assert stage == "accelerating"


class TestClassifyFading:
    """Tests for the 'fading' classification path."""

    def test_negative_velocity_trend(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """Strongly negative velocity → fading."""
        theme = _make_theme(centroid, document_count=200)
        # Declining velocity
        metrics = [
            ThemeMetrics(
                theme_id="theme_test123456",
                date=date(2025, 6, 10 + i),
                document_count=100 - i * 5,
                velocity=0.5 - i * 0.5,
            )
            for i in range(5)
        ]
        stage, confidence = classifier.classify(theme, metrics)
        assert stage == "fading"
        assert confidence > 0.6

    def test_strongly_negative_velocity_high_confidence(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """Very negative velocity → high confidence fading."""
        theme = _make_theme(centroid, document_count=500)
        # Velocity crosses zero: [0.4, 0.2, 0.0, -0.2, -0.4]
        # Crossing zero gives a small mean, amplifying the normalized trend
        metrics = [
            ThemeMetrics(
                theme_id="theme_test123456",
                date=date(2025, 6, 10 + i),
                document_count=200 - i * 10,
                velocity=0.4 - i * 0.2,
            )
            for i in range(5)
        ]
        stage, confidence = classifier.classify(theme, metrics)
        assert stage == "fading"
        assert confidence >= 0.8


class TestClassifyMature:
    """Tests for the 'mature' (default) classification path."""

    def test_stable_metrics_returns_mature(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """No strong trend signals → mature."""
        theme = _make_theme(centroid, document_count=200)
        # Flat velocity and volume
        metrics = [
            ThemeMetrics(
                theme_id="theme_test123456",
                date=date(2025, 6, 10 + i),
                document_count=100,
                velocity=0.05,
            )
            for i in range(5)
        ]
        stage, confidence = classifier.classify(theme, metrics)
        assert stage == "mature"
        assert confidence == 0.7

    def test_high_doc_count_flat_velocity(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """Large theme with stable velocity → mature."""
        theme = _make_theme(centroid, document_count=1000)
        metrics = _make_metrics(days=5, velocity=0.0, count_growth=0)
        stage, _conf = classifier.classify(theme, metrics)
        assert stage == "mature"


class TestClassifyNoneVelocity:
    """Tests for metrics with None velocity values."""

    def test_all_none_velocity_returns_mature_or_emerging(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        """All None velocities → falls through to mature (high doc count)."""
        theme = _make_theme(centroid, document_count=200)
        metrics = [
            ThemeMetrics(
                theme_id="theme_test123456",
                date=date(2025, 6, 10 + i),
                document_count=50,
                velocity=None,
            )
            for i in range(5)
        ]
        stage, _conf = classifier.classify(theme, metrics)
        # No velocity data, high doc_count → should be mature
        assert stage == "mature"


# ── LifecycleClassifier._compute_trend ──────────────────────


class TestComputeTrend:
    """Tests for the trend computation helper."""

    def test_increasing_values_positive_trend(
        self, classifier: LifecycleClassifier
    ) -> None:
        trend = classifier._compute_trend([1, 2, 3, 4, 5])
        assert trend > 0

    def test_decreasing_values_negative_trend(
        self, classifier: LifecycleClassifier
    ) -> None:
        trend = classifier._compute_trend([5, 4, 3, 2, 1])
        assert trend < 0

    def test_constant_values_zero_trend(
        self, classifier: LifecycleClassifier
    ) -> None:
        trend = classifier._compute_trend([3, 3, 3, 3])
        assert trend == 0.0

    def test_single_value_returns_zero(
        self, classifier: LifecycleClassifier
    ) -> None:
        assert classifier._compute_trend([42]) == 0.0

    def test_empty_returns_zero(
        self, classifier: LifecycleClassifier
    ) -> None:
        assert classifier._compute_trend([]) == 0.0

    def test_two_values_positive(
        self, classifier: LifecycleClassifier
    ) -> None:
        trend = classifier._compute_trend([10, 20])
        assert trend > 0

    def test_all_zeros_returns_zero(
        self, classifier: LifecycleClassifier
    ) -> None:
        """All zeros → zero mean, slope normalized to raw slope."""
        trend = classifier._compute_trend([0, 0, 0])
        assert trend == 0.0


# ── LifecycleClassifier.detect_transition ───────────────────


class TestDetectTransition:
    """Tests for transition detection."""

    def test_no_change_returns_none(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        theme = _make_theme(centroid, lifecycle_stage="mature")
        result = classifier.detect_transition(theme, "mature")
        assert result is None

    def test_stage_change_returns_transition(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        theme = _make_theme(centroid, lifecycle_stage="emerging")
        result = classifier.detect_transition(theme, "accelerating", confidence=0.85)
        assert result is not None
        assert result.from_stage == "emerging"
        assert result.to_stage == "accelerating"
        assert result.confidence == 0.85
        assert result.theme_id == theme.theme_id

    def test_transition_has_timestamp(
        self, classifier: LifecycleClassifier, centroid: np.ndarray
    ) -> None:
        theme = _make_theme(centroid, lifecycle_stage="accelerating")
        result = classifier.detect_transition(theme, "mature")
        assert result is not None
        assert result.detected_at.tzinfo is not None


# ── LifecycleTransition ─────────────────────────────────────


class TestLifecycleTransition:
    """Tests for the LifecycleTransition dataclass."""

    def test_alertable_emerging_to_accelerating(self) -> None:
        t = LifecycleTransition(
            theme_id="theme_abc",
            from_stage="emerging",
            to_stage="accelerating",
        )
        assert t.is_alertable is True
        assert t.alert_message == "Theme gaining momentum"

    def test_alertable_any_to_fading(self) -> None:
        t = LifecycleTransition(
            theme_id="theme_abc",
            from_stage="mature",
            to_stage="fading",
        )
        assert t.is_alertable is True
        assert "momentum" in t.alert_message

    def test_not_alertable_fading_to_emerging(self) -> None:
        """Reverse transitions are not in the alert table."""
        t = LifecycleTransition(
            theme_id="theme_abc",
            from_stage="fading",
            to_stage="emerging",
        )
        assert t.is_alertable is False
        assert t.alert_message is None

    def test_accelerating_to_mature_alertable(self) -> None:
        t = LifecycleTransition(
            theme_id="theme_abc",
            from_stage="accelerating",
            to_stage="mature",
        )
        assert t.is_alertable is True
        assert "peaking" in t.alert_message

    def test_to_dict_includes_all_fields(self) -> None:
        t = LifecycleTransition(
            theme_id="theme_xyz",
            from_stage="emerging",
            to_stage="accelerating",
            confidence=0.9,
            metadata={"trigger": "velocity_spike"},
        )
        d = t.to_dict()
        assert d["theme_id"] == "theme_xyz"
        assert d["from_stage"] == "emerging"
        assert d["to_stage"] == "accelerating"
        assert d["confidence"] == 0.9
        assert d["is_alertable"] is True
        assert d["alert_message"] is not None
        assert d["metadata"]["trigger"] == "velocity_spike"
        assert "detected_at" in d

    def test_default_confidence_is_one(self) -> None:
        t = LifecycleTransition(
            theme_id="theme_abc",
            from_stage="emerging",
            to_stage="mature",
        )
        assert t.confidence == 1.0
