"""Tests for decomposed narrative component scores.

Verifies that each component produces explainable sub-scores
from narrative run metrics, and that the composite correctly
blends all four.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.narrative.components import (
    ATTENTION_WEIGHT,
    CONFIRMATION_WEIGHT,
    CORROBORATION_WEIGHT,
    NOVELTY_WEIGHT,
    AttentionScore,
    ConfirmationScore,
    CorroborationScore,
    NarrativeComponents,
    NoveltyPersistenceScore,
    compute_attention,
    compute_confirmation,
    compute_corroboration,
    compute_narrative_components,
    compute_novelty_persistence,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Attention tests -------------------------------------------------------


class TestAttention:
    """Attention: velocity, acceleration, doc mass."""

    def test_high_velocity(self) -> None:
        score = compute_attention(
            current_rate_per_hour=50.0,
            current_acceleration=10.0,
            doc_count=30,
        )
        assert score.velocity > 0.8
        assert score.score > 0.5

    def test_zero_rate(self) -> None:
        score = compute_attention(0.0, 0.0, 0)
        assert score.velocity == 0.0
        assert score.acceleration == 0.0
        assert score.doc_mass == 0.0
        assert score.score == 0.0

    def test_clamped_to_one(self) -> None:
        score = compute_attention(
            current_rate_per_hour=999.0,
            current_acceleration=999.0,
            doc_count=999,
        )
        assert score.velocity <= 1.0
        assert score.acceleration <= 1.0
        assert score.doc_mass <= 1.0
        assert score.score <= 1.0

    def test_doc_mass_scales(self) -> None:
        low = compute_attention(10.0, 5.0, 5)
        high = compute_attention(10.0, 5.0, 40)
        assert high.doc_mass > low.doc_mass


# -- Corroboration tests ---------------------------------------------------


class TestCorroboration:
    """Corroboration: platform spread, source diversity, speed."""

    def test_multi_platform(self) -> None:
        score = compute_corroboration(
            platform_count=4,
            source_type_count=3,
            spread_hours=2.0,
        )
        assert score.platform_spread > 0.7
        assert score.source_diversity > 0.7
        assert score.cross_platform_speed > 0.5
        assert score.score > 0.5

    def test_single_platform(self) -> None:
        score = compute_corroboration(1, 1, None)
        assert score.platform_spread < 0.3
        assert score.cross_platform_speed == 0.0

    def test_fast_spread(self) -> None:
        fast = compute_corroboration(3, 2, 1.0)
        slow = compute_corroboration(3, 2, 5.0)
        assert fast.cross_platform_speed > slow.cross_platform_speed

    def test_no_spread_hours(self) -> None:
        score = compute_corroboration(3, 2, None)
        assert score.cross_platform_speed == 0.0


# -- Confirmation tests ----------------------------------------------------


class TestConfirmation:
    """Confirmation: authority alignment, crowd agreement."""

    def test_high_authority_strong_sentiment(self) -> None:
        score = compute_confirmation(
            avg_sentiment=0.8,
            avg_authority=0.9,
            high_authority_doc_ratio=0.6,
        )
        assert score.authority_alignment > 0.8
        assert score.crowd_agreement > 0.7
        assert score.score > 0.6

    def test_neutral_sentiment(self) -> None:
        score = compute_confirmation(0.0, 0.5, 0.3)
        assert score.crowd_agreement == 0.0

    def test_negative_sentiment_still_counts(self) -> None:
        """Absolute sentiment — strong negative = strong conviction."""
        score = compute_confirmation(-0.8, 0.7, 0.4)
        assert score.crowd_agreement > 0.7

    def test_low_authority(self) -> None:
        score = compute_confirmation(0.5, 0.1, 0.0)
        assert score.authority_alignment < 0.2
        assert score.authority_weight == 0.0


# -- Novelty/Persistence tests --------------------------------------------


class TestNoveltyPersistence:
    """Novelty/Persistence: recency, duration, balance."""

    def test_just_seen(self) -> None:
        score = compute_novelty_persistence(
            last_document_at=NOW,
            started_at=NOW - timedelta(hours=1),
            now=NOW,
        )
        assert score.recency >= 0.99
        assert score.score > 0.5

    def test_stale_evidence(self) -> None:
        score = compute_novelty_persistence(
            last_document_at=NOW - timedelta(days=5),
            started_at=NOW - timedelta(days=7),
            now=NOW,
        )
        # exp(-0.02 * 120) ≈ 0.09
        assert score.recency < 0.15

    def test_persistence_grows_with_duration(self) -> None:
        short = compute_novelty_persistence(
            last_document_at=NOW,
            started_at=NOW - timedelta(hours=1),
            now=NOW,
        )
        long = compute_novelty_persistence(
            last_document_at=NOW,
            started_at=NOW - timedelta(days=3),
            now=NOW,
        )
        assert long.persistence > short.persistence

    def test_persistence_diminishing_returns(self) -> None:
        """Logarithmic growth means 1 week is NOT 7x 1 day."""
        day1 = compute_novelty_persistence(
            last_document_at=NOW,
            started_at=NOW - timedelta(days=1),
            now=NOW,
        )
        week1 = compute_novelty_persistence(
            last_document_at=NOW,
            started_at=NOW - timedelta(days=7),
            now=NOW,
        )
        assert week1.persistence < day1.persistence * 3


# -- Composite tests -------------------------------------------------------


class TestComposite:
    """Full narrative component decomposition."""

    def test_all_components_present(self) -> None:
        result = compute_narrative_components(
            current_rate_per_hour=20.0,
            current_acceleration=8.0,
            doc_count=15,
            platform_count=3,
            avg_sentiment=0.6,
            avg_authority=0.7,
            last_document_at=NOW,
            started_at=NOW - timedelta(hours=2),
            now=NOW,
        )
        assert isinstance(result, NarrativeComponents)
        assert isinstance(result.attention, AttentionScore)
        assert isinstance(result.corroboration, CorroborationScore)
        assert isinstance(result.confirmation, ConfirmationScore)
        assert isinstance(result.novelty_persistence, NoveltyPersistenceScore)

    def test_composite_range(self) -> None:
        result = compute_narrative_components(
            current_rate_per_hour=20.0,
            current_acceleration=8.0,
            doc_count=15,
            platform_count=3,
            avg_sentiment=0.6,
            avg_authority=0.7,
            last_document_at=NOW,
            started_at=NOW - timedelta(hours=2),
            now=NOW,
        )
        assert 0.0 <= result.composite <= 100.0

    def test_composite_zero_inputs(self) -> None:
        result = compute_narrative_components(
            current_rate_per_hour=0.0,
            current_acceleration=0.0,
            doc_count=0,
            platform_count=0,
            avg_sentiment=0.0,
            avg_authority=0.0,
            last_document_at=NOW,
            started_at=NOW,
            now=NOW,
        )
        # Only novelty/persistence has non-zero (recency=1.0)
        assert result.composite < 20.0

    def test_weights_sum_to_one(self) -> None:
        total = (
            ATTENTION_WEIGHT + CORROBORATION_WEIGHT
            + CONFIRMATION_WEIGHT + NOVELTY_WEIGHT
        )
        assert abs(total - 1.0) < 1e-9

    def test_to_dict_serialization(self) -> None:
        result = compute_narrative_components(
            current_rate_per_hour=30.0,
            current_acceleration=10.0,
            doc_count=25,
            platform_count=4,
            avg_sentiment=0.7,
            avg_authority=0.8,
            last_document_at=NOW,
            started_at=NOW - timedelta(hours=6),
            now=NOW,
        )
        d = result.to_dict()
        assert "attention" in d
        assert "corroboration" in d
        assert "confirmation" in d
        assert "novelty_persistence" in d
        assert "composite" in d
        assert d["attention"]["velocity"] == result.attention.velocity

    def test_hot_narrative(self) -> None:
        """High-activity, multi-platform, authority-backed → high composite."""
        result = compute_narrative_components(
            current_rate_per_hour=50.0,
            current_acceleration=20.0,
            doc_count=40,
            platform_count=5,
            source_type_count=4,
            spread_hours=1.0,
            avg_sentiment=0.8,
            avg_authority=0.9,
            high_authority_doc_ratio=0.5,
            last_document_at=NOW,
            started_at=NOW - timedelta(hours=4),
            now=NOW,
        )
        assert result.composite > 60.0

    def test_cold_narrative(self) -> None:
        """Stale, single-platform, low-authority → low composite."""
        result = compute_narrative_components(
            current_rate_per_hour=1.0,
            current_acceleration=0.0,
            doc_count=3,
            platform_count=1,
            avg_sentiment=0.1,
            avg_authority=0.2,
            last_document_at=NOW - timedelta(days=3),
            started_at=NOW - timedelta(days=5),
            now=NOW,
        )
        assert result.composite < 20.0
