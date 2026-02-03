"""Tests for temporal sentiment aggregation."""

from datetime import datetime, timedelta, timezone

import pytest

from src.sentiment.aggregation import (
    AggregatedSentiment,
    DocumentSentiment,
    SentimentAggregator,
)
from src.sentiment.config import SentimentConfig


@pytest.fixture
def aggregator() -> SentimentAggregator:
    """Create aggregator with default settings."""
    return SentimentAggregator()


@pytest.fixture
def now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def make_doc(
    doc_id: str,
    timestamp: datetime,
    label: str,
    confidence: float = 0.9,
    authority: float | None = None,
    platform: str | None = None,
) -> DocumentSentiment:
    """Helper to create DocumentSentiment."""
    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    scores[label] = confidence
    # Distribute remaining probability
    remaining = 1.0 - confidence
    other_labels = [l for l in scores if l != label]
    for l in other_labels:
        scores[l] = remaining / 2

    return DocumentSentiment(
        document_id=doc_id,
        timestamp=timestamp,
        label=label,
        confidence=confidence,
        scores=scores,
        authority_score=authority,
        platform=platform,
    )


class TestDocumentSentiment:
    """Tests for DocumentSentiment dataclass."""

    def test_valid_creation(self, now):
        """Test creating valid DocumentSentiment."""
        doc = DocumentSentiment(
            document_id="d1",
            timestamp=now,
            label="positive",
            confidence=0.9,
            scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
        )
        assert doc.document_id == "d1"
        assert doc.label == "positive"

    def test_invalid_label_raises(self, now):
        """Test invalid label raises ValueError."""
        with pytest.raises(ValueError, match="Invalid label"):
            DocumentSentiment(
                document_id="d1",
                timestamp=now,
                label="invalid",
                confidence=0.9,
                scores={},
            )

    def test_invalid_confidence_raises(self, now):
        """Test confidence outside 0-1 raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be 0-1"):
            DocumentSentiment(
                document_id="d1",
                timestamp=now,
                label="positive",
                confidence=1.5,
                scores={},
            )


class TestAggregatedSentiment:
    """Tests for AggregatedSentiment dataclass."""

    def test_creation(self, now):
        """Test creating AggregatedSentiment."""
        agg = AggregatedSentiment(
            theme_id="AI_chips",
            ticker=None,
            window_start=now - timedelta(days=7),
            window_end=now,
            document_count=10,
            bullish_ratio=0.6,
            bearish_ratio=0.2,
            neutral_ratio=0.2,
            avg_confidence=0.85,
            avg_authority=0.7,
        )
        assert agg.theme_id == "AI_chips"
        assert agg.document_count == 10


class TestEmptyAggregation:
    """Tests for aggregation with no documents."""

    def test_empty_sentiments_returns_zero(self, aggregator, now):
        """Test empty document list returns zero ratios."""
        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=[],
            window_days=7,
            reference_time=now,
        )

        assert result.document_count == 0
        assert result.bullish_ratio == 0.0
        assert result.bearish_ratio == 0.0
        assert result.neutral_ratio == 0.0
        assert result.avg_confidence == 0.0
        assert result.sentiment_velocity is None
        assert result.extreme_sentiment is None


class TestSingleDocumentAggregation:
    """Tests for aggregation with single document."""

    def test_single_positive_doc(self, aggregator, now):
        """Test single positive document returns bullish."""
        docs = [make_doc("d1", now, "positive", 0.9)]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        assert result.document_count == 1
        assert result.bullish_ratio == 1.0
        assert result.bearish_ratio == 0.0

    def test_single_negative_doc(self, aggregator, now):
        """Test single negative document returns bearish."""
        docs = [make_doc("d1", now, "negative", 0.8)]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        assert result.document_count == 1
        assert result.bullish_ratio == 0.0
        assert result.bearish_ratio == 1.0


class TestRecencyWeighting:
    """Tests for recency-based weighting."""

    def test_recent_doc_weighted_higher(self, now):
        """Test recent documents have higher weight than old ones."""
        # Use decay_rate=0.3 (half-life ~2.3 days)
        aggregator = SentimentAggregator(decay_rate=0.3)

        # Recent positive, old negative
        docs = [
            make_doc("d1", now, "positive", 0.9),
            make_doc("d2", now - timedelta(days=5), "negative", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        # Recent positive should dominate
        assert result.bullish_ratio > result.bearish_ratio

    def test_old_doc_weighted_lower(self, now):
        """Test old documents have lower weight."""
        aggregator = SentimentAggregator(decay_rate=0.3)

        # Old positive, recent negative
        docs = [
            make_doc("d1", now - timedelta(days=5), "positive", 0.9),
            make_doc("d2", now, "negative", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        # Recent negative should dominate
        assert result.bearish_ratio > result.bullish_ratio

    def test_equal_age_equal_weight(self, now):
        """Test same-age documents have equal weight."""
        aggregator = SentimentAggregator(decay_rate=0.3)

        # Both from same time
        docs = [
            make_doc("d1", now, "positive", 0.9),
            make_doc("d2", now, "negative", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        # Should be roughly equal (accounting for confidence factor)
        assert 0.4 <= result.bullish_ratio <= 0.6


class TestWindowFiltering:
    """Tests for time window filtering."""

    def test_excludes_old_documents(self, aggregator, now):
        """Test documents outside window are excluded."""
        docs = [
            make_doc("d1", now, "positive", 0.9),
            make_doc("d2", now - timedelta(days=10), "negative", 0.9),  # Outside 7-day window
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            window_days=7,
            reference_time=now,
        )

        assert result.document_count == 1
        assert result.bullish_ratio == 1.0

    def test_includes_boundary_documents(self, aggregator, now):
        """Test documents at window boundary are included."""
        docs = [
            make_doc("d1", now - timedelta(days=7), "positive", 0.9),  # At boundary
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            window_days=7,
            reference_time=now,
        )

        assert result.document_count == 1


class TestAuthorityWeighting:
    """Tests for authority score weighting."""

    def test_high_authority_weighted_higher(self, now):
        """Test high authority documents have higher weight."""
        aggregator = SentimentAggregator(authority_weight=0.5)

        docs = [
            make_doc("d1", now, "positive", 0.9, authority=1.0),  # High authority
            make_doc("d2", now, "negative", 0.9, authority=0.0),  # Low authority
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        # High authority positive should dominate
        assert result.bullish_ratio > result.bearish_ratio

    def test_no_authority_uses_baseline(self, aggregator, now):
        """Test documents without authority use baseline weight."""
        docs = [
            make_doc("d1", now, "positive", 0.9, authority=None),
            make_doc("d2", now, "negative", 0.9, authority=None),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        # Should be roughly equal
        assert 0.4 <= result.bullish_ratio <= 0.6


class TestSentimentVelocity:
    """Tests for sentiment velocity calculation."""

    def test_increasing_sentiment_positive_velocity(self, aggregator, now):
        """Test increasing bullish ratio has positive velocity."""
        # Create documents with increasing positive sentiment over days
        docs = [
            make_doc("d1", now - timedelta(days=6), "negative", 0.9),
            make_doc("d2", now - timedelta(days=5), "negative", 0.9),
            make_doc("d3", now - timedelta(days=4), "neutral", 0.9),
            make_doc("d4", now - timedelta(days=3), "neutral", 0.9),
            make_doc("d5", now - timedelta(days=2), "positive", 0.9),
            make_doc("d6", now - timedelta(days=1), "positive", 0.9),
            make_doc("d7", now, "positive", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        assert result.sentiment_velocity is not None
        assert result.sentiment_velocity > 0

    def test_decreasing_sentiment_negative_velocity(self, aggregator, now):
        """Test decreasing bullish ratio has negative velocity."""
        # Create documents with decreasing positive sentiment over days
        docs = [
            make_doc("d1", now - timedelta(days=6), "positive", 0.9),
            make_doc("d2", now - timedelta(days=5), "positive", 0.9),
            make_doc("d3", now - timedelta(days=4), "neutral", 0.9),
            make_doc("d4", now - timedelta(days=3), "neutral", 0.9),
            make_doc("d5", now - timedelta(days=2), "negative", 0.9),
            make_doc("d6", now - timedelta(days=1), "negative", 0.9),
            make_doc("d7", now, "negative", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        assert result.sentiment_velocity is not None
        assert result.sentiment_velocity < 0

    def test_stable_sentiment_near_zero_velocity(self, aggregator, now):
        """Test stable sentiment has near-zero velocity."""
        # All positive documents
        docs = [
            make_doc(f"d{i}", now - timedelta(days=i), "positive", 0.9)
            for i in range(7)
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        # Velocity should be close to zero for stable sentiment
        if result.sentiment_velocity is not None:
            assert abs(result.sentiment_velocity) < 0.1

    def test_single_day_no_velocity(self, aggregator, now):
        """Test single day of data returns no velocity."""
        docs = [
            make_doc("d1", now, "positive", 0.9),
            make_doc("d2", now, "negative", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        # Can't compute velocity from single day
        assert result.sentiment_velocity is None


class TestExtremeSentimentDetection:
    """Tests for extreme sentiment detection."""

    def test_extreme_bullish_at_threshold(self):
        """Test extreme bullish detected at threshold."""
        aggregator = SentimentAggregator()
        result = aggregator.detect_extreme_sentiment(0.85)
        assert result == "extreme_bullish"

    def test_extreme_bullish_above_threshold(self):
        """Test extreme bullish detected above threshold."""
        aggregator = SentimentAggregator()
        result = aggregator.detect_extreme_sentiment(0.95)
        assert result == "extreme_bullish"

    def test_extreme_bearish_at_threshold(self):
        """Test extreme bearish detected at threshold."""
        aggregator = SentimentAggregator()
        result = aggregator.detect_extreme_sentiment(0.15)
        assert result == "extreme_bearish"

    def test_extreme_bearish_below_threshold(self):
        """Test extreme bearish detected below threshold."""
        aggregator = SentimentAggregator()
        result = aggregator.detect_extreme_sentiment(0.05)
        assert result == "extreme_bearish"

    def test_no_extreme_in_normal_range(self):
        """Test no extreme detection in normal range."""
        aggregator = SentimentAggregator()
        result = aggregator.detect_extreme_sentiment(0.5)
        assert result is None

    def test_extreme_in_aggregated_result(self, now):
        """Test extreme sentiment appears in aggregation result."""
        aggregator = SentimentAggregator()

        # All positive documents -> extreme bullish
        docs = [
            make_doc(f"d{i}", now - timedelta(hours=i), "positive", 0.9)
            for i in range(10)
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        assert result.extreme_sentiment == "extreme_bullish"


class TestConfigurationOverrides:
    """Tests for configuration parameter overrides."""

    def test_custom_decay_rate(self, now):
        """Test custom decay rate is applied."""
        # Very high decay rate = very short memory
        aggregator = SentimentAggregator(decay_rate=1.0)

        docs = [
            make_doc("d1", now, "positive", 0.9),
            make_doc("d2", now - timedelta(days=3), "negative", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        # With high decay, recent positive should dominate even more
        assert result.bullish_ratio > 0.9

    def test_custom_thresholds_from_config(self, now):
        """Test custom extreme thresholds from config."""
        config = SentimentConfig(
            extreme_bullish_threshold=0.7,  # Lower threshold
            extreme_bearish_threshold=0.3,  # Higher threshold
        )
        aggregator = SentimentAggregator(config=config)

        # 75% positive should trigger with lowered threshold
        docs = [
            make_doc("d1", now, "positive", 0.9),
            make_doc("d2", now, "positive", 0.9),
            make_doc("d3", now, "positive", 0.9),
            make_doc("d4", now, "negative", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        assert result.extreme_sentiment == "extreme_bullish"

    def test_get_config_summary(self):
        """Test config summary returns current settings."""
        aggregator = SentimentAggregator(
            decay_rate=0.5,
            authority_weight=0.4,
            confidence_weight=0.3,
        )

        summary = aggregator.get_config_summary()

        assert summary["decay_rate"] == 0.5
        assert summary["authority_weight"] == 0.4
        assert summary["confidence_weight"] == 0.3


class TestDailyRatios:
    """Tests for daily bullish ratio computation."""

    def test_daily_ratios_in_result(self, aggregator, now):
        """Test daily ratios are included in result."""
        docs = [
            make_doc("d1", now - timedelta(days=1), "positive", 0.9),
            make_doc("d2", now, "negative", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        assert len(result.daily_bullish_ratios) == 2

    def test_daily_ratios_sorted_by_date(self, aggregator, now):
        """Test daily ratios are sorted chronologically."""
        docs = [
            make_doc("d1", now - timedelta(days=2), "positive", 0.9),
            make_doc("d2", now, "positive", 0.9),
            make_doc("d3", now - timedelta(days=1), "positive", 0.9),
        ]

        result = aggregator.aggregate_theme_sentiment(
            theme_id="AI_chips",
            ticker=None,
            document_sentiments=docs,
            reference_time=now,
        )

        # Should be sorted by date
        dates = [dt for dt, _ in result.daily_bullish_ratios]
        assert dates == sorted(dates)
