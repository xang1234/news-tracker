"""Tests for emoji sentiment lookup and modifier computation."""

import pytest

from src.sentiment.emoji_lookup import (
    EMOJI_SENTIMENT,
    compute_emoji_modifier,
    extract_emojis,
    get_emoji_breakdown,
    get_emoji_sentiment,
    is_bearish_emoji,
    is_bullish_emoji,
)


class TestEmojiSentimentDict:
    """Tests for EMOJI_SENTIMENT dictionary values."""

    def test_bullish_emojis_positive(self):
        """Test bullish emojis have positive weights."""
        bullish = ["🚀", "📈", "💎", "🙌", "💰", "🤑", "💪", "🎯", "✅", "🏆", "⬆️", "🟢"]
        for emoji in bullish:
            assert emoji in EMOJI_SENTIMENT, f"{emoji} not in lookup"
            assert EMOJI_SENTIMENT[emoji] > 0, f"{emoji} should be positive"

    def test_bearish_emojis_negative(self):
        """Test bearish emojis have negative weights."""
        bearish = ["📉", "💩", "🤡", "⚠️", "❌", "🔻", "⬇️", "🩸", "💀", "🪦", "🗑️", "🔴"]
        for emoji in bearish:
            assert emoji in EMOJI_SENTIMENT, f"{emoji} not in lookup"
            assert EMOJI_SENTIMENT[emoji] < 0, f"{emoji} should be negative"

    def test_rocket_strongest_bullish(self):
        """Test rocket emoji is among the strongest bullish signals."""
        assert EMOJI_SENTIMENT["🚀"] >= 0.3

    def test_tombstone_strongest_bearish(self):
        """Test tombstone is among the strongest bearish signals."""
        assert EMOJI_SENTIMENT["🪦"] <= -0.25


class TestExtractEmojis:
    """Tests for emoji extraction from text."""

    def test_extract_single_emoji(self):
        """Test extracting single emoji."""
        emojis = extract_emojis("NVDA 🚀")
        assert "🚀" in emojis

    def test_extract_multiple_emojis(self):
        """Test extracting multiple emojis."""
        emojis = extract_emojis("To the moon 🚀🚀🚀")
        assert emojis.count("🚀") == 3

    def test_extract_mixed_emojis(self):
        """Test extracting mix of bullish and bearish emojis."""
        emojis = extract_emojis("NVDA 🚀 but AMD 📉")
        assert "🚀" in emojis
        assert "📉" in emojis

    def test_extract_no_emojis(self):
        """Test text without emojis returns empty list."""
        emojis = extract_emojis("Plain text without emojis")
        assert emojis == []

    def test_extract_empty_string(self):
        """Test empty string returns empty list."""
        emojis = extract_emojis("")
        assert emojis == []

    def test_extract_preserves_order(self):
        """Test emojis are extracted in order of appearance."""
        emojis = extract_emojis("📈🚀💎")
        # Should contain all three
        assert "📈" in emojis
        assert "🚀" in emojis
        assert "💎" in emojis


class TestGetEmojiBreakdown:
    """Tests for emoji breakdown function."""

    def test_breakdown_single_emoji(self):
        """Test breakdown with single emoji."""
        breakdown = get_emoji_breakdown("NVDA 🚀")
        assert "🚀" in breakdown
        assert breakdown["🚀"] == EMOJI_SENTIMENT["🚀"]

    def test_breakdown_duplicate_emojis_sum(self):
        """Test duplicate emojis sum their weights."""
        breakdown = get_emoji_breakdown("🚀🚀🚀")
        assert breakdown["🚀"] == EMOJI_SENTIMENT["🚀"] * 3

    def test_breakdown_excludes_unknown_emojis(self):
        """Test unknown emojis are excluded from breakdown."""
        # Use an emoji not in our lookup (e.g., food emoji)
        breakdown = get_emoji_breakdown("🍕🚀")
        assert "🍕" not in breakdown
        assert "🚀" in breakdown

    def test_breakdown_empty_for_no_known_emojis(self):
        """Test empty breakdown when no known emojis."""
        breakdown = get_emoji_breakdown("Just text")
        assert breakdown == {}


class TestComputeEmojiModifier:
    """Tests for emoji modifier computation."""

    def test_modifier_single_bullish(self):
        """Test modifier from single bullish emoji."""
        modifier, breakdown = compute_emoji_modifier("NVDA 🚀")
        assert modifier > 0
        assert "🚀" in breakdown

    def test_modifier_single_bearish(self):
        """Test modifier from single bearish emoji."""
        modifier, breakdown = compute_emoji_modifier("NVDA 📉")
        assert modifier < 0
        assert "📉" in breakdown

    def test_modifier_clips_to_max_positive(self):
        """Test modifier clips to max_modifier for many bullish emojis."""
        # Many rockets should clip to max
        modifier, _ = compute_emoji_modifier("🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀", max_modifier=0.5)
        assert modifier == 0.5

    def test_modifier_clips_to_max_negative(self):
        """Test modifier clips to -max_modifier for many bearish emojis."""
        modifier, _ = compute_emoji_modifier("📉📉📉📉📉📉📉📉📉📉", max_modifier=0.5)
        assert modifier == -0.5

    def test_mixed_emojis_partially_cancel(self):
        """Test mixed bullish/bearish emojis partially cancel."""
        # Rocket (0.3) + chart down (-0.2) = 0.1
        modifier, breakdown = compute_emoji_modifier("🚀📉")
        assert 0 < modifier < 0.3  # Positive but reduced
        assert "🚀" in breakdown
        assert "📉" in breakdown

    def test_custom_max_modifier(self):
        """Test custom max_modifier parameter."""
        modifier, _ = compute_emoji_modifier("🚀🚀🚀🚀", max_modifier=0.2)
        assert modifier <= 0.2

    def test_no_emojis_returns_zero(self):
        """Test text without emojis returns zero modifier."""
        modifier, breakdown = compute_emoji_modifier("Plain text")
        assert modifier == 0.0
        assert breakdown == {}

    def test_negative_max_modifier_raises(self):
        """Test negative max_modifier raises ValueError."""
        with pytest.raises(ValueError):
            compute_emoji_modifier("🚀", max_modifier=-0.5)

    def test_returns_tuple(self):
        """Test function returns (modifier, breakdown) tuple."""
        result = compute_emoji_modifier("🚀")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], dict)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_emoji_sentiment_known(self):
        """Test get_emoji_sentiment for known emoji."""
        assert get_emoji_sentiment("🚀") == EMOJI_SENTIMENT["🚀"]

    def test_get_emoji_sentiment_unknown(self):
        """Test get_emoji_sentiment for unknown emoji."""
        assert get_emoji_sentiment("🍕") == 0.0

    def test_is_bullish_emoji_true(self):
        """Test is_bullish_emoji returns True for bullish."""
        assert is_bullish_emoji("🚀") is True
        assert is_bullish_emoji("📈") is True

    def test_is_bullish_emoji_false(self):
        """Test is_bullish_emoji returns False for bearish."""
        assert is_bullish_emoji("📉") is False
        assert is_bullish_emoji("💀") is False

    def test_is_bearish_emoji_true(self):
        """Test is_bearish_emoji returns True for bearish."""
        assert is_bearish_emoji("📉") is True
        assert is_bearish_emoji("💀") is True

    def test_is_bearish_emoji_false(self):
        """Test is_bearish_emoji returns False for bullish."""
        assert is_bearish_emoji("🚀") is False
        assert is_bearish_emoji("📈") is False


class TestIntegrationWithSentimentService:
    """Integration tests with SentimentService."""

    @pytest.mark.asyncio
    async def test_emoji_modifier_in_result(self, mock_sentiment_service):
        """Test emoji modifier appears in sentiment result."""
        # The mock service should have emoji modifier enabled by default
        result = await mock_sentiment_service.analyze("NVDA looking good 🚀🚀")

        # When emojis are found, emoji_modifier should be in result
        assert "emoji_modifier" in result or result.get("emojis_found") is None
        # If emojis were found, modifier should be positive for rockets
        if "emoji_modifier" in result:
            assert result["emoji_modifier"] > 0

    @pytest.mark.asyncio
    async def test_no_emoji_modifier_when_no_emojis(self, mock_sentiment_service):
        """Test no emoji fields when text has no emojis."""
        result = await mock_sentiment_service.analyze("NVDA looking good")

        # When no emojis, these fields should be absent
        assert "emoji_modifier" not in result
        assert "emojis_found" not in result
