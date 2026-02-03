"""Tests for sentiment analysis service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import torch
import json

from src.sentiment.config import SentimentConfig
from src.sentiment.service import SentimentService, LABEL_MAPPING


class TestSentimentConfig:
    """Tests for SentimentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SentimentConfig()

        assert config.model_name == "ProsusAI/finbert"
        assert config.batch_size == 16
        assert config.use_fp16 is True
        assert config.device == "auto"
        assert config.cache_enabled is True
        assert config.cache_ttl_hours == 168
        assert config.enable_entity_sentiment is True

    def test_cache_ttl_seconds(self):
        """Test cache TTL conversion to seconds."""
        config = SentimentConfig(cache_ttl_hours=24)
        assert config.cache_ttl_seconds == 24 * 3600

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("SENTIMENT_BATCH_SIZE", "32")
        monkeypatch.setenv("SENTIMENT_CACHE_ENABLED", "false")

        config = SentimentConfig()
        assert config.batch_size == 32
        assert config.cache_enabled is False


class TestSentimentServiceInitialization:
    """Tests for SentimentService initialization."""

    def test_service_creation(self, sentiment_config):
        """Test service is created without loading model."""
        service = SentimentService(config=sentiment_config)

        assert service.is_initialized is False
        assert service._model is None
        assert service._tokenizer is None

    def test_lazy_loading(self, sentiment_config):
        """Test model is not loaded until needed."""
        service = SentimentService(config=sentiment_config)

        # Model should not be loaded yet
        assert service.is_initialized is False

    def test_device_detection_cpu(self, sentiment_config):
        """Test CPU device detection."""
        config = SentimentConfig(device="cpu")
        service = SentimentService(config=config)

        device = service.device
        assert device.type == "cpu"


class TestSentimentAnalysis:
    """Tests for sentiment analysis methods."""

    @pytest.mark.asyncio
    async def test_analyze_returns_dict(self, mock_sentiment_service, sample_positive_text):
        """Test analyze returns dictionary with expected keys."""
        result = await mock_sentiment_service.analyze(sample_positive_text)

        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert "scores" in result
        assert "model" in result
        assert "analyzed_at" in result

    @pytest.mark.asyncio
    async def test_analyze_label_values(self, mock_sentiment_service, sample_positive_text):
        """Test analyze returns valid sentiment labels."""
        result = await mock_sentiment_service.analyze(sample_positive_text)

        assert result["label"] in ["positive", "negative", "neutral"]
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_scores_sum_to_one(self, mock_sentiment_service, sample_positive_text):
        """Test that sentiment scores sum to approximately 1."""
        result = await mock_sentiment_service.analyze(sample_positive_text)

        scores = result["scores"]
        total = scores["positive"] + scores["negative"] + scores["neutral"]
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_analyze_empty_text(self, mock_sentiment_service):
        """Test analyze handles empty text."""
        result = await mock_sentiment_service.analyze("")

        assert result["label"] == "neutral"
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_analyze_whitespace_text(self, mock_sentiment_service):
        """Test analyze handles whitespace-only text."""
        result = await mock_sentiment_service.analyze("   ")

        assert result["label"] == "neutral"


class TestEntitySentiment:
    """Tests for entity-level sentiment analysis."""

    @pytest.mark.asyncio
    async def test_analyze_with_entities(
        self,
        mock_sentiment_service,
        sample_positive_text,
        sample_entities,
    ):
        """Test entity-level sentiment extraction."""
        # Extend text to include entity positions
        text = "NVIDIA stock surged on strong demand. Meanwhile AMD also gained."

        result = await mock_sentiment_service.analyze_with_entities(text, sample_entities)

        assert "entity_sentiments" in result
        assert isinstance(result["entity_sentiments"], list)

    @pytest.mark.asyncio
    async def test_entity_sentiment_fields(
        self,
        mock_sentiment_service,
        sample_positive_text,
        sample_entities,
    ):
        """Test entity sentiment contains expected fields."""
        text = "NVIDIA stock surged on strong demand."
        entities = [sample_entities[0]]  # Just NVIDIA

        result = await mock_sentiment_service.analyze_with_entities(text, entities)

        if result["entity_sentiments"]:
            entity_sent = result["entity_sentiments"][0]
            assert "entity" in entity_sent
            assert "type" in entity_sent
            assert "label" in entity_sent
            assert "confidence" in entity_sent
            assert "scores" in entity_sent
            assert "context" in entity_sent

    @pytest.mark.asyncio
    async def test_empty_entities_list(self, mock_sentiment_service, sample_positive_text):
        """Test with empty entities list returns document-level only."""
        result = await mock_sentiment_service.analyze_with_entities(sample_positive_text, [])

        # Should still have document-level sentiment
        assert "label" in result
        # entity_sentiments should be empty or absent
        assert result.get("entity_sentiments", []) == []


class TestBatchAnalysis:
    """Tests for batch sentiment analysis."""

    @pytest.mark.asyncio
    async def test_analyze_batch(self, mock_sentiment_service):
        """Test batch analysis returns list of results."""
        texts = [
            "Great earnings report from NVIDIA.",
            "AMD stock declined today.",
            "Intel announced new chips.",
        ]

        results = await mock_sentiment_service.analyze_batch(texts)

        assert len(results) == 3
        for result in results:
            assert "label" in result
            assert "confidence" in result

    @pytest.mark.asyncio
    async def test_analyze_batch_empty_list(self, mock_sentiment_service):
        """Test batch analysis with empty list."""
        results = await mock_sentiment_service.analyze_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_analyze_batch_with_empty_strings(self, mock_sentiment_service):
        """Test batch analysis handles empty strings."""
        texts = ["Good news.", "", "Bad news."]

        results = await mock_sentiment_service.analyze_batch(texts)

        assert len(results) == 3
        assert results[1]["label"] == "neutral"  # Empty string


class TestCaching:
    """Tests for Redis caching."""

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, sentiment_config):
        """Test cache key generation."""
        service = SentimentService(config=sentiment_config)

        key1 = service._make_cache_key("test text")
        key2 = service._make_cache_key("test text")
        key3 = service._make_cache_key("different text")

        assert key1 == key2
        assert key1 != key3
        assert key1.startswith("sentiment:")

    @pytest.mark.asyncio
    async def test_cache_hit(self, sentiment_config, mock_redis):
        """Test cache hit returns cached result."""
        cached_result = {
            "label": "positive",
            "confidence": 0.95,
            "scores": {"positive": 0.95, "negative": 0.02, "neutral": 0.03},
            "model": "ProsusAI/finbert",
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_result))

        service = SentimentService(
            config=sentiment_config,
            redis_client=mock_redis,
        )
        service._config.cache_enabled = True

        result = await service._get_cached_result("test text")

        assert result is not None
        assert result["label"] == "positive"

    @pytest.mark.asyncio
    async def test_cache_miss(self, sentiment_config, mock_redis):
        """Test cache miss returns None."""
        mock_redis.get = AsyncMock(return_value=None)

        service = SentimentService(
            config=sentiment_config,
            redis_client=mock_redis,
        )

        result = await service._get_cached_result("test text")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_disabled(self, sentiment_config):
        """Test caching is skipped when disabled."""
        config = SentimentConfig(cache_enabled=False)
        service = SentimentService(config=config)

        result = await service._get_cached_result("test text")

        assert result is None


class TestLabelMapping:
    """Tests for label mapping."""

    def test_label_mapping_values(self):
        """Test label mapping contains expected labels."""
        assert LABEL_MAPPING[0] == "positive"
        assert LABEL_MAPPING[1] == "negative"
        assert LABEL_MAPPING[2] == "neutral"

    def test_label_mapping_completeness(self):
        """Test label mapping covers all classes."""
        assert len(LABEL_MAPPING) == 3
        assert set(LABEL_MAPPING.values()) == {"positive", "negative", "neutral"}


class TestServiceStats:
    """Tests for service statistics."""

    def test_get_stats_before_init(self, sentiment_config):
        """Test stats before model initialization."""
        service = SentimentService(config=sentiment_config)

        stats = service.get_stats()

        assert stats["initialized"] is False
        assert stats["device"] == "not initialized"

    def test_get_stats_after_init(self, mock_sentiment_service):
        """Test stats after model initialization."""
        stats = mock_sentiment_service.get_stats()

        assert stats["initialized"] is True
        assert "model" in stats
        assert "cache_enabled" in stats


class TestTextCleaning:
    """Tests for text cleaning."""

    def test_clean_for_model_emojis(self, sentiment_config):
        """Test emoji removal from text."""
        service = SentimentService(config=sentiment_config)

        cleaned = service._clean_for_model("NVIDIA 🚀🚀🚀 to the moon!")

        # Emojis should be removed
        assert "🚀" not in cleaned
        assert "NVIDIA" in cleaned

    def test_clean_for_model_whitespace(self, sentiment_config):
        """Test whitespace normalization."""
        service = SentimentService(config=sentiment_config)

        cleaned = service._clean_for_model("NVIDIA    stock   surged")

        # Multiple spaces should be collapsed
        assert "    " not in cleaned
        assert "NVIDIA stock surged" == cleaned
