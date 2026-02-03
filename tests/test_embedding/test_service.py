"""Tests for EmbeddingService."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from src.embedding.config import EmbeddingConfig
from src.embedding.service import EmbeddingService, ModelType


class TestEmbeddingServiceInitialization:
    """Tests for EmbeddingService initialization."""

    def test_lazy_initialization(self, embedding_config):
        """Service should not load model until first embed call."""
        service = EmbeddingService(config=embedding_config)

        assert not service.is_initialized
        assert len(service._models) == 0
        assert len(service._tokenizers) == 0

    def test_config_defaults(self):
        """Default config should use FinBERT."""
        config = EmbeddingConfig()

        assert config.model_name == "ProsusAI/finbert"
        assert config.embedding_dim == 768
        assert config.max_sequence_length == 512
        assert config.batch_size == 32

    def test_device_detection_cpu(self, embedding_config):
        """Should detect CPU when no GPU available."""
        embedding_config.device = "cpu"
        service = EmbeddingService(config=embedding_config)

        device = service._detect_device()
        assert device == torch.device("cpu")

    def test_explicit_device_selection(self, embedding_config):
        """Should use explicitly specified device."""
        embedding_config.device = "cpu"
        service = EmbeddingService(config=embedding_config)

        device = service._detect_device()
        assert device.type == "cpu"


class TestEmbeddingGeneration:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_embedding_service):
        """Should generate embedding for single text."""
        embedding = await mock_embedding_service.embed("Financial news about NVIDIA")

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, mock_embedding_service):
        """Should return zero vector for empty text."""
        embedding = await mock_embedding_service.embed("")

        assert len(embedding) == 768
        assert all(x == 0.0 for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_whitespace_only(self, mock_embedding_service):
        """Should return zero vector for whitespace-only text."""
        embedding = await mock_embedding_service.embed("   \n\t  ")

        assert len(embedding) == 768
        assert all(x == 0.0 for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_embedding_service):
        """Should generate embeddings for multiple texts."""
        texts = [
            "NVIDIA reports strong Q4 earnings",
            "AMD launches new MI300X accelerator",
            "Intel restructures datacenter business",
        ]

        embeddings = await mock_embedding_service.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 768

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, mock_embedding_service):
        """Should handle empty batch."""
        embeddings = await mock_embedding_service.embed_batch([])

        assert embeddings == []

    @pytest.mark.asyncio
    async def test_embed_batch_with_empty_texts(self, mock_embedding_service):
        """Should handle batch with empty texts."""
        texts = ["Valid text here", "", "Another valid text"]

        embeddings = await mock_embedding_service.embed_batch(texts)

        assert len(embeddings) == 3
        # Empty text should have zero embedding
        assert all(x == 0.0 for x in embeddings[1])
        # Valid texts should have non-zero embeddings
        assert any(x != 0.0 for x in embeddings[0])


class TestTextChunking:
    """Tests for long document chunking."""

    def test_short_text_no_chunking(self, mock_embedding_service):
        """Short text should not be chunked."""
        text = "Short financial news"
        chunks = mock_embedding_service._chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_chunking(self, mock_embedding_service):
        """Long text should be split into chunks."""
        # Create text longer than max_sequence_length tokens
        # Each word is roughly one token
        words = ["word"] * 600
        long_text = " ".join(words)

        chunks = mock_embedding_service._chunk_text(long_text)

        # Should have multiple chunks
        assert len(chunks) > 1

    def test_chunk_overlap(self, embedding_config, mock_tokenizer, mock_model):
        """Chunks should have overlapping tokens."""
        embedding_config.chunk_overlap = 50
        service = EmbeddingService(config=embedding_config)
        service._tokenizers = {ModelType.FINBERT: mock_tokenizer, ModelType.MINILM: mock_tokenizer}
        service._models = {ModelType.FINBERT: mock_model, ModelType.MINILM: mock_model}
        service._device = torch.device("cpu")
        service._initialized = {ModelType.FINBERT: True, ModelType.MINILM: True}

        # The mock tokenizer treats each word as a token
        words = ["word"] * 600
        long_text = " ".join(words)

        chunks = service._chunk_text(long_text)

        # With overlap, we should have controlled chunk boundaries
        assert len(chunks) >= 2


class TestMeanPooling:
    """Tests for embedding pooling."""

    def test_mean_pool_single(self, mock_embedding_service):
        """Mean pool of single embedding should return same embedding."""
        embedding = np.random.randn(768).astype(np.float32)
        pooled = mock_embedding_service._mean_pool(embedding.reshape(1, -1))

        np.testing.assert_array_almost_equal(pooled, embedding)

    def test_mean_pool_multiple(self, mock_embedding_service):
        """Mean pool should average multiple embeddings."""
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        pooled = mock_embedding_service._mean_pool(embeddings)

        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(pooled, expected)


class TestCaching:
    """Tests for embedding caching."""

    def test_content_hash_deterministic(self, mock_embedding_service):
        """Same text should produce same hash."""
        text = "Financial news content"

        hash1 = mock_embedding_service._compute_content_hash(text)
        hash2 = mock_embedding_service._compute_content_hash(text)

        assert hash1 == hash2

    def test_content_hash_different_texts(self, mock_embedding_service):
        """Different texts should produce different hashes."""
        text1 = "Financial news content"
        text2 = "Different financial content"

        hash1 = mock_embedding_service._compute_content_hash(text1)
        hash2 = mock_embedding_service._compute_content_hash(text2)

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_cache_hit(self, embedding_config, sample_embedding):
        """Should return cached embedding on cache hit."""
        embedding_config.cache_enabled = True
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(sample_embedding))

        service = EmbeddingService(
            config=embedding_config,
            redis_client=mock_redis,
        )

        cached = await service._get_cached_embedding("test text")

        assert cached == sample_embedding

    @pytest.mark.asyncio
    async def test_cache_miss(self, embedding_config):
        """Should return None on cache miss."""
        embedding_config.cache_enabled = True
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        service = EmbeddingService(
            config=embedding_config,
            redis_client=mock_redis,
        )

        cached = await service._get_cached_embedding("test text")

        assert cached is None

    @pytest.mark.asyncio
    async def test_cache_disabled(self, embedding_config):
        """Should skip cache when disabled."""
        embedding_config.cache_enabled = False
        mock_redis = AsyncMock()

        service = EmbeddingService(
            config=embedding_config,
            redis_client=mock_redis,
        )

        cached = await service._get_cached_embedding("test text")

        assert cached is None
        mock_redis.get.assert_not_called()


class TestServiceStats:
    """Tests for service statistics."""

    def test_get_stats_uninitialized(self, embedding_config):
        """Should return stats for uninitialized service."""
        service = EmbeddingService(config=embedding_config)
        stats = service.get_stats()

        assert stats["initialized"] == {ModelType.FINBERT: False, ModelType.MINILM: False}
        assert stats["finbert_model"] == "ProsusAI/finbert"
        assert stats["device"] == "not initialized"

    def test_get_stats_initialized(self, mock_embedding_service):
        """Should return stats for initialized service."""
        stats = mock_embedding_service.get_stats()

        assert stats["initialized"][ModelType.FINBERT] is True
        assert stats["finbert_dim"] == 768

    @pytest.mark.asyncio
    async def test_close_service(self, mock_embedding_service):
        """Should clean up resources on close."""
        await mock_embedding_service.close()

        assert not mock_embedding_service.is_initialized
        assert len(mock_embedding_service._models) == 0
        assert len(mock_embedding_service._tokenizers) == 0
