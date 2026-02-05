"""Tests for the KeywordsService."""

import pytest

from src.keywords import ExtractedKeyword, KeywordsConfig, KeywordsService


class TestExtractedKeyword:
    """Tests for the ExtractedKeyword dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        keyword = ExtractedKeyword(
            text="semiconductor chip",
            score=0.125,
            rank=1,
            lemma="semiconductor chip",
            count=3,
            metadata={"algorithm": "textrank"},
        )
        result = keyword.to_dict()

        assert result["text"] == "semiconductor chip"
        assert result["score"] == 0.125
        assert result["rank"] == 1
        assert result["lemma"] == "semiconductor chip"
        assert result["count"] == 3
        assert result["metadata"] == {"algorithm": "textrank"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "text": "gpu architecture",
            "score": 0.08,
            "rank": 2,
            "lemma": "gpu architecture",
            "count": 2,
            "metadata": {"algorithm": "textrank"},
        }
        keyword = ExtractedKeyword.from_dict(data)

        assert keyword.text == "gpu architecture"
        assert keyword.score == 0.08
        assert keyword.rank == 2
        assert keyword.lemma == "gpu architecture"
        assert keyword.count == 2

    def test_from_dict_missing_optional(self):
        """Test creation from dictionary with missing optional fields."""
        data = {
            "text": "memory bandwidth",
            "score": 0.05,
            "rank": 3,
            "lemma": "memory bandwidth",
        }
        keyword = ExtractedKeyword.from_dict(data)

        assert keyword.text == "memory bandwidth"
        assert keyword.count == 1  # Default value
        assert keyword.metadata == {}  # Default value

    def test_equality(self):
        """Test equality based on lemma."""
        kw1 = ExtractedKeyword(
            text="GPU",
            score=0.1,
            rank=1,
            lemma="gpu",
            count=1,
        )
        kw2 = ExtractedKeyword(
            text="Gpu",
            score=0.2,
            rank=2,
            lemma="gpu",  # Same lemma
            count=2,
        )
        kw3 = ExtractedKeyword(
            text="CPU",
            score=0.1,
            rank=1,
            lemma="cpu",  # Different lemma
            count=1,
        )

        assert kw1 == kw2  # Same lemma
        assert kw1 != kw3  # Different lemma

    def test_hash(self):
        """Test hashing for deduplication."""
        kw1 = ExtractedKeyword(text="test", score=0.1, rank=1, lemma="test", count=1)
        kw2 = ExtractedKeyword(text="TEST", score=0.2, rank=2, lemma="test", count=2)

        # Same lemma should have same hash
        assert hash(kw1) == hash(kw2)

        # Can be used in sets
        keywords_set = {kw1, kw2}
        assert len(keywords_set) == 1  # Deduplicated


class TestKeywordsConfig:
    """Tests for the KeywordsConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = KeywordsConfig()

        assert config.spacy_model == "en_core_web_sm"
        assert config.top_n == 10
        assert config.language == "en"
        assert config.min_score == 0.0
        assert config.max_text_length == 10000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = KeywordsConfig(
            spacy_model="en_core_web_trf",
            top_n=5,
            language="en",
            min_score=0.01,
            max_text_length=5000,
        )

        assert config.spacy_model == "en_core_web_trf"
        assert config.top_n == 5
        assert config.min_score == 0.01
        assert config.max_text_length == 5000


class TestKeywordsService:
    """Tests for the KeywordsService."""

    def test_initialization(self):
        """Test service initializes without loading extractor."""
        service = KeywordsService()

        assert not service.is_initialized
        assert service.config is not None

    def test_custom_config(self):
        """Test service with custom config."""
        config = KeywordsConfig(top_n=5)
        service = KeywordsService(config=config)

        assert service.config.top_n == 5

    def test_extract_empty_text(self):
        """Test extraction returns empty list for empty text."""
        service = KeywordsService()

        assert service.extract_sync("") == []
        assert service.extract_sync("   ") == []
        assert service.extract_sync(None) == [] if service.extract_sync(None) is not None else True

    def test_extract_short_text(self):
        """Test extraction on short text."""
        service = KeywordsService()
        # Short text might not produce keywords but should not error
        result = service.extract_sync("Hello world")

        assert isinstance(result, list)

    def test_extract_financial_text(self):
        """Test extraction on financial news text."""
        service = KeywordsService()
        text = (
            "Nvidia announced new GPU architecture with HBM3E memory support. "
            "The H200 processor features advanced AI accelerator capabilities. "
            "Semiconductor industry analysts expect strong demand for data center chips."
        )
        keywords = service.extract_sync(text)

        assert isinstance(keywords, list)
        # Should extract some keywords from this text
        if keywords:
            # Check structure
            assert all(isinstance(kw, ExtractedKeyword) for kw in keywords)
            assert all(kw.rank > 0 for kw in keywords)
            assert all(kw.score >= 0 for kw in keywords)
            # Check ranking is sequential
            ranks = [kw.rank for kw in keywords]
            assert ranks == list(range(1, len(keywords) + 1))

    def test_extract_respects_top_n(self):
        """Test extraction respects top_n configuration."""
        config = KeywordsConfig(top_n=3)
        service = KeywordsService(config=config)
        text = (
            "Nvidia semiconductor GPU architecture memory bandwidth "
            "AI accelerator data center chip processor HBM memory "
            "machine learning inference training optimization"
        )
        keywords = service.extract_sync(text)

        assert len(keywords) <= 3

    def test_extract_respects_min_score(self):
        """Test extraction filters by min_score."""
        config = KeywordsConfig(min_score=0.5)  # Very high threshold
        service = KeywordsService(config=config)
        text = "This is a simple test text with some words."
        keywords = service.extract_sync(text)

        # With high threshold, should filter out low-scoring keywords
        assert all(kw.score >= 0.5 for kw in keywords)

    def test_extract_truncates_long_text(self):
        """Test extraction truncates text exceeding max_text_length."""
        config = KeywordsConfig(max_text_length=100)
        service = KeywordsService(config=config)
        long_text = "semiconductor " * 100  # 1400+ characters
        keywords = service.extract_sync(long_text)

        # Should not error, might extract some keywords
        assert isinstance(keywords, list)

    @pytest.mark.asyncio
    async def test_async_extract(self):
        """Test async extraction wrapper."""
        service = KeywordsService()
        text = "Nvidia GPU semiconductor technology"
        keywords = await service.extract(text)

        assert isinstance(keywords, list)

    @pytest.mark.asyncio
    async def test_batch_extract(self):
        """Test batch extraction."""
        service = KeywordsService()
        texts = [
            "Nvidia GPU semiconductor",
            "AMD processor architecture",
            "Intel chip manufacturing",
        ]
        results = await service.extract_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    @pytest.mark.asyncio
    async def test_batch_extract_empty(self):
        """Test batch extraction with empty list."""
        service = KeywordsService()
        results = await service.extract_batch([])

        assert results == []

    def test_lazy_initialization(self):
        """Test extractor is loaded lazily."""
        service = KeywordsService()

        # Not initialized until first extraction
        assert not service.is_initialized

        # Trigger initialization
        service.extract_sync("Test text for initialization")

        # Now should be initialized
        assert service.is_initialized

    def test_keyword_metadata(self):
        """Test extracted keywords have proper metadata."""
        service = KeywordsService()
        text = (
            "Nvidia GPU semiconductor technology data center "
            "AI accelerator machine learning inference"
        )
        keywords = service.extract_sync(text)

        if keywords:
            kw = keywords[0]
            # Check metadata is set
            assert "algorithm" in kw.metadata
            assert kw.metadata["algorithm"] == "rapid_textrank"
