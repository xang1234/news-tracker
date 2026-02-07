"""Tests for NER service."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.ner.config import NERConfig
from src.ner.schemas import FinancialEntity
from src.ner.service import NERService


class TestNERServiceInitialization:
    """Tests for NERService initialization."""

    def test_lazy_initialization(self, ner_config):
        """Model should not be loaded until first extract call."""
        import sys
        mock_spacy = MagicMock()
        with patch.dict(sys.modules, {"spacy": mock_spacy}):
            service = NERService(config=ner_config)

            assert not service.is_initialized
            mock_spacy.load.assert_not_called()

    def test_initialization_on_extract(self, ner_config):
        """Model should load on first extract call."""
        import sys
        mock_spacy = MagicMock()
        mock_nlp = MagicMock()
        mock_nlp.meta = {"name": "test_model"}
        mock_nlp.return_value = MagicMock(ents=[])
        mock_spacy.load.return_value = mock_nlp

        with patch.dict(sys.modules, {"spacy": mock_spacy}):
            service = NERService(config=ner_config)
            service.extract_sync("Test text")

            assert service.is_initialized
            mock_spacy.load.assert_called_once()

    def test_fallback_model(self, ner_config):
        """Should try fallback model if primary fails."""
        import sys
        mock_spacy = MagicMock()
        mock_nlp = MagicMock()
        mock_nlp.meta = {"name": "fallback_model"}
        mock_nlp.return_value = MagicMock(ents=[])

        # First call raises OSError, second succeeds
        mock_spacy.load.side_effect = [OSError("Model not found"), mock_nlp]

        with patch.dict(sys.modules, {"spacy": mock_spacy}):
            service = NERService(config=ner_config)
            service.extract_sync("Test text")

            assert mock_spacy.load.call_count == 2


class TestCashtagExtraction:
    """Tests for cashtag ($TICKER) extraction."""

    def test_extract_cashtag(self, mock_ner_service):
        """Should extract $TICKER cashtags."""
        entities = mock_ner_service.extract_sync("Looking at $NVDA today")

        ticker_entities = [e for e in entities if e.type == "TICKER"]
        assert len(ticker_entities) >= 1
        assert any(e.normalized == "NVDA" for e in ticker_entities)

    def test_extract_multiple_cashtags(self, mock_ner_service):
        """Should extract multiple cashtags."""
        entities = mock_ner_service.extract_sync("$NVDA and $AMD are up, $INTC is flat")

        tickers = mock_ner_service.extract_tickers("$NVDA and $AMD are up, $INTC is flat")
        assert "NVDA" in tickers
        assert "AMD" in tickers
        assert "INTC" in tickers

    def test_filter_non_semiconductor_tickers(self, mock_ner_service):
        """Should not extract tickers not in semiconductor list."""
        entities = mock_ner_service.extract_sync("$AAPL is a great stock")

        # AAPL is not in SEMICONDUCTOR_TICKERS
        ticker_entities = [e for e in entities if e.type == "TICKER"]
        assert not any(e.normalized == "AAPL" for e in ticker_entities)


class TestCompanyExtraction:
    """Tests for company name extraction."""

    def test_extract_company_from_spacy(self, mock_ner_service):
        """Should extract company names from spaCy NER."""
        text = "Nvidia announced the H100 GPU will power the next generation of AI workloads."
        entities = mock_ner_service.extract_sync(text)

        company_entities = [e for e in entities if e.type == "COMPANY"]
        assert len(company_entities) >= 1
        assert any("NVIDIA" in e.normalized.upper() for e in company_entities)

    def test_company_with_ticker_metadata(self, mock_ner_service):
        """Company entities should include ticker in metadata."""
        text = "Nvidia announced the H100 GPU will power the next generation of AI workloads."
        entities = mock_ner_service.extract_sync(text)

        nvidia_entities = [
            e for e in entities
            if e.type == "COMPANY" and "NVIDIA" in e.normalized.upper()
        ]
        if nvidia_entities:
            assert nvidia_entities[0].metadata.get("ticker") == "NVDA"


class TestFuzzyMatching:
    """Tests for fuzzy company name matching."""

    def test_fuzzy_match_company_variation(self, ner_config):
        """Should match company name variations via fuzzy matching."""
        mock_nlp = MagicMock()
        mock_nlp.meta = {"name": "test_model"}
        mock_nlp.return_value = MagicMock(ents=[])

        service = NERService(config=ner_config)
        service._initialized = True
        service._nlp = mock_nlp
        service._coref_model = None

        # The fuzzy matcher should find "nvidia" variation
        entities = service._fuzzy_match_companies("NVIDIA is great", [])

        nvidia_entities = [
            e for e in entities if "NVIDIA" in e.normalized.upper()
        ]
        assert len(nvidia_entities) >= 1


class TestEntityDeduplication:
    """Tests for entity deduplication."""

    def test_deduplicate_overlapping(self, overlapping_entities):
        """Should keep longer entity when spans overlap."""
        service = NERService()
        result = service._deduplicate_entities(overlapping_entities)

        # Should keep "Taiwan Semiconductor Manufacturing" (longest)
        assert len(result) == 1
        assert "MANUFACTURING" in result[0].normalized

    def test_deduplicate_by_normalized(self):
        """Should deduplicate entities with same normalized form."""
        entities = [
            FinancialEntity(
                text="Nvidia",
                type="COMPANY",
                normalized="NVIDIA",
                start=0,
                end=6,
                confidence=0.9,
            ),
            FinancialEntity(
                text="nvidia",
                type="COMPANY",
                normalized="NVIDIA",
                start=50,
                end=56,
                confidence=0.85,
            ),
        ]

        service = NERService()
        result = service._deduplicate_entities(entities)

        assert len(result) == 1


class TestTechnologyExtraction:
    """Tests for technology term extraction."""

    def test_extract_technology(self, mock_ner_service):
        """Should extract technology terms."""
        text = "TSMC's 3nm process with CoWoS packaging enables HBM3E memory integration."
        entities = mock_ner_service.extract_sync(text)

        tech_entities = [e for e in entities if e.type == "TECHNOLOGY"]
        normalized_techs = [e.normalized.upper() for e in tech_entities]

        # At least some of these should be found
        found = [t for t in ["3NM", "COWOS", "HBM3E"] if t in normalized_techs]
        assert len(found) >= 1


class TestMetricExtraction:
    """Tests for metric extraction."""

    def test_extract_metrics(self, mock_ner_service):
        """Should extract financial metrics."""
        text = "AMD reported Q4 revenue of $5.6 billion, up 10% YoY with gross margin of 51%."
        entities = mock_ner_service.extract_sync(text)

        metric_entities = [e for e in entities if e.type == "METRIC"]
        # Should find at least the percentage values
        assert len(metric_entities) >= 1


class TestBatchExtraction:
    """Tests for batch extraction."""

    @pytest.mark.asyncio
    async def test_extract_batch(self, mock_ner_service, sample_financial_texts):
        """Should process multiple texts in batch."""
        results = await mock_ner_service.extract_batch(sample_financial_texts[:3])

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    @pytest.mark.asyncio
    async def test_extract_batch_empty(self, mock_ner_service):
        """Should handle empty batch."""
        results = await mock_ner_service.extract_batch([])

        assert results == []


class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_extract_tickers(self, mock_ner_service):
        """extract_tickers should return just ticker symbols."""
        text = "$NVDA and $AMD are leading. Nvidia is bullish."
        tickers = mock_ner_service.extract_tickers(text)

        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)
        assert "NVDA" in tickers

    def test_link_entities_to_theme(self, mock_ner_service):
        """Should calculate theme relevance scores."""
        entities = [
            FinancialEntity(
                text="Nvidia",
                type="COMPANY",
                normalized="NVIDIA",
                start=0,
                end=6,
                confidence=0.95,
                metadata={"ticker": "NVDA"},
            ),
            FinancialEntity(
                text="HBM3E",
                type="TECHNOLOGY",
                normalized="HBM3E",
                start=20,
                end=25,
                confidence=0.9,
            ),
        ]

        theme_keywords = ["nvidia", "gpu", "ai", "semiconductor"]
        scores = mock_ner_service.link_entities_to_theme(entities, theme_keywords)

        assert "NVIDIA" in scores
        assert "HBM3E" in scores
        assert scores["NVIDIA"] > 0  # Should have some relevance


class TestSemanticThemeLinking:
    """Tests for embedding-based semantic entity-theme linking."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service for testing."""
        mock = MagicMock()

        # Generate deterministic "embeddings" based on text content
        async def mock_embed_minilm(text: str) -> list[float]:
            # Create pseudo-embeddings where similar concepts have similar vectors
            vec = np.zeros(384)

            # Seed based on text for reproducibility
            text_lower = text.lower()

            # Semiconductor/GPU-related terms cluster together
            if any(kw in text_lower for kw in ["nvidia", "gpu", "graphics", "cuda"]):
                vec[:100] = 0.8
                vec[100:200] = 0.3
            elif any(kw in text_lower for kw in ["amd", "radeon", "ryzen"]):
                vec[:100] = 0.7
                vec[100:200] = 0.4
            elif any(kw in text_lower for kw in ["memory", "hbm", "dram", "ram"]):
                vec[150:250] = 0.8
                vec[50:150] = 0.2
            elif any(kw in text_lower for kw in ["ai", "deep learning", "accelerator"]):
                vec[:100] = 0.6
                vec[200:300] = 0.7
            elif any(kw in text_lower for kw in ["apple", "fruit", "food"]):
                # Clearly different domain
                vec[300:384] = 0.9
            else:
                # Generic text
                vec[50:150] = 0.3

            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            return vec.tolist()

        mock.embed_minilm = AsyncMock(side_effect=mock_embed_minilm)
        return mock

    @pytest.fixture
    def ner_service_with_embeddings(self, ner_config, mock_embedding_service):
        """Create NER service with mock embedding service."""
        service = NERService(
            config=ner_config,
            embedding_service=mock_embedding_service,
        )
        service._initialized = True
        service._nlp = MagicMock()
        service._nlp.meta = {"name": "test_model"}
        service._coref_model = None
        return service

    def test_has_embedding_service_property(self, mock_ner_service, mock_embedding_service):
        """Service should report embedding service availability."""
        # Without embedding service
        assert mock_ner_service.has_embedding_service is False

        # With embedding service
        service_with_emb = NERService(embedding_service=mock_embedding_service)
        assert service_with_emb.has_embedding_service is True

    @pytest.mark.asyncio
    async def test_semantic_linking_basic(self, ner_service_with_embeddings):
        """Should compute semantic similarity scores."""
        entities = [
            FinancialEntity(
                text="Nvidia",
                type="COMPANY",
                normalized="NVIDIA",
                start=0,
                end=6,
                confidence=0.95,
                metadata={"ticker": "NVDA"},
            ),
        ]

        theme_keywords = ["GPU", "graphics", "gaming"]
        scores = await ner_service_with_embeddings.link_entities_to_theme_semantic(
            entities, theme_keywords
        )

        assert "NVIDIA" in scores
        assert scores["NVIDIA"] > 0.5  # Should have high semantic similarity

    @pytest.mark.asyncio
    async def test_semantic_linking_domain_bonus(self, ner_service_with_embeddings):
        """Semiconductor tickers and TECHNOLOGY types should get bonus."""
        entities = [
            FinancialEntity(
                text="HBM3E",
                type="TECHNOLOGY",
                normalized="HBM3E",
                start=0,
                end=5,
                confidence=0.9,
            ),
            FinancialEntity(
                text="AMD",
                type="COMPANY",
                normalized="AMD",
                start=10,
                end=13,
                confidence=0.9,
                metadata={"ticker": "AMD"},
            ),
        ]

        theme_keywords = ["memory", "bandwidth"]
        scores = await ner_service_with_embeddings.link_entities_to_theme_semantic(
            entities, theme_keywords
        )

        # HBM3E is TECHNOLOGY type (+0.1 bonus)
        assert "HBM3E" in scores

        # AMD has semiconductor ticker (+0.2 bonus)
        assert "AMD" in scores
        assert scores["AMD"] >= 0.2  # At minimum the domain bonus

    @pytest.mark.asyncio
    async def test_semantic_linking_low_similarity_filtered(
        self, ner_service_with_embeddings
    ):
        """Entities below threshold should get zero score."""
        # Configure threshold
        ner_service_with_embeddings.config.semantic_similarity_threshold = 0.7

        entities = [
            FinancialEntity(
                text="Apple",
                type="COMPANY",
                normalized="APPLE",
                start=0,
                end=5,
                confidence=0.9,
            ),
        ]

        # Theme is completely unrelated to our mock Apple embedding
        theme_keywords = ["GPU", "graphics", "cuda"]
        scores = await ner_service_with_embeddings.link_entities_to_theme_semantic(
            entities, theme_keywords
        )

        # Apple (fruit) should have low similarity to GPU theme
        # and be filtered below threshold
        assert "APPLE" in scores
        # Score should be low since similarity is below threshold
        assert scores["APPLE"] < 0.3

    @pytest.mark.asyncio
    async def test_semantic_linking_empty_entities(self, ner_service_with_embeddings):
        """Should return empty dict for empty entity list."""
        scores = await ner_service_with_embeddings.link_entities_to_theme_semantic(
            [], ["ai", "deep learning"]
        )
        assert scores == {}

    @pytest.mark.asyncio
    async def test_semantic_linking_fallback_without_embedding_service(
        self, mock_ner_service
    ):
        """Should fall back to substring matching without embedding service."""
        entities = [
            FinancialEntity(
                text="Nvidia",
                type="COMPANY",
                normalized="NVIDIA",
                start=0,
                end=6,
                confidence=0.95,
                metadata={"ticker": "NVDA"},
            ),
        ]

        theme_keywords = ["nvidia", "gpu"]  # "nvidia" appears in theme
        scores = await mock_ner_service.link_entities_to_theme_semantic(
            entities, theme_keywords
        )

        # Should use substring matching fallback
        assert "NVIDIA" in scores
        assert scores["NVIDIA"] >= 0.5  # Substring match gives +0.5

    @pytest.mark.asyncio
    async def test_semantic_linking_handles_embedding_failure(
        self, ner_config, mock_embedding_service
    ):
        """Should gracefully handle embedding failures."""
        # Make embedding fail
        mock_embedding_service.embed_minilm = AsyncMock(
            side_effect=Exception("Embedding failed")
        )

        service = NERService(
            config=ner_config,
            embedding_service=mock_embedding_service,
        )
        service._initialized = True
        service._nlp = MagicMock()
        service._nlp.meta = {"name": "test"}
        service._coref_model = None

        entities = [
            FinancialEntity(
                text="Nvidia",
                type="COMPANY",
                normalized="NVIDIA",
                start=0,
                end=6,
                confidence=0.95,
                metadata={"ticker": "NVDA"},
            ),
        ]

        # Should not raise, should fall back to substring matching
        scores = await service.link_entities_to_theme_semantic(
            entities, ["nvidia", "gpu"]
        )

        # Falls back to substring matching
        assert "NVIDIA" in scores

    @pytest.mark.asyncio
    async def test_cosine_similarity_calculation(self, ner_service_with_embeddings):
        """Cosine similarity should be computed correctly."""
        # Test identical vectors
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([1.0, 0.0, 0.0])
        assert ner_service_with_embeddings._cosine_similarity(vec_a, vec_b) == 1.0

        # Test orthogonal vectors
        vec_c = np.array([0.0, 1.0, 0.0])
        assert ner_service_with_embeddings._cosine_similarity(vec_a, vec_c) == 0.0

        # Test zero vector
        vec_zero = np.array([0.0, 0.0, 0.0])
        assert ner_service_with_embeddings._cosine_similarity(vec_a, vec_zero) == 0.0

    @pytest.mark.asyncio
    async def test_semantic_linking_disambiguation(self, ner_service_with_embeddings):
        """Should disambiguate entities based on theme context."""
        # Two entities with same text but different contexts
        entities = [
            FinancialEntity(
                text="Nvidia GPU",
                type="PRODUCT",
                normalized="NVIDIA GPU",
                start=0,
                end=10,
                confidence=0.9,
            ),
        ]

        # GPU-related theme
        gpu_theme = ["graphics processing unit", "CUDA cores", "gaming"]
        scores_gpu = await ner_service_with_embeddings.link_entities_to_theme_semantic(
            entities, gpu_theme
        )

        # AI-related theme (also relevant to Nvidia)
        ai_theme = ["deep learning", "AI accelerator", "neural network"]
        scores_ai = await ner_service_with_embeddings.link_entities_to_theme_semantic(
            entities, ai_theme
        )

        # Both should have positive scores since Nvidia is relevant to both
        assert scores_gpu["NVIDIA GPU"] > 0
        assert scores_ai["NVIDIA GPU"] > 0


class TestCoreferenceResolution:
    """Tests for pre-NER coreference resolution via fastcoref."""

    def test_coref_bypassed_for_short_text(self, mock_ner_service_with_coref):
        """Short text (<coref_min_length) should skip coreference resolution."""
        short_text = "Nvidia is great. It leads AI."
        assert len(short_text) < mock_ner_service_with_coref.config.coref_min_length

        entities = mock_ner_service_with_coref.extract_sync(short_text)

        # Should find Nvidia but NOT resolve "It" (coref skipped)
        company_entities = [e for e in entities if e.type == "COMPANY"]
        assert any("NVIDIA" in e.normalized.upper() for e in company_entities)

    def test_coref_runs_for_long_text(self, mock_ner_service_with_coref):
        """Long text (>=coref_min_length) should resolve coreferences before NER."""
        # This text is below 500 chars, so set min_length low for testing
        mock_ner_service_with_coref.config.coref_min_length = 10

        text = (
            "Samsung announced new HBM capacity. "
            "The Korean chipmaker expects strong demand."
        )

        entities = mock_ner_service_with_coref.extract_sync(text)

        # After coref resolution, "The Korean chipmaker" becomes "Samsung",
        # so NER should find Samsung in the resolved text
        company_entities = [e for e in entities if e.type == "COMPANY"]
        assert any("SAMSUNG" in e.normalized.upper() for e in company_entities)

    def test_coref_min_length_boundary(self, mock_ner_service_with_coref):
        """Text exactly at coref_min_length should trigger resolution."""
        text = (
            "Samsung announced new HBM capacity. "
            "The Korean chipmaker expects strong demand."
        )
        # Set min_length to exactly the text length
        mock_ner_service_with_coref.config.coref_min_length = len(text)

        entities = mock_ner_service_with_coref.extract_sync(text)

        # Should still run coref (>= threshold)
        company_entities = [e for e in entities if e.type == "COMPANY"]
        assert len(company_entities) >= 1

    def test_resolve_text_returns_resolved(self, mock_ner_service_with_coref):
        """_resolve_text should replace pronouns with antecedents."""
        text = (
            "Nvidia unveiled the Blackwell architecture. "
            "The company claims it delivers 4x the performance of Hopper."
        )

        resolved = mock_ner_service_with_coref._resolve_text(text)

        assert "Nvidia" in resolved
        # "The company" should be replaced
        assert "The company claims" not in resolved

    def test_resolve_text_graceful_on_failure(self, ner_config_with_coref):
        """_resolve_text should return original text if coref model fails."""
        service = NERService(config=ner_config_with_coref)
        service._initialized = True
        service._nlp = MagicMock()
        service._nlp.meta = {"name": "test"}

        # Create a coref model that raises
        mock_coref = MagicMock()
        mock_coref.predict.side_effect = RuntimeError("Model error")
        service._coref_model = mock_coref

        text = "Some text that will fail coref."
        resolved = service._resolve_text(text)

        assert resolved == text  # Falls back to original

    def test_resolve_text_without_coref_model(self, mock_ner_service):
        """_resolve_text should return original text when no coref model."""
        text = "Some text without coref."
        resolved = mock_ner_service._resolve_text(text)
        assert resolved == text

    def test_coref_disabled_in_config(self, ner_config):
        """When enable_coreference=False, no coref model should be loaded."""
        ner_config.enable_coreference = False

        service = NERService(config=ner_config)
        assert service._coref_model is None

    @pytest.mark.asyncio
    async def test_batch_coref_respects_min_length(self, mock_ner_service_with_coref):
        """Batch extraction should apply coref only to texts above min_length."""
        mock_ner_service_with_coref.config.coref_min_length = 50

        short_text = "Nvidia is great. It leads AI."  # < 50 chars
        long_text = (
            "Samsung announced new HBM capacity. "
            "The Korean chipmaker expects strong demand."
        )  # > 50 chars

        results = await mock_ner_service_with_coref.extract_batch(
            [short_text, long_text]
        )

        assert len(results) == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self, mock_ner_service):
        """Should handle empty text."""
        entities = mock_ner_service.extract_sync("")
        assert entities == []

    def test_whitespace_text(self, mock_ner_service):
        """Should handle whitespace-only text."""
        entities = mock_ner_service.extract_sync("   \n\t  ")
        assert entities == []

    def test_long_text_truncation(self, ner_config):
        """Should truncate very long texts."""
        ner_config.max_text_length = 100

        mock_nlp = MagicMock()
        mock_nlp.meta = {"name": "test_model"}
        mock_nlp.return_value = MagicMock(ents=[])

        service = NERService(config=ner_config)
        service._initialized = True
        service._nlp = mock_nlp
        service._coref_model = None

        long_text = "Nvidia " * 100  # ~700 chars
        service.extract_sync(long_text)

        # Check that nlp was called with truncated text
        call_args = mock_nlp.call_args[0][0]
        assert len(call_args) <= 100

    def test_confidence_filtering(self, ner_config):
        """Should filter entities below confidence threshold."""
        ner_config.confidence_threshold = 0.8

        mock_nlp = MagicMock()
        mock_nlp.meta = {"name": "test_model"}
        mock_nlp.return_value = MagicMock(ents=[])

        service = NERService(config=ner_config)
        service._initialized = True
        service._nlp = mock_nlp
        service._coref_model = None

        # Create entities with varying confidence
        entities = [
            FinancialEntity("A", "COMPANY", "A", 0, 1, confidence=0.9),
            FinancialEntity("B", "COMPANY", "B", 5, 6, confidence=0.5),  # Below threshold
        ]

        # Mock _extract_from_doc to return our entities
        service._extract_from_doc = MagicMock(return_value=entities)
        service._extract_cashtags = MagicMock(return_value=[])
        service._fuzzy_match_companies = MagicMock(return_value=[])

        result = service.extract_sync("test")

        # Only the high-confidence entity should remain
        assert len(result) == 1
        assert result[0].normalized == "A"


class TestIntegration:
    """Integration tests requiring real spaCy model."""

    @pytest.mark.integration
    def test_real_extraction(self):
        """Integration test with real spaCy model."""
        config = NERConfig(
            spacy_model="en_core_web_sm",
            enable_coreference=False,
        )
        service = NERService(config=config)

        text = "Nvidia announced the H100 GPU for AI training."
        entities = service.extract_sync(text)

        # Should find at least one entity
        assert len(entities) >= 1

    @pytest.mark.integration
    @pytest.mark.performance
    def test_extraction_latency(self):
        """Performance test for extraction latency."""
        import time

        config = NERConfig(
            spacy_model="en_core_web_sm",
            enable_coreference=False,
        )
        service = NERService(config=config)

        # Warm up
        service.extract_sync("Nvidia test")

        # Measure
        text = (
            "Nvidia announced the H100 GPU with HBM3E memory support. "
            "The company expects 20% revenue growth driven by AI demand."
        )
        start = time.perf_counter()
        for _ in range(10):
            service.extract_sync(text)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 10) * 1000
        # Should be reasonably fast with small model
        assert avg_ms < 500, f"Extraction too slow: {avg_ms:.1f}ms average"
