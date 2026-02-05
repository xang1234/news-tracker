"""Tests for KeywordsService integration with Preprocessor."""

import pytest
from datetime import datetime, timezone

from src.ingestion.preprocessor import Preprocessor
from src.ingestion.schemas import NormalizedDocument, Platform
from src.keywords.service import KeywordsService


class TestPreprocessorKeywordsIntegration:
    """Tests for keywords integration in the preprocessing pipeline."""

    def create_test_doc(self, content: str) -> NormalizedDocument:
        """Create a test document with given content."""
        return NormalizedDocument(
            id="test_1",
            platform=Platform.NEWS,
            url="https://example.com/article",
            timestamp=datetime.now(timezone.utc),
            author_id="author_1",
            author_name="Test Author",
            content=content,
        )

    def test_preprocessor_with_keywords_disabled(self):
        """Test preprocessor does not extract keywords when disabled."""
        preprocessor = Preprocessor(enable_keywords=False)
        doc = self.create_test_doc(
            "Nvidia announced new GPU architecture with HBM3E memory support."
        )

        processed = preprocessor.process(doc)

        assert processed.keywords_extracted == []

    def test_preprocessor_with_keywords_enabled_no_service(self):
        """Test preprocessor with keywords enabled but no service provided."""
        preprocessor = Preprocessor(enable_keywords=True, keywords_service=None)
        doc = self.create_test_doc(
            "Nvidia announced new GPU architecture with HBM3E memory support."
        )

        processed = preprocessor.process(doc)

        # Should not crash, just return empty keywords
        assert processed.keywords_extracted == []

    def test_preprocessor_with_keywords_enabled(self):
        """Test preprocessor extracts keywords when enabled with service."""
        keywords_svc = KeywordsService()
        preprocessor = Preprocessor(
            keywords_service=keywords_svc,
            enable_keywords=True,
        )
        doc = self.create_test_doc(
            "Nvidia announced new GPU architecture with HBM3E memory support. "
            "The semiconductor industry expects strong demand for AI accelerator chips."
        )

        processed = preprocessor.process(doc)

        assert len(processed.keywords_extracted) > 0
        # Check structure of extracted keywords
        for kw in processed.keywords_extracted:
            assert "text" in kw
            assert "score" in kw
            assert "rank" in kw
            assert "lemma" in kw

    def test_preprocessor_keywords_are_serializable(self):
        """Test extracted keywords can be serialized to JSON."""
        import json

        keywords_svc = KeywordsService()
        preprocessor = Preprocessor(
            keywords_service=keywords_svc,
            enable_keywords=True,
        )
        doc = self.create_test_doc(
            "Nvidia announced new GPU architecture for data center applications."
        )

        processed = preprocessor.process(doc)

        # Should be JSON serializable
        json_str = json.dumps(processed.keywords_extracted)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_preprocessor_empty_content(self):
        """Test preprocessor handles empty content gracefully."""
        keywords_svc = KeywordsService()
        preprocessor = Preprocessor(
            keywords_service=keywords_svc,
            enable_keywords=True,
        )
        doc = self.create_test_doc("Short text")

        processed = preprocessor.process(doc)

        # Should not crash, keywords may or may not be extracted
        assert isinstance(processed.keywords_extracted, list)

    def test_preprocessor_combined_with_tickers(self):
        """Test keywords extraction works alongside ticker extraction."""
        keywords_svc = KeywordsService()
        preprocessor = Preprocessor(
            keywords_service=keywords_svc,
            enable_keywords=True,
        )
        doc = self.create_test_doc(
            "$NVDA announced new GPU architecture. "
            "Intel and AMD are also developing AI accelerators. "
            "The semiconductor industry expects strong growth."
        )

        processed = preprocessor.process(doc)

        # Both tickers and keywords should be extracted
        assert "NVDA" in processed.tickers_mentioned
        assert len(processed.keywords_extracted) > 0
