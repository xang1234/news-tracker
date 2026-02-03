"""Unit tests for vectorstore base classes and data models."""

import pytest

from src.vectorstore.base import VectorSearchFilter, VectorSearchResult


class TestVectorSearchResult:
    """Tests for VectorSearchResult dataclass."""

    def test_create_basic_result(self):
        """Test creating a basic search result."""
        result = VectorSearchResult(
            document_id="doc_123",
            score=0.85,
            metadata={"platform": "twitter"},
        )

        assert result.document_id == "doc_123"
        assert result.score == 0.85
        assert result.metadata["platform"] == "twitter"
        assert result.embedding is None

    def test_create_result_with_embedding(self, sample_embedding):
        """Test creating a result with embedding included."""
        result = VectorSearchResult(
            document_id="doc_456",
            score=0.92,
            metadata={},
            embedding=sample_embedding,
        )

        assert result.embedding is not None
        assert len(result.embedding) == 768

    def test_invalid_score_above_one(self):
        """Test that scores above 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="Score must be between"):
            VectorSearchResult(
                document_id="doc_123",
                score=1.5,
                metadata={},
            )

    def test_invalid_score_below_zero(self):
        """Test that scores below 0.0 raise ValueError."""
        with pytest.raises(ValueError, match="Score must be between"):
            VectorSearchResult(
                document_id="doc_123",
                score=-0.1,
                metadata={},
            )

    def test_edge_case_score_zero(self):
        """Test that score of 0.0 is valid."""
        result = VectorSearchResult(
            document_id="doc_123",
            score=0.0,
            metadata={},
        )
        assert result.score == 0.0

    def test_edge_case_score_one(self):
        """Test that score of 1.0 is valid."""
        result = VectorSearchResult(
            document_id="doc_123",
            score=1.0,
            metadata={},
        )
        assert result.score == 1.0


class TestVectorSearchFilter:
    """Tests for VectorSearchFilter dataclass."""

    def test_create_empty_filter(self):
        """Test creating an empty filter."""
        f = VectorSearchFilter()

        assert f.platforms is None
        assert f.tickers is None
        assert f.theme_ids is None
        assert f.min_authority_score is None
        assert f.exclude_ids is None
        assert f.is_empty is True

    def test_create_filter_with_platforms(self):
        """Test creating a filter with platforms."""
        f = VectorSearchFilter(platforms=["twitter", "reddit"])

        assert f.platforms == ["twitter", "reddit"]
        assert f.is_empty is False

    def test_create_filter_with_tickers(self):
        """Test creating a filter with tickers."""
        f = VectorSearchFilter(tickers=["NVDA", "AMD", "INTC"])

        assert f.tickers == ["NVDA", "AMD", "INTC"]
        assert f.is_empty is False

    def test_create_filter_with_theme_ids(self):
        """Test creating a filter with theme IDs."""
        f = VectorSearchFilter(theme_ids=["ai_hype", "earnings"])

        assert f.theme_ids == ["ai_hype", "earnings"]
        assert f.is_empty is False

    def test_create_filter_with_authority_score(self):
        """Test creating a filter with minimum authority score."""
        f = VectorSearchFilter(min_authority_score=0.6)

        assert f.min_authority_score == 0.6
        assert f.is_empty is False

    def test_create_filter_with_exclude_ids(self):
        """Test creating a filter with excluded IDs."""
        f = VectorSearchFilter(exclude_ids=["doc_1", "doc_2"])

        assert f.exclude_ids == ["doc_1", "doc_2"]
        assert f.is_empty is False

    def test_create_full_filter(self, sample_filter):
        """Test creating a filter with all options."""
        assert sample_filter.platforms == ["twitter", "reddit"]
        assert sample_filter.tickers == ["NVDA", "AMD"]
        assert sample_filter.min_authority_score == 0.5
        assert sample_filter.is_empty is False

    def test_invalid_authority_score_above_one(self):
        """Test that authority score above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_authority_score must be between"):
            VectorSearchFilter(min_authority_score=1.5)

    def test_invalid_authority_score_below_zero(self):
        """Test that authority score below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_authority_score must be between"):
            VectorSearchFilter(min_authority_score=-0.1)

    def test_edge_case_authority_zero(self):
        """Test that authority score of 0.0 is valid."""
        f = VectorSearchFilter(min_authority_score=0.0)
        assert f.min_authority_score == 0.0

    def test_edge_case_authority_one(self):
        """Test that authority score of 1.0 is valid."""
        f = VectorSearchFilter(min_authority_score=1.0)
        assert f.min_authority_score == 1.0
