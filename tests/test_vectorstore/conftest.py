"""Pytest fixtures for vectorstore tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform
from src.vectorstore.base import VectorSearchFilter, VectorSearchResult
from src.vectorstore.config import VectorStoreConfig


@pytest.fixture
def vector_store_config() -> VectorStoreConfig:
    """Default vector store configuration for tests."""
    return VectorStoreConfig(
        default_limit=10,
        default_threshold=0.7,
        centroid_default_limit=100,
        centroid_default_threshold=0.5,
    )


@pytest.fixture
def sample_embedding() -> list[float]:
    """Sample 768-dimensional embedding."""
    # Return a normalized unit vector for testing
    import math
    dim = 768
    value = 1.0 / math.sqrt(dim)
    return [value] * dim


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Multiple sample embeddings for batch testing."""
    import math
    dim = 768
    embeddings = []
    for i in range(5):
        # Create slightly different embeddings
        value = (1.0 + i * 0.1) / math.sqrt(dim)
        embeddings.append([value] * dim)
    return embeddings


@pytest.fixture
def sample_search_result() -> VectorSearchResult:
    """Sample search result."""
    return VectorSearchResult(
        document_id="twitter_123",
        score=0.85,
        metadata={
            "platform": "twitter",
            "title": "NVIDIA AI News",
            "content_preview": "Breaking news about NVIDIA...",
            "author_name": "analyst_pro",
            "author_verified": True,
            "tickers": ["NVDA"],
            "authority_score": 0.75,
        },
    )


@pytest.fixture
def sample_search_results() -> list[VectorSearchResult]:
    """Multiple sample search results."""
    return [
        VectorSearchResult(
            document_id="twitter_1",
            score=0.95,
            metadata={"platform": "twitter", "tickers": ["NVDA"]},
        ),
        VectorSearchResult(
            document_id="reddit_2",
            score=0.88,
            metadata={"platform": "reddit", "tickers": ["NVDA", "AMD"]},
        ),
        VectorSearchResult(
            document_id="news_3",
            score=0.75,
            metadata={"platform": "news", "tickers": ["INTC"]},
        ),
    ]


@pytest.fixture
def sample_filter() -> VectorSearchFilter:
    """Sample search filter."""
    return VectorSearchFilter(
        platforms=["twitter", "reddit"],
        tickers=["NVDA", "AMD"],
        min_authority_score=0.5,
    )


@pytest.fixture
def mock_database() -> AsyncMock:
    """Mock Database instance."""
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock(return_value=None)
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="OK")
    return db


@pytest.fixture
def mock_repository() -> AsyncMock:
    """Mock DocumentRepository instance."""
    repo = AsyncMock()
    repo.update_embedding = AsyncMock(return_value=True)
    repo.get_by_id = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """Mock EmbeddingService instance."""
    import math
    dim = 768
    value = 1.0 / math.sqrt(dim)
    embedding = [value] * dim

    service = AsyncMock()
    service.embed_finbert = AsyncMock(return_value=embedding)
    service.embed_batch = AsyncMock(return_value=[embedding])
    service.close = AsyncMock()
    return service


@pytest.fixture
def document_for_authority() -> NormalizedDocument:
    """Document for testing authority score computation."""
    return NormalizedDocument(
        id="twitter_auth_test",
        platform=Platform.TWITTER,
        timestamp=datetime.now(timezone.utc),
        author_id="verified_analyst",
        author_name="Top Analyst",
        author_followers=100000,
        author_verified=True,
        content="$NVDA analysis: Strong growth expected in datacenter segment.",
        engagement=EngagementMetrics(
            likes=500,
            shares=100,
            comments=50,
        ),
        spam_score=0.1,
    )


@pytest.fixture
def low_authority_document() -> NormalizedDocument:
    """Low authority document for testing."""
    return NormalizedDocument(
        id="twitter_low_auth",
        platform=Platform.TWITTER,
        timestamp=datetime.now(timezone.utc),
        author_id="new_user",
        author_name="New User",
        author_followers=10,
        author_verified=False,
        content="Just bought some $NVDA",
        engagement=EngagementMetrics(
            likes=1,
            shares=0,
            comments=0,
        ),
        spam_score=0.5,
    )
