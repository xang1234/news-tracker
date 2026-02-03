"""
Performance benchmarks for vectorstore operations.

These tests verify the system meets performance requirements:
- Upsert throughput: >1000 vectors/second
- Search latency: <50ms

Run with: uv run pytest tests/test_vectorstore/test_performance.py -v -m performance

Note: These tests require a running PostgreSQL instance and may take
longer than typical unit tests. They are marked with @pytest.mark.performance
to allow selective execution.
"""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.vectorstore.base import VectorSearchFilter
from src.vectorstore.pgvector_store import PgVectorStore


@pytest.fixture
def large_embeddings():
    """Generate 1000 random 768-dimensional embeddings."""
    np.random.seed(42)  # Reproducibility
    return [np.random.randn(768).tolist() for _ in range(1000)]


@pytest.fixture
def large_ids():
    """Generate 1000 document IDs."""
    return [f"perf_test_doc_{i}" for i in range(1000)]


@pytest.mark.performance
class TestVectorStorePerformance:
    """Performance benchmarks for vectorstore operations."""

    @pytest.mark.asyncio
    async def test_upsert_throughput_mock(self, mock_database, large_embeddings, large_ids):
        """
        Verify upsert can handle >1000 vectors/second.

        This test uses mocked database to measure the overhead of the
        PgVectorStore layer itself, not the database.
        """
        # Mock returns True for all updates (fast)
        mock_repository = AsyncMock()
        mock_repository.update_embedding = AsyncMock(return_value=True)

        store = PgVectorStore(
            database=mock_database,
            repository=mock_repository,
        )

        n_vectors = len(large_embeddings)

        start = time.perf_counter()
        result = await store.upsert(ids=large_ids, embeddings=large_embeddings)
        elapsed = time.perf_counter() - start

        throughput = n_vectors / elapsed

        assert result == n_vectors, f"Expected {n_vectors} upserts, got {result}"
        # With mocked DB, throughput should be much higher than 1000/s
        # This test ensures the Python overhead is minimal
        assert throughput > 1000, (
            f"Upsert throughput {throughput:.0f}/s below 1000/s target "
            f"(mocked DB should be faster)"
        )

    @pytest.mark.asyncio
    async def test_search_latency_mock(self, mock_database, sample_embedding):
        """
        Verify search returns in <50ms with mocked database.

        This measures the Python-side overhead of query building and
        result processing, not database latency.
        """
        # Mock database returns results immediately
        mock_database.fetch = AsyncMock(return_value=[
            {
                "id": f"doc_{i}",
                "platform": "twitter",
                "url": f"https://twitter.com/user/status/{i}",
                "title": f"Document {i}",
                "content": f"Content for document {i} about semiconductors.",
                "author_name": f"user_{i}",
                "author_verified": i % 2 == 0,
                "author_followers": 1000 + i * 100,
                "tickers": ["NVDA", "AMD"],
                "theme_ids": [],
                "spam_score": 0.1,
                "authority_score": 0.7,
                "engagement": "{}",
                "timestamp": datetime.now(timezone.utc),
                "similarity": 0.95 - (i * 0.01),
            }
            for i in range(10)
        ])

        store = PgVectorStore(database=mock_database)

        # Warm up (first call may have initialization overhead)
        await store.search(query_embedding=sample_embedding, limit=10, threshold=0.7)

        # Measure search latency
        iterations = 10
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            results = await store.search(
                query_embedding=sample_embedding,
                limit=10,
                threshold=0.7,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            assert len(results) == 10

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        # With mocked DB, latency should be very low
        assert avg_latency < 50, (
            f"Average search latency {avg_latency:.1f}ms exceeds 50ms target"
        )
        assert max_latency < 100, (
            f"Max search latency {max_latency:.1f}ms exceeds 100ms threshold"
        )

    @pytest.mark.asyncio
    async def test_search_with_filters_latency_mock(self, mock_database, sample_embedding):
        """
        Verify filtered search adds minimal overhead.

        Filters add complexity to SQL building; this ensures it stays fast.
        """
        mock_database.fetch = AsyncMock(return_value=[])

        store = PgVectorStore(database=mock_database)

        # Create complex filter
        filters = VectorSearchFilter(
            platforms=["twitter", "reddit", "news"],
            tickers=["NVDA", "AMD", "INTC", "TSM"],
            min_authority_score=0.5,
            exclude_ids=[f"exclude_{i}" for i in range(100)],
            timestamp_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
            timestamp_before=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )

        # Measure filtered search latency
        iterations = 10
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            await store.search(
                query_embedding=sample_embedding,
                limit=100,
                threshold=0.5,
                filters=filters,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        avg_latency = sum(latencies) / len(latencies)

        # Filtered search should still be fast (just SQL building overhead)
        assert avg_latency < 50, (
            f"Filtered search latency {avg_latency:.1f}ms exceeds 50ms target"
        )

    @pytest.mark.asyncio
    async def test_result_conversion_throughput(self, mock_database):
        """
        Verify result conversion can handle large result sets.

        When searching with high limits, result conversion should be fast.
        """
        # Generate a large result set
        n_results = 500
        mock_rows = [
            {
                "id": f"doc_{i}",
                "platform": "twitter",
                "url": f"https://twitter.com/user/status/{i}",
                "title": f"Document title {i} about NVIDIA and AI",
                "content": f"Long content for document {i}. " * 50,  # ~2KB content
                "author_name": f"analyst_user_{i}",
                "author_verified": i % 3 == 0,
                "author_followers": 5000 + i * 10,
                "tickers": ["NVDA", "AMD", "INTC"],
                "theme_ids": ["ai_growth", "earnings"],
                "spam_score": 0.05,
                "authority_score": 0.8,
                "engagement": '{"likes": 100, "shares": 20, "comments": 5}',
                "timestamp": datetime.now(timezone.utc),
                "similarity": 0.99 - (i * 0.001),
            }
            for i in range(n_results)
        ]
        mock_database.fetch = AsyncMock(return_value=mock_rows)

        store = PgVectorStore(database=mock_database)

        # Generate a sample embedding
        sample_emb = [0.01] * 768

        start = time.perf_counter()
        results = await store.search(
            query_embedding=sample_emb,
            limit=n_results,
            threshold=0.0,  # Return all
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) == n_results
        # Processing 500 results should take <100ms
        assert elapsed_ms < 100, (
            f"Processing {n_results} results took {elapsed_ms:.1f}ms, expected <100ms"
        )
