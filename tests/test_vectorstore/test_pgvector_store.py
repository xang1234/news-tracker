"""Unit tests for PgVectorStore implementation."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.vectorstore.base import VectorSearchFilter
from src.vectorstore.pgvector_store import PgVectorStore


class TestPgVectorStoreUpsert:
    """Tests for PgVectorStore.upsert()."""

    @pytest.mark.asyncio
    async def test_upsert_single_embedding(
        self,
        mock_database,
        mock_repository,
        sample_embedding,
    ):
        """Test upserting a single embedding."""
        mock_repository.update_embedding = AsyncMock(return_value=True)
        store = PgVectorStore(
            database=mock_database,
            repository=mock_repository,
        )

        result = await store.upsert(
            ids=["doc_1"],
            embeddings=[sample_embedding],
        )

        assert result == 1
        mock_repository.update_embedding.assert_called_once_with(
            "doc_1", sample_embedding
        )

    @pytest.mark.asyncio
    async def test_upsert_multiple_embeddings(
        self,
        mock_database,
        mock_repository,
        sample_embeddings,
    ):
        """Test upserting multiple embeddings."""
        mock_repository.update_embedding = AsyncMock(return_value=True)
        store = PgVectorStore(
            database=mock_database,
            repository=mock_repository,
        )

        ids = [f"doc_{i}" for i in range(5)]
        result = await store.upsert(ids=ids, embeddings=sample_embeddings)

        assert result == 5
        assert mock_repository.update_embedding.call_count == 5

    @pytest.mark.asyncio
    async def test_upsert_partial_success(
        self,
        mock_database,
        mock_repository,
        sample_embeddings,
    ):
        """Test upsert with some failures."""
        # First and third succeed, others fail
        mock_repository.update_embedding = AsyncMock(
            side_effect=[True, False, True, False, False]
        )
        store = PgVectorStore(
            database=mock_database,
            repository=mock_repository,
        )

        ids = [f"doc_{i}" for i in range(5)]
        result = await store.upsert(ids=ids, embeddings=sample_embeddings)

        assert result == 2  # Only 2 succeeded

    @pytest.mark.asyncio
    async def test_upsert_mismatched_lengths(
        self,
        mock_database,
        mock_repository,
        sample_embedding,
    ):
        """Test upsert raises error for mismatched lengths."""
        store = PgVectorStore(
            database=mock_database,
            repository=mock_repository,
        )

        with pytest.raises(ValueError, match="must have same length"):
            await store.upsert(
                ids=["doc_1", "doc_2"],
                embeddings=[sample_embedding],  # Only 1 embedding for 2 IDs
            )


class TestPgVectorStoreSearch:
    """Tests for PgVectorStore.search()."""

    @pytest.mark.asyncio
    async def test_search_no_filters(
        self,
        mock_database,
        sample_embedding,
    ):
        """Test basic search without filters."""
        mock_database.fetch = AsyncMock(return_value=[
            {
                "id": "doc_1",
                "platform": "twitter",
                "url": "https://twitter.com/...",
                "title": None,
                "content": "NVIDIA reports strong earnings",
                "author_name": "analyst",
                "author_verified": True,
                "author_followers": 5000,
                "tickers": ["NVDA"],
                "theme_ids": [],
                "spam_score": 0.1,
                "authority_score": 0.8,
                "engagement": json.dumps({"likes": 100, "shares": 20}),
                "timestamp": datetime.now(timezone.utc),
                "similarity": 0.92,
            }
        ])

        store = PgVectorStore(database=mock_database)
        results = await store.search(
            query_embedding=sample_embedding,
            limit=10,
            threshold=0.7,
        )

        assert len(results) == 1
        assert results[0].document_id == "doc_1"
        assert results[0].score == 0.92
        assert results[0].metadata["platform"] == "twitter"

    @pytest.mark.asyncio
    async def test_search_with_platform_filter(
        self,
        mock_database,
        sample_embedding,
    ):
        """Test search with platform filter."""
        mock_database.fetch = AsyncMock(return_value=[])
        store = PgVectorStore(database=mock_database)

        filters = VectorSearchFilter(platforms=["twitter", "reddit"])
        await store.search(
            query_embedding=sample_embedding,
            limit=10,
            threshold=0.7,
            filters=filters,
        )

        # Verify the SQL includes platform filter
        call_args = mock_database.fetch.call_args
        sql = call_args[0][0]
        assert "platform = ANY" in sql

    @pytest.mark.asyncio
    async def test_search_with_ticker_filter(
        self,
        mock_database,
        sample_embedding,
    ):
        """Test search with ticker filter."""
        mock_database.fetch = AsyncMock(return_value=[])
        store = PgVectorStore(database=mock_database)

        filters = VectorSearchFilter(tickers=["NVDA", "AMD"])
        await store.search(
            query_embedding=sample_embedding,
            limit=10,
            threshold=0.7,
            filters=filters,
        )

        call_args = mock_database.fetch.call_args
        sql = call_args[0][0]
        assert "tickers &&" in sql  # Array overlap operator

    @pytest.mark.asyncio
    async def test_search_with_authority_filter(
        self,
        mock_database,
        sample_embedding,
    ):
        """Test search with authority score filter."""
        mock_database.fetch = AsyncMock(return_value=[])
        store = PgVectorStore(database=mock_database)

        filters = VectorSearchFilter(min_authority_score=0.6)
        await store.search(
            query_embedding=sample_embedding,
            limit=10,
            threshold=0.7,
            filters=filters,
        )

        call_args = mock_database.fetch.call_args
        sql = call_args[0][0]
        assert "authority_score >=" in sql

    @pytest.mark.asyncio
    async def test_search_with_exclude_ids(
        self,
        mock_database,
        sample_embedding,
    ):
        """Test search with excluded IDs."""
        mock_database.fetch = AsyncMock(return_value=[])
        store = PgVectorStore(database=mock_database)

        filters = VectorSearchFilter(exclude_ids=["doc_1", "doc_2"])
        await store.search(
            query_embedding=sample_embedding,
            limit=10,
            threshold=0.7,
            filters=filters,
        )

        call_args = mock_database.fetch.call_args
        sql = call_args[0][0]
        assert "id != ALL" in sql


class TestPgVectorStoreDelete:
    """Tests for PgVectorStore.delete()."""

    @pytest.mark.asyncio
    async def test_delete_documents(self, mock_database):
        """Test deleting documents."""
        mock_database.fetch = AsyncMock(return_value=[
            {"id": "doc_1"},
            {"id": "doc_2"},
        ])
        store = PgVectorStore(database=mock_database)

        result = await store.delete(["doc_1", "doc_2", "doc_3"])

        assert result == 2  # Only 2 were actually deleted

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, mock_database):
        """Test deleting with empty list."""
        store = PgVectorStore(database=mock_database)

        result = await store.delete([])

        assert result == 0
        mock_database.fetch.assert_not_called()


class TestPgVectorStoreGetByIds:
    """Tests for PgVectorStore.get_by_ids()."""

    @pytest.mark.asyncio
    async def test_get_by_ids(self, mock_database):
        """Test getting documents by IDs."""
        mock_database.fetch = AsyncMock(return_value=[
            {
                "id": "doc_1",
                "platform": "twitter",
                "url": None,
                "title": "Test",
                "content": "Content here",
                "author_name": "user",
                "author_verified": False,
                "author_followers": 100,
                "tickers": ["NVDA"],
                "theme_ids": [],
                "spam_score": 0.0,
                "authority_score": 0.5,
                "engagement": "{}",
                "timestamp": datetime.now(timezone.utc),
            }
        ])
        store = PgVectorStore(database=mock_database)

        results = await store.get_by_ids(["doc_1"])

        assert len(results) == 1
        assert results[0].document_id == "doc_1"
        assert results[0].score == 1.0  # Exact match

    @pytest.mark.asyncio
    async def test_get_by_ids_empty(self, mock_database):
        """Test get_by_ids with empty list."""
        store = PgVectorStore(database=mock_database)

        results = await store.get_by_ids([])

        assert results == []
        mock_database.fetch.assert_not_called()
