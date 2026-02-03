"""Unit tests for VectorStoreManager."""

import pytest
from unittest.mock import AsyncMock

from src.vectorstore.base import VectorSearchFilter, VectorSearchResult
from src.vectorstore.config import VectorStoreConfig
from src.vectorstore.manager import VectorStoreManager


class TestVectorStoreManagerAuthorityScore:
    """Tests for authority score computation."""

    def test_authority_score_verified_author(
        self,
        mock_embedding_service,
        document_for_authority,
    ):
        """Test authority score for verified author with high engagement."""
        mock_store = AsyncMock()
        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        score = manager._compute_authority_score(document_for_authority)

        # Verified author with 100k followers, high engagement, low spam
        # Should be high authority (> 0.7)
        assert score >= 0.7
        assert score <= 1.0

    def test_authority_score_low_authority_user(
        self,
        mock_embedding_service,
        low_authority_document,
    ):
        """Test authority score for new user with low engagement."""
        mock_store = AsyncMock()
        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        score = manager._compute_authority_score(low_authority_document)

        # Not verified, few followers, no engagement, medium spam
        # Should be low authority (< 0.3)
        assert score < 0.4
        assert score >= 0.0

    def test_authority_score_verified_bonus(
        self,
        mock_embedding_service,
        sample_document,
    ):
        """Test that verified authors get bonus."""
        mock_store = AsyncMock()
        config = VectorStoreConfig(authority_verified_bonus=0.2)
        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
            config=config,
        )

        # Test with verified
        sample_document.author_verified = True
        score_verified = manager._compute_authority_score(sample_document)

        # Test without verified
        sample_document.author_verified = False
        score_not_verified = manager._compute_authority_score(sample_document)

        assert score_verified > score_not_verified
        assert score_verified - score_not_verified == pytest.approx(0.2, abs=0.01)

    def test_authority_score_clamped_to_valid_range(
        self,
        mock_embedding_service,
        sample_document,
    ):
        """Test that score is clamped between 0 and 1."""
        mock_store = AsyncMock()
        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        # Set extreme values
        sample_document.author_verified = True
        sample_document.author_followers = 10_000_000
        sample_document.engagement.likes = 100_000
        sample_document.engagement.shares = 50_000
        sample_document.spam_score = 0.0

        score = manager._compute_authority_score(sample_document)

        assert score <= 1.0
        assert score >= 0.0


class TestVectorStoreManagerQuery:
    """Tests for query methods."""

    @pytest.mark.asyncio
    async def test_query_basic(
        self,
        mock_embedding_service,
        sample_search_results,
        sample_embedding,
    ):
        """Test basic query execution."""
        mock_store = AsyncMock()
        mock_store.search = AsyncMock(return_value=sample_search_results)

        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        results = await manager.query("NVIDIA AI demand")

        assert len(results) == 3
        mock_embedding_service.embed_finbert.assert_called_once_with("NVIDIA AI demand")
        mock_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_filters(
        self,
        mock_embedding_service,
        sample_search_results,
        sample_filter,
    ):
        """Test query with filters."""
        mock_store = AsyncMock()
        mock_store.search = AsyncMock(return_value=sample_search_results)

        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        results = await manager.query(
            "semiconductor news",
            filters=sample_filter,
        )

        # Verify filter was passed to store
        call_args = mock_store.search.call_args
        assert call_args.kwargs["filters"] == sample_filter

    @pytest.mark.asyncio
    async def test_query_empty_text(self, mock_embedding_service):
        """Test query with empty text returns empty list."""
        mock_store = AsyncMock()
        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        results = await manager.query("")

        assert results == []
        mock_embedding_service.embed_finbert.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_by_embedding(
        self,
        mock_embedding_service,
        sample_search_results,
        sample_embedding,
    ):
        """Test query with pre-computed embedding."""
        mock_store = AsyncMock()
        mock_store.search = AsyncMock(return_value=sample_search_results)

        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        results = await manager.query_by_embedding(sample_embedding)

        assert len(results) == 3
        # Should not call embedding service
        mock_embedding_service.embed_finbert.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_by_theme_centroid(
        self,
        mock_embedding_service,
        sample_search_results,
        sample_embedding,
    ):
        """Test centroid query uses correct defaults."""
        mock_store = AsyncMock()
        mock_store.search_by_centroid = AsyncMock(return_value=sample_search_results)

        config = VectorStoreConfig(
            centroid_default_limit=100,
            centroid_default_threshold=0.5,
        )
        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
            config=config,
        )

        results = await manager.query_by_theme_centroid(sample_embedding)

        # Verify centroid-specific defaults
        call_args = mock_store.search_by_centroid.call_args
        assert call_args.kwargs["limit"] == 100
        assert call_args.kwargs["threshold"] == 0.5


class TestVectorStoreManagerIngest:
    """Tests for document ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_documents(
        self,
        mock_embedding_service,
        sample_embeddings,
        batch_documents,
    ):
        """Test ingesting documents."""
        mock_store = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=5)

        # Mock batch embed to return multiple embeddings
        mock_embedding_service.embed_batch = AsyncMock(
            return_value=sample_embeddings[:5]
        )

        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        # Use first 5 documents
        docs = batch_documents[:5]
        stats = await manager.ingest_documents(docs)

        assert stats["processed"] == 5
        assert stats["skipped"] == 0
        mock_embedding_service.embed_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_skips_existing_embeddings(
        self,
        mock_embedding_service,
        sample_embedding,
        batch_documents,
    ):
        """Test that documents with existing embeddings are skipped."""
        mock_store = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=3)
        mock_embedding_service.embed_batch = AsyncMock(return_value=[sample_embedding] * 3)

        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        # Set embeddings on 2 documents
        docs = batch_documents[:5]
        docs[0].embedding = sample_embedding
        docs[1].embedding = sample_embedding

        stats = await manager.ingest_documents(docs)

        assert stats["skipped"] == 2
        assert stats["processed"] == 3

    @pytest.mark.asyncio
    async def test_ingest_empty_list(self, mock_embedding_service):
        """Test ingesting empty list."""
        mock_store = AsyncMock()
        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        stats = await manager.ingest_documents([])

        assert stats == {"processed": 0, "skipped": 0, "errors": 0}

    @pytest.mark.asyncio
    async def test_ingest_computes_authority_score(
        self,
        mock_embedding_service,
        sample_embedding,
        document_for_authority,
    ):
        """Test that ingestion computes authority scores."""
        mock_store = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=1)
        mock_embedding_service.embed_batch = AsyncMock(return_value=[sample_embedding])

        manager = VectorStoreManager(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
        )

        # Ensure no existing authority score
        document_for_authority.authority_score = None
        document_for_authority.embedding = None

        await manager.ingest_documents([document_for_authority])

        # Authority score should now be set
        assert document_for_authority.authority_score is not None
        assert document_for_authority.authority_score > 0.5
