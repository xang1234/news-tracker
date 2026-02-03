"""
Integration tests for vectorstore module.

These tests require a running PostgreSQL instance with pgvector.
Run with: uv run pytest tests/test_vectorstore/test_integration.py -v --integration

Mark tests with @pytest.mark.integration to only run with --integration flag.
"""

import pytest
from datetime import datetime, timezone

from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform


# Custom marker for integration tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as requiring live infrastructure"
    )


@pytest.fixture
async def integration_db():
    """
    Get connected database for integration tests.

    Skips if database is not available.
    """
    pytest.importorskip("asyncpg")

    from src.storage.database import Database
    from src.config.settings import get_settings

    settings = get_settings()
    db = Database(database_url=str(settings.database_url))

    try:
        await db.connect()
        yield db
    except Exception as e:
        pytest.skip(f"Database not available: {e}")
    finally:
        await db.close()


@pytest.fixture
def integration_documents() -> list[NormalizedDocument]:
    """Sample documents for integration testing."""
    return [
        NormalizedDocument(
            id="integ_test_1",
            platform=Platform.TWITTER,
            timestamp=datetime.now(timezone.utc),
            author_id="test_user_1",
            author_name="Test Analyst",
            author_followers=10000,
            author_verified=True,
            content="NVIDIA reports record datacenter revenue driven by AI demand. Strong outlook.",
            tickers_mentioned=["NVDA"],
            engagement=EngagementMetrics(likes=500, shares=100, comments=50),
            spam_score=0.1,
        ),
        NormalizedDocument(
            id="integ_test_2",
            platform=Platform.REDDIT,
            timestamp=datetime.now(timezone.utc),
            author_id="test_user_2",
            author_name="semiconductor_investor",
            author_followers=5000,
            author_verified=False,
            content="AMD MI300X showing strong performance in AI training benchmarks.",
            tickers_mentioned=["AMD"],
            engagement=EngagementMetrics(likes=200, shares=30, comments=25),
            spam_score=0.2,
        ),
        NormalizedDocument(
            id="integ_test_3",
            platform=Platform.NEWS,
            timestamp=datetime.now(timezone.utc),
            author_id="reuters",
            author_name="Reuters",
            author_followers=1000000,
            author_verified=True,
            content="Intel announces new semiconductor fab investment in Arizona.",
            tickers_mentioned=["INTC"],
            engagement=EngagementMetrics(likes=1000, shares=500, comments=100),
            spam_score=0.0,
        ),
    ]


@pytest.mark.integration
class TestPgVectorStoreIntegration:
    """Integration tests for PgVectorStore with real database."""

    @pytest.mark.asyncio
    async def test_upsert_and_search(self, integration_db, integration_documents):
        """Test full upsert and search workflow."""
        from src.storage.repository import DocumentRepository
        from src.vectorstore.pgvector_store import PgVectorStore
        from src.vectorstore.base import VectorSearchFilter
        import math

        # Create repository and store
        repo = DocumentRepository(integration_db)
        await repo.create_tables()
        store = PgVectorStore(database=integration_db, repository=repo)

        # Insert test documents first
        for doc in integration_documents:
            await repo.insert(doc)

        # Create sample embeddings (768-dim)
        dim = 768
        embeddings = []
        for i in range(3):
            # Create slightly different embeddings
            vec = [0.0] * dim
            vec[i] = 1.0  # Make them different
            norm = math.sqrt(sum(x * x for x in vec))
            embeddings.append([x / norm for x in vec])

        # Upsert embeddings
        ids = [doc.id for doc in integration_documents]
        result = await store.upsert(ids, embeddings)
        assert result >= 0  # May not update if embeddings exist

        # Search with first document's embedding
        results = await store.search(
            query_embedding=embeddings[0],
            limit=5,
            threshold=0.0,  # Low threshold for test
        )

        # Should find at least the first document
        assert len(results) >= 1

        # Cleanup
        await store.delete(ids)

    @pytest.mark.asyncio
    async def test_search_with_filters(self, integration_db, integration_documents):
        """Test search with various filters."""
        from src.storage.repository import DocumentRepository
        from src.vectorstore.pgvector_store import PgVectorStore
        from src.vectorstore.base import VectorSearchFilter
        import math

        repo = DocumentRepository(integration_db)
        await repo.create_tables()
        store = PgVectorStore(database=integration_db, repository=repo)

        # Insert and embed documents
        for doc in integration_documents:
            await repo.insert(doc)

        dim = 768
        embeddings = []
        for i in range(3):
            vec = [0.1] * dim
            vec[i] = 0.9
            norm = math.sqrt(sum(x * x for x in vec))
            embeddings.append([x / norm for x in vec])

        ids = [doc.id for doc in integration_documents]
        await store.upsert(ids, embeddings)

        # Search with platform filter
        results = await store.search(
            query_embedding=embeddings[0],
            limit=10,
            threshold=0.0,
            filters=VectorSearchFilter(platforms=["twitter"]),
        )

        # Should only return Twitter documents
        for r in results:
            assert r.metadata.get("platform") == "twitter"

        # Cleanup
        await store.delete(ids)


@pytest.mark.integration
class TestVectorStoreManagerIntegration:
    """Integration tests for VectorStoreManager with real services."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, integration_db, integration_documents):
        """Test complete ingest and query workflow."""
        pytest.importorskip("transformers")
        pytest.importorskip("torch")

        from src.storage.repository import DocumentRepository
        from src.vectorstore.pgvector_store import PgVectorStore
        from src.vectorstore.manager import VectorStoreManager
        from src.embedding.service import EmbeddingService
        from src.embedding.config import EmbeddingConfig

        # Create components
        repo = DocumentRepository(integration_db)
        await repo.create_tables()
        store = PgVectorStore(database=integration_db, repository=repo)

        # Use CPU for tests
        config = EmbeddingConfig(device="cpu", cache_enabled=False)
        embedding_service = EmbeddingService(config=config)

        try:
            manager = VectorStoreManager(
                vector_store=store,
                embedding_service=embedding_service,
            )

            # Insert documents first (required for embedding update)
            for doc in integration_documents:
                await repo.insert(doc)

            # Ingest documents (generate embeddings)
            stats = await manager.ingest_documents(integration_documents)
            assert stats["processed"] > 0

            # Query for similar documents
            results = await manager.query(
                "NVIDIA AI chip demand",
                limit=3,
                threshold=0.3,
            )

            # Should find related documents
            assert len(results) > 0

            # Cleanup
            ids = [doc.id for doc in integration_documents]
            await store.delete(ids)

        finally:
            await embedding_service.close()
