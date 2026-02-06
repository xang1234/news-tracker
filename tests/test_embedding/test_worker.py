"""Tests for EmbeddingWorker."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clustering.queue import ClusteringQueue
from src.embedding.config import EmbeddingConfig
from src.embedding.queue import EmbeddingJob, EmbeddingQueue
from src.embedding.service import EmbeddingService, ModelType
from src.embedding.worker import EmbeddingWorker
from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform
from src.storage.database import Database
from src.storage.repository import DocumentRepository


@pytest.fixture
def sample_documents() -> list[NormalizedDocument]:
    """Create sample documents for testing."""
    return [
        NormalizedDocument(
            id=f"doc_{i}",
            platform=Platform.NEWS,
            url=f"https://example.com/article/{i}",
            timestamp=datetime.now(timezone.utc),
            author_id=f"author_{i}",
            author_name=f"Author {i}",
            content=f"Financial news content about semiconductors number {i}",
            content_type="article",
            title=f"Article Title {i}",
            engagement=EngagementMetrics(likes=10, shares=5, comments=2),
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_database():
    """Create a mock database."""
    db = AsyncMock(spec=Database)
    db.connect = AsyncMock()
    db.close = AsyncMock()
    db.health_check = AsyncMock(return_value=True)
    return db


@pytest.fixture
def mock_repository(sample_documents):
    """Create a mock repository."""
    repo = AsyncMock(spec=DocumentRepository)

    # Map documents by ID
    doc_map = {doc.id: doc for doc in sample_documents}

    async def get_by_id(doc_id):
        return doc_map.get(doc_id)

    repo.get_by_id = AsyncMock(side_effect=get_by_id)
    repo.update_embedding = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_queue():
    """Create a mock embedding queue."""
    queue = AsyncMock(spec=EmbeddingQueue)
    queue.connect = AsyncMock()
    queue.close = AsyncMock()
    queue.ack = AsyncMock()
    queue.nack = AsyncMock()
    queue.health_check = AsyncMock(return_value=True)
    return queue


@pytest.fixture
def mock_embedding_service_for_worker(sample_embeddings):
    """Create a mock embedding service for worker tests."""
    service = AsyncMock(spec=EmbeddingService)

    call_count = [0]

    async def embed(text, model_type=ModelType.FINBERT):
        idx = call_count[0] % len(sample_embeddings)
        call_count[0] += 1
        return sample_embeddings[idx]

    async def embed_batch(texts, model_type=ModelType.FINBERT, show_progress=False):
        return [sample_embeddings[i % len(sample_embeddings)] for i in range(len(texts))]

    service.embed = AsyncMock(side_effect=embed)
    service.embed_batch = AsyncMock(side_effect=embed_batch)
    service.close = AsyncMock()
    service.get_stats = MagicMock(return_value={"initialized": True})

    return service


class TestEmbeddingWorkerInitialization:
    """Tests for worker initialization."""

    def test_worker_initialization(self):
        """Should initialize with default config."""
        worker = EmbeddingWorker()

        assert worker._batch_size == 32
        assert not worker.is_running

    def test_worker_custom_config(self):
        """Should accept custom configuration."""
        config = EmbeddingConfig(batch_size=16)
        worker = EmbeddingWorker(config=config, batch_size=16)

        assert worker._batch_size == 16


class TestEmbeddingWorkerProcessing:
    """Tests for embedding job processing."""

    @pytest.mark.asyncio
    async def test_process_batch_success(
        self,
        mock_queue,
        mock_database,
        mock_repository,
        mock_embedding_service_for_worker,
        sample_documents,
    ):
        """Should process batch successfully."""
        config = EmbeddingConfig(batch_size=4)
        worker = EmbeddingWorker(
            queue=mock_queue,
            database=mock_database,
            embedding_service=mock_embedding_service_for_worker,
            config=config,
        )
        worker._repository = mock_repository

        # Create jobs for first 3 documents
        jobs = [
            EmbeddingJob(message_id=f"msg_{i}", document_id=f"doc_{i}")
            for i in range(3)
        ]

        await worker._process_batch(jobs)

        # All jobs should be acknowledged
        assert mock_queue.ack.call_count == 3

        # Embeddings should be generated
        mock_embedding_service_for_worker.embed_batch.assert_called_once()

        # Documents should be updated
        assert mock_repository.update_embedding.call_count == 3

    @pytest.mark.asyncio
    async def test_process_batch_document_not_found(
        self,
        mock_queue,
        mock_database,
        mock_embedding_service_for_worker,
    ):
        """Should skip jobs for non-existent documents."""
        config = EmbeddingConfig(batch_size=4)
        worker = EmbeddingWorker(
            queue=mock_queue,
            database=mock_database,
            embedding_service=mock_embedding_service_for_worker,
            config=config,
        )

        # Mock repository that returns None
        mock_repo = AsyncMock()
        mock_repo.get_by_id = AsyncMock(return_value=None)
        mock_repo.update_embedding = AsyncMock()
        worker._repository = mock_repo

        jobs = [EmbeddingJob(message_id="msg_1", document_id="nonexistent")]

        await worker._process_batch(jobs)

        # Job should be acknowledged (skipped)
        mock_queue.ack.assert_called_once()

        # No embedding should be generated
        mock_embedding_service_for_worker.embed_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_batch_already_embedded(
        self,
        mock_queue,
        mock_database,
        mock_embedding_service_for_worker,
        sample_documents,
    ):
        """Should skip documents that already have embeddings."""
        # Add embedding to document
        doc = sample_documents[0]
        doc.embedding = [0.1] * 768

        config = EmbeddingConfig(batch_size=4)
        worker = EmbeddingWorker(
            queue=mock_queue,
            database=mock_database,
            embedding_service=mock_embedding_service_for_worker,
            config=config,
        )

        mock_repo = AsyncMock()
        mock_repo.get_by_id = AsyncMock(return_value=doc)
        mock_repo.update_embedding = AsyncMock()
        worker._repository = mock_repo

        jobs = [EmbeddingJob(message_id="msg_1", document_id=doc.id)]

        await worker._process_batch(jobs)

        # Job should be acknowledged (skipped)
        mock_queue.ack.assert_called_once()

        # No new embedding should be generated
        mock_embedding_service_for_worker.embed_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_batch_embedding_error(
        self,
        mock_queue,
        mock_database,
        sample_documents,
    ):
        """Should nack jobs when embedding fails."""
        # Mock service that raises error
        mock_service = AsyncMock()
        mock_service.embed_batch = AsyncMock(side_effect=Exception("Model error"))
        mock_service.close = AsyncMock()

        config = EmbeddingConfig(batch_size=4)
        worker = EmbeddingWorker(
            queue=mock_queue,
            database=mock_database,
            embedding_service=mock_service,
            config=config,
        )

        mock_repo = AsyncMock()
        mock_repo.get_by_id = AsyncMock(side_effect=lambda id: sample_documents[0])
        worker._repository = mock_repo

        jobs = [EmbeddingJob(message_id="msg_1", document_id="doc_0")]

        await worker._process_batch(jobs)

        # Job should be nacked due to error
        mock_queue.nack.assert_called()

    @pytest.mark.asyncio
    async def test_process_batch_combines_title_and_content(
        self,
        mock_queue,
        mock_database,
        mock_repository,
        mock_embedding_service_for_worker,
        sample_documents,
    ):
        """Should combine title and content for embedding."""
        config = EmbeddingConfig(batch_size=4)
        worker = EmbeddingWorker(
            queue=mock_queue,
            database=mock_database,
            embedding_service=mock_embedding_service_for_worker,
            config=config,
        )
        worker._repository = mock_repository

        jobs = [EmbeddingJob(message_id="msg_0", document_id="doc_0")]

        await worker._process_batch(jobs)

        # Check that embed_batch was called with title + content
        call_args = mock_embedding_service_for_worker.embed_batch.call_args
        texts = call_args[0][0]
        assert len(texts) == 1
        assert "Article Title 0" in texts[0]
        assert "Financial news content" in texts[0]


class TestEmbeddingWorkerRunOnce:
    """Tests for run_once method."""

    @pytest.mark.asyncio
    async def test_run_once_processes_documents(
        self,
        mock_database,
        mock_embedding_service_for_worker,
        sample_documents,
        mock_redis,
    ):
        """Should process documents without queue."""
        config = EmbeddingConfig(batch_size=4)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            worker = EmbeddingWorker(
                database=mock_database,
                embedding_service=mock_embedding_service_for_worker,
                config=config,
            )

            mock_repo = AsyncMock()
            doc_map = {doc.id: doc for doc in sample_documents[:3]}
            mock_repo.get_by_id = AsyncMock(side_effect=lambda id: doc_map.get(id))
            mock_repo.update_embedding = AsyncMock(return_value=True)

            # Patch repository creation
            with patch.object(
                worker, "_fetch_documents",
                return_value=list(doc_map.values()),
            ):
                worker._repository = mock_repo

                stats = await worker.run_once(["doc_0", "doc_1", "doc_2"])

                assert stats["total"] == 3
                assert stats["processed"] == 3
                assert stats["errors"] == 0


class TestEmbeddingWorkerHealth:
    """Tests for worker health checks."""

    @pytest.mark.asyncio
    async def test_health_check(
        self,
        mock_queue,
        mock_database,
        mock_embedding_service_for_worker,
    ):
        """Should return health status."""
        worker = EmbeddingWorker(
            queue=mock_queue,
            database=mock_database,
            embedding_service=mock_embedding_service_for_worker,
        )

        health = await worker.health_check()

        assert "running" in health
        assert "queue_healthy" in health
        assert "database_healthy" in health


class TestEmbeddingWorkerClusteringIntegration:
    """Tests for clustering queue integration in EmbeddingWorker."""

    @pytest.mark.asyncio
    async def test_enqueues_to_clustering_when_enabled(
        self,
        mock_queue,
        mock_database,
        mock_repository,
        mock_embedding_service_for_worker,
        sample_documents,
    ):
        """Should enqueue to clustering queue after successful embedding."""
        mock_clustering_queue = AsyncMock(spec=ClusteringQueue)
        mock_clustering_queue.publish = AsyncMock(return_value="clust_msg_1")

        config = EmbeddingConfig(batch_size=4)

        with patch("src.embedding.worker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.clustering_enabled = True
            settings.redis_url = "redis://localhost:6379/0"
            mock_settings.return_value = settings

            worker = EmbeddingWorker(
                queue=mock_queue,
                database=mock_database,
                embedding_service=mock_embedding_service_for_worker,
                config=config,
                clustering_queue=mock_clustering_queue,
            )
            worker._repository = mock_repository

            jobs = [
                EmbeddingJob(message_id=f"msg_{i}", document_id=f"doc_{i}")
                for i in range(3)
            ]

            await worker._process_batch(jobs)

            # All 3 documents should be enqueued for clustering
            assert mock_clustering_queue.publish.call_count == 3

            # Verify the publish calls include doc_id and model type
            for call in mock_clustering_queue.publish.call_args_list:
                args = call[0]
                assert args[0].startswith("doc_")
                assert args[1] == "finbert"

    @pytest.mark.asyncio
    async def test_no_enqueue_when_clustering_disabled(
        self,
        mock_queue,
        mock_database,
        mock_repository,
        mock_embedding_service_for_worker,
        sample_documents,
    ):
        """Should not enqueue when clustering is disabled."""
        mock_clustering_queue = AsyncMock(spec=ClusteringQueue)

        config = EmbeddingConfig(batch_size=4)

        with patch("src.embedding.worker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.clustering_enabled = False
            mock_settings.return_value = settings

            worker = EmbeddingWorker(
                queue=mock_queue,
                database=mock_database,
                embedding_service=mock_embedding_service_for_worker,
                config=config,
                clustering_queue=mock_clustering_queue,
            )
            worker._repository = mock_repository

            jobs = [
                EmbeddingJob(message_id="msg_0", document_id="doc_0")
            ]

            await worker._process_batch(jobs)

            # Should NOT enqueue since clustering is disabled
            mock_clustering_queue.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_clustering_enqueue_failure_does_not_fail_embedding(
        self,
        mock_queue,
        mock_database,
        mock_repository,
        mock_embedding_service_for_worker,
        sample_documents,
    ):
        """Clustering enqueue failure should not affect embedding processing."""
        mock_clustering_queue = AsyncMock(spec=ClusteringQueue)
        mock_clustering_queue.publish = AsyncMock(
            side_effect=Exception("Redis connection lost")
        )

        config = EmbeddingConfig(batch_size=4)

        with patch("src.embedding.worker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.clustering_enabled = True
            settings.redis_url = "redis://localhost:6379/0"
            mock_settings.return_value = settings

            worker = EmbeddingWorker(
                queue=mock_queue,
                database=mock_database,
                embedding_service=mock_embedding_service_for_worker,
                config=config,
                clustering_queue=mock_clustering_queue,
            )
            worker._repository = mock_repository

            jobs = [
                EmbeddingJob(message_id="msg_0", document_id="doc_0")
            ]

            # Should NOT raise even though clustering enqueue fails
            await worker._process_batch(jobs)

            # Embedding should still be acknowledged
            mock_queue.ack.assert_called_once()

            # Embedding should still be stored
            mock_repository.update_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_enqueue_when_embedding_update_fails(
        self,
        mock_queue,
        mock_database,
        mock_embedding_service_for_worker,
        sample_documents,
    ):
        """Should not enqueue for clustering when embedding DB update fails."""
        mock_clustering_queue = AsyncMock(spec=ClusteringQueue)

        config = EmbeddingConfig(batch_size=4)

        # Repository that returns False (update failed)
        mock_repo = AsyncMock()
        doc_map = {doc.id: doc for doc in sample_documents}
        mock_repo.get_by_id = AsyncMock(side_effect=lambda id: doc_map.get(id))
        mock_repo.update_embedding = AsyncMock(return_value=False)

        with patch("src.embedding.worker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.clustering_enabled = True
            settings.redis_url = "redis://localhost:6379/0"
            mock_settings.return_value = settings

            worker = EmbeddingWorker(
                queue=mock_queue,
                database=mock_database,
                embedding_service=mock_embedding_service_for_worker,
                config=config,
                clustering_queue=mock_clustering_queue,
            )
            worker._repository = mock_repo

            jobs = [
                EmbeddingJob(message_id="msg_0", document_id="doc_0")
            ]

            await worker._process_batch(jobs)

            # Should NOT enqueue since embedding update returned False
            mock_clustering_queue.publish.assert_not_called()

    def test_worker_init_reads_clustering_enabled(self):
        """Should read clustering_enabled from settings on init."""
        with patch("src.embedding.worker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.clustering_enabled = True
            mock_settings.return_value = settings

            worker = EmbeddingWorker()
            assert worker._clustering_enabled is True

        with patch("src.embedding.worker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.clustering_enabled = False
            mock_settings.return_value = settings

            worker = EmbeddingWorker()
            assert worker._clustering_enabled is False

    @pytest.mark.asyncio
    async def test_cleanup_closes_clustering_queue(
        self,
        mock_queue,
        mock_database,
        mock_embedding_service_for_worker,
    ):
        """Should close clustering queue during cleanup."""
        mock_clustering_queue = AsyncMock(spec=ClusteringQueue)

        with patch("src.embedding.worker.get_settings") as mock_settings:
            settings = MagicMock()
            settings.clustering_enabled = True
            mock_settings.return_value = settings

            worker = EmbeddingWorker(
                queue=mock_queue,
                database=mock_database,
                embedding_service=mock_embedding_service_for_worker,
                clustering_queue=mock_clustering_queue,
            )

            await worker._cleanup()

            mock_clustering_queue.close.assert_called_once()
