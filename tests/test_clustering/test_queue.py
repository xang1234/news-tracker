"""Tests for ClusteringQueue."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clustering.config import ClusteringConfig
from src.clustering.queue import ClusteringJob, ClusteringQueue


@pytest.fixture
def clustering_config():
    """Create test clustering config."""
    return ClusteringConfig(
        stream_name="test_clustering_queue",
        consumer_group="test_clustering_workers",
        dlq_stream_name="test_clustering_queue:dlq",
        max_stream_length=1000,
    )


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    r = AsyncMock()
    r.xadd = AsyncMock(return_value="msg_001")
    r.xreadgroup = AsyncMock(return_value=[])
    r.xack = AsyncMock()
    r.xrange = AsyncMock(return_value=[])
    r.xgroup_create = AsyncMock()
    r.xlen = AsyncMock(return_value=0)
    r.xpending = AsyncMock(return_value={"pending": 0})
    r.ping = AsyncMock()
    r.close = AsyncMock()

    pipe = AsyncMock()
    pipe.xadd = MagicMock()
    pipe.execute = AsyncMock(return_value=["msg_001", "msg_002", "msg_003"])
    r.pipeline = MagicMock(return_value=pipe)

    return r


class TestClusteringJob:
    """Tests for the ClusteringJob dataclass."""

    def test_job_creation(self):
        """Should create a job with required fields."""
        job = ClusteringJob(
            document_id="doc_1",
            embedding_model="finbert",
            message_id="msg_1",
        )
        assert job.document_id == "doc_1"
        assert job.embedding_model == "finbert"
        assert job.message_id == "msg_1"
        assert job.retry_count == 0

    def test_job_with_retry_count(self):
        """Should accept custom retry count."""
        job = ClusteringJob(
            document_id="doc_1",
            embedding_model="minilm",
            message_id="msg_1",
            retry_count=2,
        )
        assert job.retry_count == 2


class TestClusteringQueueConfig:
    """Tests for queue initialization and configuration."""

    def test_default_config(self):
        """Should initialize with default ClusteringConfig."""
        queue = ClusteringQueue()
        assert queue._config.stream_name == "clustering_queue"
        assert queue._config.consumer_group == "clustering_workers"

    def test_custom_config(self, clustering_config):
        """Should accept custom config."""
        queue = ClusteringQueue(config=clustering_config)
        assert queue._config.stream_name == "test_clustering_queue"

    def test_stream_config(self, clustering_config):
        """Should return correct StreamConfig."""
        queue = ClusteringQueue(config=clustering_config)
        sc = queue._get_stream_config()
        assert sc.stream_name == "test_clustering_queue"
        assert sc.consumer_group == "test_clustering_workers"
        assert sc.dlq_stream_name == "test_clustering_queue:dlq"
        assert sc.max_stream_length == 1000

    def test_consumer_prefix(self, clustering_config):
        """Should return clustering_worker prefix."""
        queue = ClusteringQueue(config=clustering_config)
        assert queue._get_consumer_prefix() == "clustering_worker"


class TestClusteringQueueParsing:
    """Tests for message parsing."""

    def test_parse_job(self, clustering_config):
        """Should parse Redis message into ClusteringJob."""
        queue = ClusteringQueue(config=clustering_config)
        job = queue._parse_job("msg_123", {
            "document_id": "doc_abc",
            "embedding_model": "finbert",
            "queued_at": "1234567890.0",
        })
        assert job.document_id == "doc_abc"
        assert job.embedding_model == "finbert"
        assert job.message_id == "msg_123"

    def test_parse_job_default_model(self, clustering_config):
        """Should default to finbert when embedding_model missing."""
        queue = ClusteringQueue(config=clustering_config)
        job = queue._parse_job("msg_123", {
            "document_id": "doc_abc",
        })
        assert job.embedding_model == "finbert"

    def test_set_retry_count(self, clustering_config):
        """Should set retry count on job."""
        queue = ClusteringQueue(config=clustering_config)
        job = ClusteringJob(
            document_id="doc_1",
            embedding_model="finbert",
            message_id="msg_1",
        )
        queue._set_job_retry_count(job, 3)
        assert job.retry_count == 3


class TestClusteringQueuePublish:
    """Tests for publishing clustering jobs."""

    @pytest.mark.asyncio
    async def test_publish_single(self, clustering_config, mock_redis):
        """Should publish a single clustering job."""
        queue = ClusteringQueue(config=clustering_config)
        queue._redis = mock_redis
        queue._stream_config = queue._get_stream_config()

        msg_id = await queue.publish("doc_1", "finbert")

        assert msg_id == "msg_001"
        mock_redis.xadd.assert_called_once()
        call_kwargs = mock_redis.xadd.call_args
        fields = call_kwargs.kwargs["fields"]
        assert fields["document_id"] == "doc_1"
        assert fields["embedding_model"] == "finbert"
        assert "queued_at" in fields

    @pytest.mark.asyncio
    async def test_publish_batch(self, clustering_config, mock_redis):
        """Should publish multiple clustering jobs via pipeline."""
        queue = ClusteringQueue(config=clustering_config)
        queue._redis = mock_redis
        queue._stream_config = queue._get_stream_config()

        msg_ids = await queue.publish_batch(
            ["doc_1", "doc_2", "doc_3"], "finbert"
        )

        assert len(msg_ids) == 3
        pipe = mock_redis.pipeline.return_value
        assert pipe.xadd.call_count == 3
        pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_batch_empty(self, clustering_config, mock_redis):
        """Should return empty list for empty batch."""
        queue = ClusteringQueue(config=clustering_config)
        queue._redis = mock_redis
        queue._stream_config = queue._get_stream_config()

        msg_ids = await queue.publish_batch([], "finbert")

        assert msg_ids == []
        mock_redis.pipeline.assert_not_called()
