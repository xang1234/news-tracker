"""Tests for EmbeddingQueue."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.embedding.config import EmbeddingConfig
from src.embedding.queue import EmbeddingJob, EmbeddingQueue


class TestEmbeddingQueueConnection:
    """Tests for queue connection management."""

    @pytest.mark.asyncio
    async def test_connect_creates_consumer_group(self, mock_redis):
        """Should create consumer group on connect."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            queue = EmbeddingQueue()
            await queue.connect()

            mock_redis.xgroup_create.assert_called_once()
            assert queue._consumer_name is not None

            await queue.close()

    @pytest.mark.asyncio
    async def test_connect_handles_existing_group(self, mock_redis):
        """Should handle BUSYGROUP error for existing consumer group."""
        import redis.asyncio as redis_module

        error = redis_module.ResponseError("BUSYGROUP Consumer Group name already exists")
        mock_redis.xgroup_create = AsyncMock(side_effect=error)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            queue = EmbeddingQueue()
            # Should not raise
            await queue.connect()
            await queue.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_redis):
        """Should work as async context manager."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                assert queue._redis is not None
                assert queue._consumer_name is not None

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, mock_redis):
        """Should close Redis connection on cleanup."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            queue = EmbeddingQueue()
            await queue.connect()
            await queue.close()

            mock_redis.close.assert_called_once()
            assert queue._redis is None


class TestEmbeddingQueuePublish:
    """Tests for publishing embedding jobs."""

    @pytest.mark.asyncio
    async def test_publish_single(self, mock_redis):
        """Should publish single document ID."""
        mock_redis.xadd = AsyncMock(return_value="msg_123")

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                msg_id = await queue.publish("doc_456")

                assert msg_id == "msg_123"
                mock_redis.xadd.assert_called_once()

                # Verify call arguments
                call_args = mock_redis.xadd.call_args
                assert call_args.kwargs["name"] == "embedding_queue"
                assert call_args.kwargs["fields"]["document_id"] == "doc_456"

    @pytest.mark.asyncio
    async def test_publish_with_priority(self, mock_redis):
        """Should publish with priority level."""
        mock_redis.xadd = AsyncMock(return_value="msg_123")

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                await queue.publish("doc_456", priority=5)

                call_args = mock_redis.xadd.call_args
                assert call_args.kwargs["fields"]["priority"] == "5"

    @pytest.mark.asyncio
    async def test_publish_batch(self, mock_redis):
        """Should publish multiple document IDs efficiently."""
        pipeline_mock = MagicMock()
        pipeline_mock.xadd = MagicMock()
        pipeline_mock.execute = AsyncMock(return_value=["msg_1", "msg_2", "msg_3"])
        mock_redis.pipeline = MagicMock(return_value=pipeline_mock)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                msg_ids = await queue.publish_batch(["doc_1", "doc_2", "doc_3"])

                assert len(msg_ids) == 3
                assert pipeline_mock.xadd.call_count == 3

    @pytest.mark.asyncio
    async def test_publish_batch_empty(self, mock_redis):
        """Should handle empty batch."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                msg_ids = await queue.publish_batch([])

                assert msg_ids == []


class TestEmbeddingQueueAcknowledge:
    """Tests for message acknowledgment."""

    @pytest.mark.asyncio
    async def test_ack(self, mock_redis):
        """Should acknowledge message."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                await queue.ack("msg_123")

                mock_redis.xack.assert_called_once_with(
                    "embedding_queue",
                    "embedding_workers",
                    "msg_123",
                )

    @pytest.mark.asyncio
    async def test_nack_moves_to_dlq(self, mock_redis):
        """Should move failed message to DLQ on nack."""
        mock_redis.xrange = AsyncMock(return_value=[
            ("msg_123", {"document_id": "doc_456", "priority": "0"})
        ])

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                await queue.nack("msg_123", "Processing error")

                # Should add to DLQ
                dlq_call = mock_redis.xadd.call_args_list[-1]
                assert "embedding_queue:dlq" in str(dlq_call)

                # Should acknowledge original
                mock_redis.xack.assert_called()


class TestEmbeddingQueueConsume:
    """Tests for consuming embedding jobs."""

    @pytest.mark.asyncio
    async def test_consume_yields_jobs(self, mock_redis):
        """Should yield EmbeddingJob objects."""
        mock_redis.xreadgroup = AsyncMock(side_effect=[
            [("embedding_queue", [
                ("msg_1", {"document_id": "doc_1", "priority": "0"}),
                ("msg_2", {"document_id": "doc_2", "priority": "5"}),
            ])],
            [],  # No more messages
        ])

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                jobs = []
                count = 0
                async for job in queue.consume(count=10, block_ms=100):
                    jobs.append(job)
                    count += 1
                    if count >= 2:
                        break

                assert len(jobs) == 2
                assert jobs[0].document_id == "doc_1"
                assert jobs[0].message_id == "msg_1"
                assert jobs[1].document_id == "doc_2"
                assert jobs[1].priority == 5


class TestEmbeddingQueueHealth:
    """Tests for queue health checks."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_redis):
        """Should return True when Redis is healthy."""
        mock_redis.ping = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                healthy = await queue.health_check()

                assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_redis):
        """Should return False when Redis is down."""
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                healthy = await queue.health_check()

                assert healthy is False

    @pytest.mark.asyncio
    async def test_get_pending_count(self, mock_redis):
        """Should return count of pending messages."""
        mock_redis.xpending = AsyncMock(return_value={"pending": 42})

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                count = await queue.get_pending_count()

                assert count == 42

    @pytest.mark.asyncio
    async def test_get_stream_length(self, mock_redis):
        """Should return stream length."""
        mock_redis.xlen = AsyncMock(return_value=100)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with EmbeddingQueue() as queue:
                length = await queue.get_stream_length()

                assert length == 100


class TestEmbeddingJob:
    """Tests for EmbeddingJob dataclass."""

    def test_embedding_job_defaults(self):
        """Should have correct defaults."""
        job = EmbeddingJob(
            message_id="msg_123",
            document_id="doc_456",
        )

        assert job.message_id == "msg_123"
        assert job.document_id == "doc_456"
        assert job.priority == 0

    def test_embedding_job_with_priority(self):
        """Should accept custom priority."""
        job = EmbeddingJob(
            message_id="msg_123",
            document_id="doc_456",
            priority=10,
        )

        assert job.priority == 10
