"""
Tests for Redis Streams pending message reclaim functionality.

Tests verify that the BaseRedisQueue correctly:
- Reclaims idle messages after timeout
- Respects max delivery attempts and moves to DLQ
- Processes pending messages before new messages
- Handles Redis errors gracefully
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.queues import BaseRedisQueue, QueueConfig, StreamConfig


@dataclass
class MockJob:
    """Simple mock job type for testing."""

    message_id: str
    data: str
    retry_count: int = 0


class MockQueue(BaseRedisQueue[MockJob]):
    """Concrete test queue implementation."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_config: QueueConfig | None = None,
    ):
        super().__init__(redis_url=redis_url, queue_config=queue_config)

    def _get_stream_config(self) -> StreamConfig:
        return StreamConfig(
            stream_name="test_stream",
            consumer_group="test_workers",
            dlq_stream_name="test_stream:dlq",
            max_stream_length=1000,
        )

    def _get_consumer_prefix(self) -> str:
        return "test_worker"

    def _parse_job(self, message_id: str, fields: dict[str, str]) -> MockJob:
        return MockJob(
            message_id=message_id,
            data=fields["data"],
        )

    def _set_job_retry_count(self, job: MockJob, retry_count: int) -> None:
        job.retry_count = retry_count


class MockQueueConfig:
    """Tests for QueueConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QueueConfig()
        assert config.idle_timeout_ms == 30_000
        assert config.max_delivery_attempts == 3
        assert config.reclaim_batch_size == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = QueueConfig(
            idle_timeout_ms=60_000,
            max_delivery_attempts=5,
            reclaim_batch_size=20,
        )
        assert config.idle_timeout_ms == 60_000
        assert config.max_delivery_attempts == 5
        assert config.reclaim_batch_size == 20


class TestStreamConfig:
    """Tests for StreamConfig dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        config = StreamConfig(
            stream_name="my_stream",
            consumer_group="my_workers",
            dlq_stream_name="my_stream:dlq",
        )
        assert config.stream_name == "my_stream"
        assert config.consumer_group == "my_workers"
        assert config.dlq_stream_name == "my_stream:dlq"
        assert config.max_stream_length == 50_000  # default

    def test_custom_max_length(self):
        """Test custom max stream length."""
        config = StreamConfig(
            stream_name="my_stream",
            consumer_group="my_workers",
            dlq_stream_name="my_stream:dlq",
            max_stream_length=100_000,
        )
        assert config.max_stream_length == 100_000


class TestBaseRedisQueueReclaim:
    """Tests for pending message reclaim functionality."""

    @pytest.fixture
    def queue_config(self):
        """Create test queue configuration with short timeout."""
        return QueueConfig(
            idle_timeout_ms=1_000,  # 1 second for testing
            max_delivery_attempts=3,
            reclaim_batch_size=10,
        )

    @pytest.fixture
    def queue(self, queue_config):
        """Create test queue with mocked Redis."""
        q = MockQueue(queue_config=queue_config)
        q._redis = AsyncMock()
        q._consumer_name = "test_worker_abc123"
        q._stream_config = q._get_stream_config()
        return q

    @pytest.mark.asyncio
    async def test_reclaim_idle_messages(self, queue):
        """Test that idle messages are reclaimed."""
        # Setup: XAUTOCLAIM returns one claimed message
        queue._redis.xautoclaim.return_value = [
            "0-0",  # next start ID
            [("msg-1", {"data": "test_data"})],  # claimed messages
            [],  # deleted IDs
        ]
        # XPENDING_RANGE returns delivery count = 2
        queue._redis.xpending_range.return_value = [
            {"message_id": "msg-1", "consumer": "old_worker", "times_delivered": 2}
        ]

        # Execute: collect reclaimed messages
        reclaimed = []
        async for job in queue._reclaim_pending(count=10):
            reclaimed.append(job)

        # Verify: message was reclaimed with correct retry count
        assert len(reclaimed) == 1
        assert reclaimed[0].message_id == "msg-1"
        assert reclaimed[0].data == "test_data"
        assert reclaimed[0].retry_count == 1  # delivery_count - 1

        # Verify XAUTOCLAIM was called with correct parameters
        queue._redis.xautoclaim.assert_called_once_with(
            name="test_stream",
            groupname="test_workers",
            consumername="test_worker_abc123",
            min_idle_time=1_000,
            start_id="0-0",
            count=10,
        )

    @pytest.mark.asyncio
    async def test_max_delivery_attempts_moves_to_dlq(self, queue):
        """Test that messages exceeding max delivery attempts are moved to DLQ."""
        # Setup: XAUTOCLAIM returns message that exceeded max retries
        queue._redis.xautoclaim.return_value = [
            "0-0",
            [("msg-1", {"data": "failing_data"})],
            [],
        ]
        # Delivery count = 4, exceeds max_delivery_attempts = 3
        queue._redis.xpending_range.return_value = [
            {"message_id": "msg-1", "consumer": "old_worker", "times_delivered": 4}
        ]

        # Execute: collect reclaimed messages
        reclaimed = []
        async for job in queue._reclaim_pending(count=10):
            reclaimed.append(job)

        # Verify: no messages yielded (went to DLQ instead)
        assert len(reclaimed) == 0

        # Verify message was moved to DLQ
        queue._redis.xadd.assert_called_once()
        call_args = queue._redis.xadd.call_args
        # xadd is called as xadd(dlq_stream_name, fields, maxlen=...)
        dlq_stream = call_args[0][0]
        dlq_fields = call_args[0][1]
        assert dlq_stream == "test_stream:dlq"
        assert dlq_fields["error"] == "max_retries_exceeded"

        # Verify message was ACKed after DLQ
        queue._redis.xack.assert_called_once_with(
            "test_stream", "test_workers", "msg-1"
        )

    @pytest.mark.asyncio
    async def test_processes_pending_before_new(self, queue):
        """Test that pending messages are processed before new messages."""
        # Track call order
        call_order = []

        async def mock_xautoclaim(*args, **kwargs):
            call_order.append("xautoclaim")
            return ["0-0", [("pending-1", {"data": "pending"})], []]

        async def mock_xreadgroup(*args, **kwargs):
            call_order.append("xreadgroup")
            # Return empty to stop the loop
            return []

        queue._redis.xautoclaim = mock_xautoclaim
        queue._redis.xreadgroup = mock_xreadgroup
        queue._redis.xpending_range.return_value = [
            {"message_id": "pending-1", "consumer": "old", "times_delivered": 1}
        ]

        # Execute: consume with a limit
        consumed = []
        count = 0
        async for job in queue.consume(count=10, block_ms=100):
            consumed.append(job)
            count += 1
            if count >= 1:
                break

        # Verify: xautoclaim called before xreadgroup
        assert call_order[0] == "xautoclaim"
        # Verify we got the pending message
        assert len(consumed) == 1
        assert consumed[0].data == "pending"

    @pytest.mark.asyncio
    async def test_handles_xautoclaim_not_available(self, queue):
        """Test graceful handling when Redis version < 6.2."""
        from redis import ResponseError

        # Setup: XAUTOCLAIM command not recognized
        queue._redis.xautoclaim.side_effect = ResponseError(
            "ERR unknown command 'XAUTOCLAIM'"
        )

        # Execute: should not raise, just skip reclaim
        reclaimed = []
        async for job in queue._reclaim_pending(count=10):
            reclaimed.append(job)

        # Verify: no messages reclaimed, no error raised
        assert len(reclaimed) == 0

    @pytest.mark.asyncio
    async def test_handles_redis_error_gracefully(self, queue):
        """Test graceful handling of Redis errors during reclaim."""
        from redis import ResponseError

        # Setup: Redis connection error
        queue._redis.xautoclaim.side_effect = ResponseError("LOADING Redis is loading")

        # Execute: should not raise, just skip reclaim
        reclaimed = []
        async for job in queue._reclaim_pending(count=10):
            reclaimed.append(job)

        # Verify: no messages reclaimed, no error raised
        assert len(reclaimed) == 0

    @pytest.mark.asyncio
    async def test_handles_parse_error_in_reclaimed_message(self, queue):
        """Test that parse errors in reclaimed messages move to DLQ."""
        # Setup: XAUTOCLAIM returns message with invalid data
        queue._redis.xautoclaim.return_value = [
            "0-0",
            [("msg-1", {"invalid_field": "no_data_field"})],  # Missing 'data' field
            [],
        ]
        queue._redis.xpending_range.return_value = [
            {"message_id": "msg-1", "consumer": "old_worker", "times_delivered": 1}
        ]

        # Create a queue that fails to parse
        class FailingQueue(MockQueue):
            def _parse_job(self, message_id, fields):
                raise KeyError("data")

        failing_queue = FailingQueue(queue_config=QueueConfig())
        failing_queue._redis = queue._redis
        failing_queue._consumer_name = "test_worker_abc123"
        failing_queue._stream_config = failing_queue._get_stream_config()

        # Execute
        reclaimed = []
        async for job in failing_queue._reclaim_pending(count=10):
            reclaimed.append(job)

        # Verify: no messages yielded
        assert len(reclaimed) == 0

        # Verify: moved to DLQ
        queue._redis.xadd.assert_called_once()
        call_args = queue._redis.xadd.call_args
        dlq_stream = call_args[0][0]
        assert dlq_stream == "test_stream:dlq"

    @pytest.mark.asyncio
    async def test_no_pending_messages(self, queue):
        """Test behavior when no pending messages exist."""
        # Setup: XAUTOCLAIM returns empty
        queue._redis.xautoclaim.return_value = ["0-0", [], []]

        # Execute
        reclaimed = []
        async for job in queue._reclaim_pending(count=10):
            reclaimed.append(job)

        # Verify: no messages, no errors
        assert len(reclaimed) == 0
        queue._redis.xpending_range.assert_not_called()

    @pytest.mark.asyncio
    async def test_delivery_count_defaults_to_one(self, queue):
        """Test that missing delivery count defaults to 1."""
        # Setup: XAUTOCLAIM returns message but xpending_range returns empty
        queue._redis.xautoclaim.return_value = [
            "0-0",
            [("msg-1", {"data": "test"})],
            [],
        ]
        queue._redis.xpending_range.return_value = []  # No info returned

        # Execute
        reclaimed = []
        async for job in queue._reclaim_pending(count=10):
            reclaimed.append(job)

        # Verify: message reclaimed with default retry count
        assert len(reclaimed) == 1
        assert reclaimed[0].retry_count == 0  # delivery_count(1) - 1 = 0


class TestBaseRedisQueueLifecycle:
    """Tests for queue lifecycle methods."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        with patch("src.queues.base.redis") as mock_redis_module:
            mock_redis = AsyncMock()
            mock_redis_module.from_url.return_value = mock_redis
            mock_redis.xgroup_create = AsyncMock()
            mock_redis.close = AsyncMock()

            async with MockQueue() as queue:
                assert queue._redis is not None
                assert queue._consumer_name is not None

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when Redis is healthy."""
        queue = MockQueue()
        queue._redis = AsyncMock()
        queue._redis.ping = AsyncMock(return_value=True)
        queue._stream_config = queue._get_stream_config()

        assert await queue.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check when Redis is unhealthy."""
        from redis import ConnectionError

        queue = MockQueue()
        queue._redis = AsyncMock()
        queue._redis.ping = AsyncMock(side_effect=ConnectionError())
        queue._stream_config = queue._get_stream_config()

        assert await queue.health_check() is False

    @pytest.mark.asyncio
    async def test_get_pending_count(self):
        """Test getting pending message count."""
        queue = MockQueue()
        queue._redis = AsyncMock()
        queue._redis.xpending.return_value = {"pending": 42}
        queue._stream_config = queue._get_stream_config()

        count = await queue.get_pending_count()
        assert count == 42

    @pytest.mark.asyncio
    async def test_get_stream_length(self):
        """Test getting stream length."""
        queue = MockQueue()
        queue._redis = AsyncMock()
        queue._redis.xlen.return_value = 100
        queue._stream_config = queue._get_stream_config()

        length = await queue.get_stream_length()
        assert length == 100


class TestBaseRedisQueueAck:
    """Tests for acknowledgment methods."""

    @pytest.fixture
    def queue(self):
        """Create test queue with mocked Redis."""
        q = MockQueue()
        q._redis = AsyncMock()
        q._stream_config = q._get_stream_config()
        return q

    @pytest.mark.asyncio
    async def test_ack(self, queue):
        """Test message acknowledgment."""
        await queue.ack("msg-123")

        queue._redis.xack.assert_called_once_with(
            "test_stream", "test_workers", "msg-123"
        )

    @pytest.mark.asyncio
    async def test_nack_moves_to_dlq(self, queue):
        """Test negative acknowledgment moves message to DLQ."""
        # Setup: xrange returns the original message
        queue._redis.xrange.return_value = [
            ("msg-123", {"data": "failed_data"})
        ]

        await queue.nack("msg-123", error="Processing failed")

        # Verify: message moved to DLQ
        queue._redis.xadd.assert_called_once()
        call_args = queue._redis.xadd.call_args
        dlq_stream = call_args[0][0]
        dlq_fields = call_args[0][1]
        assert dlq_stream == "test_stream:dlq"
        assert dlq_fields["error"] == "Processing failed"
        assert dlq_fields["original_id"] == "msg-123"

        # Verify: original message ACKed
        queue._redis.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_nack_missing_message(self, queue):
        """Test nack when original message not found."""
        # Setup: xrange returns empty (message was already deleted)
        queue._redis.xrange.return_value = []

        await queue.nack("msg-123", error="Processing failed")

        # Verify: no DLQ write, but still ACKed
        queue._redis.xadd.assert_not_called()
        queue._redis.xack.assert_called_once()
