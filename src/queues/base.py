"""
Abstract base class for Redis Streams queue implementations.

Provides common functionality for Redis Streams-based message queues including:
- Connection lifecycle management
- Consumer group creation
- Message consumption with automatic pending message reclaim (XAUTOCLAIM)
- Acknowledgment and dead letter queue handling
- Health check and metrics

The pending reclaim mechanism uses XAUTOCLAIM (Redis 6.2+) to recover messages
that were delivered to a consumer but never acknowledged - typically due to
worker crashes or network failures. This ensures at-least-once delivery.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, TypeVar

import redis.asyncio as redis

from src.observability.metrics import get_metrics
from src.queues.config import QueueConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class StreamConfig:
    """
    Configuration for a Redis Stream.

    Attributes:
        stream_name: Name of the Redis stream
        consumer_group: Name of the consumer group
        dlq_stream_name: Name of the dead letter queue stream
        max_stream_length: Maximum stream length before trimming
    """

    stream_name: str
    consumer_group: str
    dlq_stream_name: str
    max_stream_length: int = 50_000


class BaseRedisQueue(ABC, Generic[T]):
    """
    Abstract base class for Redis Streams queue implementations.

    Provides common Redis connection lifecycle, message consumption with
    automatic pending message reclaim, and acknowledgment handling.

    Subclasses must implement:
        - _parse_job(): Convert Redis message fields to job type T
        - _get_stream_config(): Return StreamConfig for this queue
        - _get_consumer_prefix(): Return prefix for consumer name generation

    Type Parameters:
        T: The job/message type yielded by consume()

    Usage:
        class MyQueue(BaseRedisQueue[MyJob]):
            def _parse_job(self, message_id, fields) -> MyJob:
                return MyJob(message_id=message_id, data=fields["data"])

            def _get_stream_config(self) -> StreamConfig:
                return StreamConfig(
                    stream_name="my_stream",
                    consumer_group="my_workers",
                    dlq_stream_name="my_stream:dlq",
                )

            def _get_consumer_prefix(self) -> str:
                return "my_worker"

        async with MyQueue() as queue:
            async for job in queue.consume():
                process(job)
                await queue.ack(job.message_id)
    """

    def __init__(
        self,
        redis_url: str,
        queue_config: QueueConfig | None = None,
    ):
        """
        Initialize the base queue.

        Args:
            redis_url: Redis connection URL
            queue_config: Configuration for reclaim behavior
        """
        self._redis_url = redis_url
        self._queue_config = queue_config or QueueConfig()

        self._redis: redis.Redis | None = None
        self._consumer_name: str | None = None
        self._stream_config: StreamConfig | None = None

    @abstractmethod
    def _parse_job(self, message_id: str, fields: dict[str, str]) -> T:
        """
        Parse a Redis message into a job object.

        Args:
            message_id: Redis stream message ID
            fields: Message fields from Redis

        Returns:
            Parsed job object of type T
        """
        ...

    @abstractmethod
    def _get_stream_config(self) -> StreamConfig:
        """
        Get the stream configuration for this queue.

        Returns:
            StreamConfig with stream names and settings
        """
        ...

    @abstractmethod
    def _get_consumer_prefix(self) -> str:
        """
        Get the prefix for consumer name generation.

        Returns:
            String prefix like "worker", "emb_worker", etc.
        """
        ...

    @abstractmethod
    def _set_job_retry_count(self, job: T, retry_count: int) -> None:
        """
        Set the retry count on a job object.

        Args:
            job: The job object to modify
            retry_count: Number of previous delivery attempts
        """
        ...

    async def connect(self) -> None:
        """Establish Redis connection and ensure stream/group exist."""
        self._redis = redis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

        # Cache stream config
        self._stream_config = self._get_stream_config()

        # Generate unique consumer name
        prefix = self._get_consumer_prefix()
        self._consumer_name = f"{prefix}_{uuid.uuid4().hex[:8]}"

        # Create consumer group if it doesn't exist
        try:
            await self._redis.xgroup_create(
                name=self._stream_config.stream_name,
                groupname=self._stream_config.consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(
                f"Created consumer group '{self._stream_config.consumer_group}' "
                f"for stream '{self._stream_config.stream_name}'"
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            # Group already exists - this is fine

        logger.info(
            f"Connected to Redis, consumer={self._consumer_name}, "
            f"stream={self._stream_config.stream_name}"
        )

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info(
                f"Redis connection closed for stream "
                f"{self._stream_config.stream_name if self._stream_config else 'unknown'}"
            )

    async def __aenter__(self) -> "BaseRedisQueue[T]":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def redis(self) -> redis.Redis:
        """Get Redis client, raising if not connected."""
        if self._redis is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")
        return self._redis

    @property
    def stream_config(self) -> StreamConfig:
        """Get stream configuration, raising if not connected."""
        if self._stream_config is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._stream_config

    async def consume(
        self,
        count: int = 10,
        block_ms: int = 5000,
    ) -> AsyncIterator[T]:
        """
        Consume messages from the queue.

        Uses XAUTOCLAIM to first reclaim idle pending messages, then reads
        new messages with XREADGROUP. This ensures:
        1. Messages orphaned by crashed workers are recovered
        2. Messages exceeding max delivery attempts go to DLQ
        3. New messages are processed after pending recovery

        Args:
            count: Maximum number of messages to fetch per iteration
            block_ms: How long to block waiting for new messages (milliseconds)

        Yields:
            Job objects of type T with message_id for acknowledgment
        """
        if self._consumer_name is None:
            raise RuntimeError("Not connected. Call connect() first.")

        while True:
            try:
                # Step 1: Reclaim idle pending messages FIRST (fairness + recovery)
                async for job in self._reclaim_pending(count):
                    yield job

                # Step 2: Read new messages (existing behavior)
                messages = await self.redis.xreadgroup(
                    groupname=self.stream_config.consumer_group,
                    consumername=self._consumer_name,
                    streams={self.stream_config.stream_name: ">"},
                    count=count,
                    block=block_ms,
                )

                if not messages:
                    # No messages within block time, continue waiting
                    continue

                # messages is a list of [stream_name, [(id, fields), ...]]
                for stream_name, msg_list in messages:
                    for msg_id, fields in msg_list:
                        try:
                            job = self._parse_job(msg_id, fields)
                            # New messages have retry_count = 0
                            self._set_job_retry_count(job, 0)
                            yield job
                        except Exception as e:
                            logger.error(f"Failed to parse message {msg_id}: {e}")
                            # Move to DLQ and acknowledge original
                            await self._move_to_dlq(msg_id, fields, str(e))
                            await self.ack(msg_id)

            except asyncio.CancelledError:
                logger.info("Consumer cancelled, stopping gracefully")
                break
            except Exception as e:
                logger.error(f"Error consuming messages: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def _reclaim_pending(self, count: int) -> AsyncIterator[T]:
        """
        Reclaim idle pending messages using XAUTOCLAIM.

        Claims messages that have been pending longer than idle_timeout_ms
        and either yields them for reprocessing or moves them to DLQ if
        they've exceeded max_delivery_attempts.

        Args:
            count: Maximum number of messages to reclaim

        Yields:
            Job objects of type T for messages that should be reprocessed
        """
        metrics = get_metrics()

        try:
            # XAUTOCLAIM returns: [next_start_id, [(msg_id, fields), ...], [deleted_ids]]
            result = await self.redis.xautoclaim(
                name=self.stream_config.stream_name,
                groupname=self.stream_config.consumer_group,
                consumername=self._consumer_name,
                min_idle_time=self._queue_config.idle_timeout_ms,
                start_id="0-0",
                count=count,
            )

            if not result or not result[1]:
                return

            claimed_messages = result[1]  # List of (msg_id, fields)

            if claimed_messages:
                logger.info(
                    f"Reclaimed {len(claimed_messages)} pending messages "
                    f"from {self.stream_config.stream_name}"
                )

            # Get delivery counts for all claimed messages
            delivery_counts = await self._get_delivery_counts(
                [msg_id for msg_id, _ in claimed_messages]
            )

            for msg_id, fields in claimed_messages:
                delivery_count = delivery_counts.get(msg_id, 1)

                # Check if message has exceeded max delivery attempts
                if delivery_count > self._queue_config.max_delivery_attempts:
                    logger.warning(
                        f"Message {msg_id} exceeded max delivery attempts "
                        f"({delivery_count}/{self._queue_config.max_delivery_attempts}), "
                        f"moving to DLQ"
                    )
                    await self._move_to_dlq(
                        msg_id, fields, "max_retries_exceeded"
                    )
                    await self.ack(msg_id)

                    # Record metric
                    metrics.dlq_max_retries.labels(
                        queue=self.stream_config.stream_name
                    ).inc()
                    continue

                try:
                    job = self._parse_job(msg_id, fields)
                    # Set retry_count = delivery_count - 1 (current delivery doesn't count)
                    self._set_job_retry_count(job, delivery_count - 1)

                    # Record reclaim metric
                    metrics.pending_reclaimed.labels(
                        queue=self.stream_config.stream_name
                    ).inc()

                    yield job
                except Exception as e:
                    logger.error(f"Failed to parse reclaimed message {msg_id}: {e}")
                    await self._move_to_dlq(msg_id, fields, str(e))
                    await self.ack(msg_id)

        except redis.ResponseError as e:
            # XAUTOCLAIM requires Redis 6.2+
            if "unknown command" in str(e).lower():
                logger.warning(
                    "XAUTOCLAIM not available (requires Redis 6.2+), "
                    "skipping pending reclaim"
                )
            else:
                logger.error(f"Error reclaiming pending messages: {e}")
        except Exception as e:
            logger.error(f"Error reclaiming pending messages: {e}")

    async def _get_delivery_counts(
        self, message_ids: list[str]
    ) -> dict[str, int]:
        """
        Get delivery counts for a list of message IDs.

        Uses XPENDING with range to get detailed info including delivery count.

        Args:
            message_ids: List of message IDs to check

        Returns:
            Dict mapping message_id to delivery count
        """
        if not message_ids:
            return {}

        try:
            # XPENDING with range returns detailed info
            # Format: [[msg_id, consumer, idle_time, delivery_count], ...]
            pending_info = await self.redis.xpending_range(
                name=self.stream_config.stream_name,
                groupname=self.stream_config.consumer_group,
                min="-",
                max="+",
                count=len(message_ids) * 2,  # Get extra in case of gaps
            )

            delivery_counts = {}
            for info in pending_info:
                msg_id = info["message_id"]
                if msg_id in message_ids:
                    delivery_counts[msg_id] = info["times_delivered"]

            return delivery_counts

        except Exception as e:
            logger.error(f"Error getting delivery counts: {e}")
            # Return default count of 1 for all messages
            return {msg_id: 1 for msg_id in message_ids}

    async def ack(self, message_id: str) -> None:
        """
        Acknowledge successful processing of a message.

        Args:
            message_id: The message ID to acknowledge
        """
        await self.redis.xack(
            self.stream_config.stream_name,
            self.stream_config.consumer_group,
            message_id,
        )
        logger.debug(f"Acknowledged message {message_id}")

    async def nack(
        self,
        message_id: str,
        error: str | None = None,
    ) -> None:
        """
        Negative acknowledgment - move message to dead letter queue.

        Args:
            message_id: The message ID that failed
            error: Optional error message for debugging
        """
        # Get the original message data
        messages = await self.redis.xrange(
            self.stream_config.stream_name,
            min=message_id,
            max=message_id,
        )

        if messages:
            _, fields = messages[0]
            await self._move_to_dlq(message_id, fields, error)

        # Acknowledge to remove from pending
        await self.ack(message_id)

    async def _move_to_dlq(
        self,
        original_id: str,
        fields: dict[str, str],
        error: str | None,
    ) -> None:
        """Move a failed message to the dead letter queue."""
        dlq_fields = {
            **fields,
            "original_id": original_id,
            "error": error or "unknown",
            "failed_at": str(time.time()),
        }

        await self.redis.xadd(
            self.stream_config.dlq_stream_name,
            dlq_fields,
            maxlen=10000,  # Keep last 10k failed messages
        )
        logger.warning(f"Moved message {original_id} to DLQ: {error}")

    async def get_pending_count(self) -> int:
        """Get count of pending (unacknowledged) messages."""
        try:
            info = await self.redis.xpending(
                self.stream_config.stream_name,
                self.stream_config.consumer_group,
            )
            return info["pending"] if info else 0
        except Exception:
            return 0

    async def get_stream_length(self) -> int:
        """Get total number of messages in the stream."""
        return await self.redis.xlen(self.stream_config.stream_name)

    async def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False
