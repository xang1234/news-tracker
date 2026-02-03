"""
Redis Streams wrapper for embedding job queue.

Provides a message queue abstraction for embedding jobs with:
- Consumer groups for horizontal scaling
- At-least-once delivery semantics
- Dead letter queue for failed jobs
- Batch publishing for efficiency
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import TracebackType

import redis.asyncio as redis

from src.config.settings import get_settings
from src.embedding.config import EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingJob:
    """
    An embedding job from the queue.

    Attributes:
        message_id: Redis stream message ID for acknowledgment
        document_id: ID of the document to embed
        priority: Optional priority level (higher = more urgent)
    """

    message_id: str
    document_id: str
    priority: int = 0


class EmbeddingQueue:
    """
    Redis Streams wrapper for the embedding job queue.

    Streams:
        - 'embedding_queue': Main stream for embedding job IDs
        - 'embedding_queue:dlq': Dead letter queue for failed jobs

    Consumer Groups:
        - 'embedding_workers': Workers that process embedding jobs

    Usage:
        async with EmbeddingQueue() as queue:
            await queue.publish("doc_123")

            async for job in queue.consume():
                process(job.document_id)
                await queue.ack(job.message_id)
    """

    def __init__(
        self,
        redis_url: str | None = None,
        config: EmbeddingConfig | None = None,
    ):
        """
        Initialize the embedding queue.

        Args:
            redis_url: Redis connection URL (uses settings if None)
            config: Embedding configuration (uses defaults if None)
        """
        settings = get_settings()
        self._config = config or EmbeddingConfig()

        self._redis_url = redis_url or str(settings.redis_url)
        self._stream_name = self._config.stream_name
        self._consumer_group = self._config.consumer_group
        self._max_stream_length = self._config.max_stream_length
        self._dlq_stream = self._config.dlq_stream_name

        self._redis: redis.Redis | None = None
        self._consumer_name: str | None = None

    async def connect(self) -> None:
        """Establish Redis connection and ensure stream/group exist."""
        self._redis = redis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

        # Generate unique consumer name for this instance
        import uuid

        self._consumer_name = f"emb_worker_{uuid.uuid4().hex[:8]}"

        # Create consumer group if it doesn't exist
        try:
            await self._redis.xgroup_create(
                name=self._stream_name,
                groupname=self._consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(
                f"Created consumer group '{self._consumer_group}' "
                f"for stream '{self._stream_name}'"
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            # Group already exists - this is fine

        logger.info(
            f"Connected to Redis, consumer={self._consumer_name}, "
            f"stream={self._stream_name}"
        )

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Embedding queue Redis connection closed")

    async def __aenter__(self) -> "EmbeddingQueue":
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

    async def publish(self, document_id: str, priority: int = 0) -> str:
        """
        Add a document ID to the embedding queue.

        Args:
            document_id: ID of the document to embed
            priority: Optional priority (higher = more urgent)

        Returns:
            Message ID assigned by Redis
        """
        message_id = await self.redis.xadd(
            name=self._stream_name,
            fields={
                "document_id": document_id,
                "priority": str(priority),
                "queued_at": str(time.time()),
            },
            maxlen=self._max_stream_length,
            approximate=True,
        )

        logger.debug(f"Published embedding job for {document_id}, id={message_id}")
        return str(message_id)

    async def publish_batch(
        self,
        document_ids: list[str],
        priority: int = 0,
    ) -> list[str]:
        """
        Publish multiple document IDs efficiently using pipeline.

        Args:
            document_ids: List of document IDs to embed
            priority: Priority for all jobs

        Returns:
            List of message IDs
        """
        if not document_ids:
            return []

        queued_at = str(time.time())
        pipe = self.redis.pipeline()

        for doc_id in document_ids:
            pipe.xadd(
                name=self._stream_name,
                fields={
                    "document_id": doc_id,
                    "priority": str(priority),
                    "queued_at": queued_at,
                },
                maxlen=self._max_stream_length,
                approximate=True,
            )

        results = await pipe.execute()
        message_ids = [str(r) for r in results]

        logger.info(f"Published batch of {len(document_ids)} embedding jobs")
        return message_ids

    async def consume(
        self,
        count: int = 10,
        block_ms: int = 5000,
    ) -> AsyncIterator[EmbeddingJob]:
        """
        Consume embedding jobs from the queue.

        Uses XREADGROUP for consumer group semantics. Jobs must be
        acknowledged with ack() after successful processing.

        Args:
            count: Maximum number of jobs to fetch per iteration
            block_ms: How long to block waiting for jobs (milliseconds)

        Yields:
            EmbeddingJob with document_id and message_id for acknowledgment
        """
        if self._consumer_name is None:
            raise RuntimeError("Not connected. Call connect() first.")

        while True:
            try:
                # Read new messages (">") from the consumer group
                messages = await self.redis.xreadgroup(
                    groupname=self._consumer_group,
                    consumername=self._consumer_name,
                    streams={self._stream_name: ">"},
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
                            yield EmbeddingJob(
                                message_id=msg_id,
                                document_id=fields["document_id"],
                                priority=int(fields.get("priority", 0)),
                            )
                        except Exception as e:
                            logger.error(f"Failed to parse job {msg_id}: {e}")
                            # Move to DLQ and acknowledge original
                            await self._move_to_dlq(msg_id, fields, str(e))
                            await self.ack(msg_id)

            except asyncio.CancelledError:
                logger.info("Consumer cancelled, stopping gracefully")
                break
            except Exception as e:
                logger.error(f"Error consuming embedding jobs: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def ack(self, message_id: str) -> None:
        """
        Acknowledge successful processing of a job.

        Args:
            message_id: The message ID to acknowledge
        """
        await self.redis.xack(
            self._stream_name,
            self._consumer_group,
            message_id,
        )
        logger.debug(f"Acknowledged embedding job {message_id}")

    async def nack(
        self,
        message_id: str,
        error: str | None = None,
    ) -> None:
        """
        Negative acknowledgment - move job to dead letter queue.

        Args:
            message_id: The message ID that failed
            error: Optional error message for debugging
        """
        # Get the original message data
        messages = await self.redis.xrange(
            self._stream_name,
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
        """Move a failed job to the dead letter queue."""
        dlq_fields = {
            **fields,
            "original_id": original_id,
            "error": error or "unknown",
            "failed_at": str(time.time()),
        }

        await self.redis.xadd(
            self._dlq_stream,
            dlq_fields,
            maxlen=10000,  # Keep last 10k failed jobs
        )
        logger.warning(f"Moved embedding job {original_id} to DLQ: {error}")

    async def get_pending_count(self) -> int:
        """Get count of pending (unacknowledged) jobs."""
        info = await self.redis.xpending(
            self._stream_name,
            self._consumer_group,
        )
        return info["pending"] if info else 0

    async def get_stream_length(self) -> int:
        """Get total number of jobs in the stream."""
        return await self.redis.xlen(self._stream_name)

    async def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False
