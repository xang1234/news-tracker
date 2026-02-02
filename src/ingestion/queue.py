"""
Redis Streams wrapper for document ingestion pipeline.

Provides a message queue abstraction using Redis Streams with:
- Consumer groups for horizontal scaling
- At-least-once delivery semantics
- Automatic dead letter queue for failed messages
- Backpressure via stream length limits
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import redis.asyncio as redis

from src.config.settings import get_settings
from src.ingestion.schemas import NormalizedDocument

logger = logging.getLogger(__name__)


@dataclass
class QueueMessage:
    """A message from the queue with its ID for acknowledgment."""

    message_id: str
    document: NormalizedDocument


class DocumentQueue:
    """
    Redis Streams wrapper for the document ingestion pipeline.

    Streams:
        - 'raw_documents': Main stream for incoming normalized documents
        - 'raw_documents:dlq': Dead letter queue for failed messages

    Consumer Groups:
        - 'processing_workers': Workers that process documents

    Usage:
        async with DocumentQueue() as queue:
            await queue.publish(doc)

            async for msg in queue.consume():
                process(msg.document)
                await queue.ack(msg.message_id)
    """

    def __init__(
        self,
        redis_url: str | None = None,
        stream_name: str | None = None,
        consumer_group: str | None = None,
        max_stream_length: int | None = None,
    ):
        settings = get_settings()

        self._redis_url = redis_url or str(settings.redis_url)
        self._stream_name = stream_name or settings.redis_stream_name
        self._consumer_group = consumer_group or settings.redis_consumer_group
        self._max_stream_length = max_stream_length or settings.redis_max_stream_length
        self._dlq_stream = f"{self._stream_name}:dlq"

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
        self._consumer_name = f"worker_{uuid.uuid4().hex[:8]}"

        # Create consumer group if it doesn't exist
        # Use MKSTREAM to create stream if it doesn't exist
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
            logger.info("Redis connection closed")

    async def __aenter__(self) -> "DocumentQueue":
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

    async def publish(self, doc: NormalizedDocument) -> str:
        """
        Add document to the processing queue.

        Args:
            doc: Normalized document to publish

        Returns:
            Message ID assigned by Redis

        Raises:
            RuntimeError: If not connected to Redis
        """
        message_id = await self.redis.xadd(
            name=self._stream_name,
            fields={"data": doc.model_dump_json()},
            maxlen=self._max_stream_length,
            approximate=True,  # More efficient trimming
        )

        logger.debug(
            f"Published document {doc.id} to stream, message_id={message_id}"
        )
        return str(message_id)

    async def publish_batch(self, docs: list[NormalizedDocument]) -> list[str]:
        """
        Publish multiple documents efficiently using pipeline.

        Args:
            docs: List of documents to publish

        Returns:
            List of message IDs
        """
        if not docs:
            return []

        pipe = self.redis.pipeline()
        for doc in docs:
            pipe.xadd(
                name=self._stream_name,
                fields={"data": doc.model_dump_json()},
                maxlen=self._max_stream_length,
                approximate=True,
            )

        results = await pipe.execute()
        message_ids = [str(r) for r in results]

        logger.info(f"Published batch of {len(docs)} documents")
        return message_ids

    async def consume(
        self,
        count: int = 10,
        block_ms: int = 5000,
    ) -> AsyncIterator[QueueMessage]:
        """
        Consume documents from the queue.

        Uses XREADGROUP for consumer group semantics. Messages must be
        acknowledged with ack() after successful processing.

        Args:
            count: Maximum number of messages to fetch per iteration
            block_ms: How long to block waiting for messages (milliseconds)

        Yields:
            QueueMessage with document and message_id for acknowledgment
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
                            doc = NormalizedDocument.model_validate_json(
                                fields["data"]
                            )
                            yield QueueMessage(
                                message_id=msg_id,
                                document=doc,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to parse message {msg_id}: {e}"
                            )
                            # Move to DLQ and acknowledge original
                            await self._move_to_dlq(msg_id, fields, str(e))
                            await self.ack(msg_id)

            except asyncio.CancelledError:
                logger.info("Consumer cancelled, stopping gracefully")
                break
            except Exception as e:
                logger.error(f"Error consuming messages: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def ack(self, message_id: str) -> None:
        """
        Acknowledge successful processing of a message.

        Args:
            message_id: The message ID to acknowledge
        """
        await self.redis.xack(
            self._stream_name,
            self._consumer_group,
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
        """Move a failed message to the dead letter queue."""
        import time as time_module
        dlq_fields = {
            **fields,
            "original_id": original_id,
            "error": error or "unknown",
            "failed_at": time_module.time(),  # Unix timestamp
        }

        await self.redis.xadd(
            self._dlq_stream,
            dlq_fields,
            maxlen=10000,  # Keep last 10k failed messages
        )
        logger.warning(
            f"Moved message {original_id} to DLQ: {error}"
        )

    async def get_pending_count(self) -> int:
        """Get count of pending (unacknowledged) messages."""
        info = await self.redis.xpending(
            self._stream_name,
            self._consumer_group,
        )
        return info["pending"] if info else 0

    async def get_stream_length(self) -> int:
        """Get total number of messages in the stream."""
        return await self.redis.xlen(self._stream_name)

    async def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False


@asynccontextmanager
async def get_queue() -> AsyncIterator[DocumentQueue]:
    """
    Context manager for getting a connected queue instance.

    Usage:
        async with get_queue() as queue:
            await queue.publish(doc)
    """
    queue = DocumentQueue()
    try:
        await queue.connect()
        yield queue
    finally:
        await queue.close()
