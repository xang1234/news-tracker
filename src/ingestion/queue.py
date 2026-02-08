"""
Redis Streams wrapper for document ingestion pipeline.

Provides a message queue abstraction using Redis Streams with:
- Consumer groups for horizontal scaling
- At-least-once delivery semantics with automatic pending message reclaim
- Automatic dead letter queue for failed messages
- Backpressure via stream length limits
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from src.config.settings import get_settings
from src.ingestion.schemas import NormalizedDocument
from src.queues import BaseRedisQueue, QueueConfig, StreamConfig

logger = logging.getLogger(__name__)


@dataclass
class QueueMessage:
    """A message from the queue with its ID for acknowledgment."""

    message_id: str
    document: NormalizedDocument
    retry_count: int = 0


class DocumentQueue(BaseRedisQueue[QueueMessage]):
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

        self._custom_stream_name = stream_name or settings.redis_stream_name
        self._custom_consumer_group = consumer_group or settings.redis_consumer_group
        self._max_stream_length = max_stream_length or settings.redis_max_stream_length

        # Initialize base class with reclaim configuration
        super().__init__(
            redis_url=redis_url or str(settings.redis_url),
            queue_config=QueueConfig(
                idle_timeout_ms=settings.queue_idle_timeout_ms,
                max_delivery_attempts=settings.queue_max_delivery_attempts,
            ),
        )

    def _get_stream_config(self) -> StreamConfig:
        """Return stream configuration for document queue."""
        return StreamConfig(
            stream_name=self._custom_stream_name,
            consumer_group=self._custom_consumer_group,
            dlq_stream_name=f"{self._custom_stream_name}:dlq",
            max_stream_length=self._max_stream_length,
        )

    def _get_consumer_prefix(self) -> str:
        """Return consumer name prefix."""
        return "worker"

    def _parse_job(self, message_id: str, fields: dict[str, str]) -> QueueMessage:
        """Parse Redis message into QueueMessage."""
        doc = NormalizedDocument.model_validate_json(fields["data"])
        return QueueMessage(
            message_id=message_id,
            document=doc,
        )

    def _set_job_retry_count(self, job: QueueMessage, retry_count: int) -> None:
        """Set retry count on QueueMessage."""
        job.retry_count = retry_count

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
            name=self.stream_config.stream_name,
            fields={"data": doc.model_dump_json(), **self._trace_fields()},
            maxlen=self.stream_config.max_stream_length,
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

        trace_fields = self._trace_fields()
        pipe = self.redis.pipeline()
        for doc in docs:
            pipe.xadd(
                name=self.stream_config.stream_name,
                fields={"data": doc.model_dump_json(), **trace_fields},
                maxlen=self.stream_config.max_stream_length,
                approximate=True,
            )

        results = await pipe.execute()
        message_ids = [str(r) for r in results]

        logger.info(f"Published batch of {len(docs)} documents")
        return message_ids


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
