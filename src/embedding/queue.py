"""
Redis Streams wrapper for embedding job queue.

Provides a message queue abstraction for embedding jobs with:
- Consumer groups for horizontal scaling
- At-least-once delivery semantics with automatic pending message reclaim
- Dead letter queue for failed jobs
- Batch publishing for efficiency
"""

import logging
import time
from dataclasses import dataclass

from src.config.settings import get_settings
from src.embedding.config import EmbeddingConfig
from src.queues import BaseRedisQueue, QueueConfig, StreamConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingJob:
    """
    An embedding job from the queue.

    Attributes:
        message_id: Redis stream message ID for acknowledgment
        document_id: ID of the document to embed
        priority: Optional priority level (higher = more urgent)
        retry_count: Number of previous delivery attempts
    """

    message_id: str
    document_id: str
    priority: int = 0
    retry_count: int = 0


class EmbeddingQueue(BaseRedisQueue[EmbeddingJob]):
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

        # Initialize base class with reclaim configuration
        super().__init__(
            redis_url=redis_url or str(settings.redis_url),
            queue_config=QueueConfig(
                idle_timeout_ms=self._config.idle_timeout_ms,
                max_delivery_attempts=self._config.max_delivery_attempts,
            ),
        )

    def _get_stream_config(self) -> StreamConfig:
        """Return stream configuration for embedding queue."""
        return StreamConfig(
            stream_name=self._config.stream_name,
            consumer_group=self._config.consumer_group,
            dlq_stream_name=self._config.dlq_stream_name,
            max_stream_length=self._config.max_stream_length,
        )

    def _get_consumer_prefix(self) -> str:
        """Return consumer name prefix."""
        return "emb_worker"

    def _parse_job(self, message_id: str, fields: dict[str, str]) -> EmbeddingJob:
        """Parse Redis message into EmbeddingJob."""
        return EmbeddingJob(
            message_id=message_id,
            document_id=fields["document_id"],
            priority=int(fields.get("priority", 0)),
        )

    def _set_job_retry_count(self, job: EmbeddingJob, retry_count: int) -> None:
        """Set retry count on EmbeddingJob."""
        job.retry_count = retry_count

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
            name=self.stream_config.stream_name,
            fields={
                "document_id": document_id,
                "priority": str(priority),
                "queued_at": str(time.time()),
            },
            maxlen=self.stream_config.max_stream_length,
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
                name=self.stream_config.stream_name,
                fields={
                    "document_id": doc_id,
                    "priority": str(priority),
                    "queued_at": queued_at,
                },
                maxlen=self.stream_config.max_stream_length,
                approximate=True,
            )

        results = await pipe.execute()
        message_ids = [str(r) for r in results]

        logger.info(f"Published batch of {len(document_ids)} embedding jobs")
        return message_ids
