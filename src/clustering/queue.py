"""
Redis Streams queue for clustering jobs.

Provides a clean abstraction over Redis Streams for clustering job processing
with consumer groups, automatic pending message reclaim, DLQ support, and
graceful shutdown.
"""

import logging
import time
from dataclasses import dataclass

from src.clustering.config import ClusteringConfig
from src.config.settings import get_settings
from src.queues import BaseRedisQueue, QueueConfig, StreamConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusteringJob:
    """Clustering job from the queue."""

    document_id: str
    embedding_model: str
    message_id: str
    retry_count: int = 0


class ClusteringQueue(BaseRedisQueue[ClusteringJob]):
    """
    Redis Streams wrapper for clustering job processing.

    Handles:
    - Publishing jobs to the clustering stream
    - Consuming jobs with consumer groups
    - Automatic reclaim of pending messages from crashed workers
    - Acknowledgements and retries
    - Dead letter queue for failed jobs
    - Stream trimming to prevent memory growth

    Usage:
        queue = ClusteringQueue()
        await queue.connect()

        # Publish job
        await queue.publish("twitter_123", "finbert")

        # Consume jobs
        async for job in queue.consume():
            # Process job
            await queue.ack(job.message_id)
    """

    def __init__(self, config: ClusteringConfig | None = None):
        """
        Initialize queue.

        Args:
            config: Clustering configuration
        """
        settings = get_settings()
        self._config = config or ClusteringConfig()

        super().__init__(
            redis_url=str(settings.redis_url),
            queue_config=QueueConfig(
                idle_timeout_ms=self._config.idle_timeout_ms,
                max_delivery_attempts=self._config.max_delivery_attempts,
            ),
        )

    def _get_stream_config(self) -> StreamConfig:
        """Return stream configuration for clustering queue."""
        return StreamConfig(
            stream_name=self._config.stream_name,
            consumer_group=self._config.consumer_group,
            dlq_stream_name=self._config.dlq_stream_name,
            max_stream_length=self._config.max_stream_length,
        )

    def _get_consumer_prefix(self) -> str:
        """Return consumer name prefix."""
        return "clustering_worker"

    def _parse_job(self, message_id: str, fields: dict[str, str]) -> ClusteringJob:
        """Parse Redis message into ClusteringJob."""
        return ClusteringJob(
            document_id=fields["document_id"],
            embedding_model=fields.get("embedding_model", "finbert"),
            message_id=message_id,
        )

    def _set_job_retry_count(self, job: ClusteringJob, retry_count: int) -> None:
        """Set retry count on ClusteringJob."""
        job.retry_count = retry_count

    async def publish(self, document_id: str, embedding_model: str) -> str:
        """
        Publish a clustering job to the queue.

        Args:
            document_id: Document ID to process
            embedding_model: Model used to generate the embedding (e.g. "finbert")

        Returns:
            Message ID
        """
        message_id = await self.redis.xadd(
            name=self.stream_config.stream_name,
            fields={
                "document_id": document_id,
                "embedding_model": embedding_model,
                "queued_at": str(time.time()),
            },
            maxlen=self.stream_config.max_stream_length,
            approximate=True,
        )

        logger.debug(
            f"Published clustering job for document_id={document_id}, "
            f"model={embedding_model}"
        )
        return str(message_id)

    async def publish_batch(
        self, document_ids: list[str], embedding_model: str
    ) -> list[str]:
        """
        Publish multiple clustering jobs to the queue.

        Args:
            document_ids: List of document IDs to process
            embedding_model: Model used to generate the embeddings

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
                    "embedding_model": embedding_model,
                    "queued_at": queued_at,
                },
                maxlen=self.stream_config.max_stream_length,
                approximate=True,
            )

        results = await pipe.execute()
        message_ids = [str(r) for r in results]

        logger.info(f"Published {len(document_ids)} clustering jobs")
        return message_ids
