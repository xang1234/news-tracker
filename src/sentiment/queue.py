"""
Redis Streams queue for sentiment analysis jobs.

Provides a clean abstraction over Redis Streams for sentiment job processing
with consumer groups, automatic pending message reclaim, DLQ support, and
graceful shutdown.
"""

import logging
from dataclasses import dataclass

from src.config.settings import get_settings
from src.queues import BaseRedisQueue, QueueConfig, StreamConfig
from src.sentiment.config import SentimentConfig

logger = logging.getLogger(__name__)


@dataclass
class SentimentJob:
    """Sentiment analysis job from the queue."""

    document_id: str
    message_id: str
    retry_count: int = 0


class SentimentQueue(BaseRedisQueue[SentimentJob]):
    """
    Redis Streams wrapper for sentiment job processing.

    Handles:
    - Publishing jobs to the sentiment stream
    - Consuming jobs with consumer groups
    - Automatic reclaim of pending messages from crashed workers
    - Acknowledgements and retries
    - Dead letter queue for failed jobs
    - Stream trimming to prevent memory growth

    Usage:
        queue = SentimentQueue()
        await queue.connect()

        # Publish job
        await queue.publish("twitter_123")

        # Consume jobs
        async for job in queue.consume():
            # Process job
            await queue.ack(job.message_id)
    """

    def __init__(self, config: SentimentConfig | None = None):
        """
        Initialize queue.

        Args:
            config: Sentiment configuration
        """
        settings = get_settings()
        self._config = config or SentimentConfig()

        # Initialize base class with reclaim configuration
        super().__init__(
            redis_url=str(settings.redis_url),
            queue_config=QueueConfig(
                idle_timeout_ms=self._config.idle_timeout_ms,
                max_delivery_attempts=self._config.max_delivery_attempts,
            ),
        )

    def _get_stream_config(self) -> StreamConfig:
        """Return stream configuration for sentiment queue."""
        return StreamConfig(
            stream_name=self._config.stream_name,
            consumer_group=self._config.consumer_group,
            dlq_stream_name=self._config.dlq_stream_name,
            max_stream_length=self._config.max_stream_length,
        )

    def _get_consumer_prefix(self) -> str:
        """Return consumer name prefix."""
        return "sentiment_worker"

    def _parse_job(self, message_id: str, fields: dict[str, str]) -> SentimentJob:
        """Parse Redis message into SentimentJob."""
        return SentimentJob(
            document_id=fields["document_id"],
            message_id=message_id,
        )

    def _set_job_retry_count(self, job: SentimentJob, retry_count: int) -> None:
        """Set retry count on SentimentJob."""
        job.retry_count = retry_count

    async def publish(self, document_id: str) -> str:
        """
        Publish a sentiment job to the queue.

        Args:
            document_id: Document ID to process

        Returns:
            Message ID
        """
        message_id = await self.redis.xadd(
            name=self.stream_config.stream_name,
            fields={"document_id": document_id, **self._trace_fields()},
            maxlen=self.stream_config.max_stream_length,
            approximate=True,
        )

        logger.debug(f"Published sentiment job for document_id={document_id}")
        return str(message_id)

    async def publish_batch(self, document_ids: list[str]) -> list[str]:
        """
        Publish multiple sentiment jobs to the queue.

        Args:
            document_ids: List of document IDs to process

        Returns:
            List of message IDs
        """
        if not document_ids:
            return []

        trace_fields = self._trace_fields()
        pipe = self.redis.pipeline()
        for doc_id in document_ids:
            pipe.xadd(
                name=self.stream_config.stream_name,
                fields={"document_id": doc_id, **trace_fields},
                maxlen=self.stream_config.max_stream_length,
                approximate=True,
            )

        results = await pipe.execute()
        message_ids = [str(r) for r in results]

        logger.info(f"Published {len(document_ids)} sentiment jobs")
        return message_ids
