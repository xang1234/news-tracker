"""Redis queue for narrative jobs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.config.settings import get_settings
from src.narrative.config import NarrativeConfig
from src.queues import BaseRedisQueue, QueueConfig, StreamConfig

logger = logging.getLogger(__name__)


@dataclass
class NarrativeJob:
    """Narrative processing job."""

    document_id: str
    theme_id: str
    theme_similarity: float
    message_id: str
    retry_count: int = 0


class NarrativeQueue(BaseRedisQueue[NarrativeJob]):
    """Redis Streams wrapper for narrative jobs."""

    def __init__(self, config: NarrativeConfig | None = None):
        settings = get_settings()
        self._config = config or NarrativeConfig()
        super().__init__(
            redis_url=str(settings.redis_url),
            queue_config=QueueConfig(
                idle_timeout_ms=self._config.idle_timeout_ms,
                max_delivery_attempts=self._config.max_delivery_attempts,
            ),
        )

    def _get_stream_config(self) -> StreamConfig:
        return StreamConfig(
            stream_name=self._config.stream_name,
            consumer_group=self._config.consumer_group,
            dlq_stream_name=self._config.dlq_stream_name,
            max_stream_length=self._config.max_stream_length,
        )

    def _get_consumer_prefix(self) -> str:
        return "narrative_worker"

    def _parse_job(self, message_id: str, fields: dict[str, str]) -> NarrativeJob:
        return NarrativeJob(
            document_id=fields["document_id"],
            theme_id=fields["theme_id"],
            theme_similarity=float(fields.get("theme_similarity", "0")),
            message_id=message_id,
        )

    def _set_job_retry_count(self, job: NarrativeJob, retry_count: int) -> None:
        job.retry_count = retry_count

    async def publish(
        self,
        document_id: str,
        theme_id: str,
        theme_similarity: float,
    ) -> str:
        message_id = await self.redis.xadd(
            name=self.stream_config.stream_name,
            fields={
                "document_id": document_id,
                "theme_id": theme_id,
                "theme_similarity": f"{theme_similarity:.6f}",
                "queued_at": str(time.time()),
                **self._trace_fields(),
            },
            maxlen=self.stream_config.max_stream_length,
            approximate=True,
        )
        logger.debug(
            "Published narrative job for %s theme=%s similarity=%.4f",
            document_id,
            theme_id,
            theme_similarity,
        )
        return str(message_id)
