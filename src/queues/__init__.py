"""
Redis Streams queue abstractions with automatic pending message reclaim.

This package provides a base class for Redis Streams queues that implements
at-least-once delivery semantics by automatically reclaiming messages that
were delivered but never acknowledged (typically due to worker crashes).

Classes:
    BaseRedisQueue: Abstract base class for Redis Streams queues
    StreamConfig: Configuration for a Redis Stream
    QueueConfig: Configuration for reclaim behavior

Example:
    from src.queues import BaseRedisQueue, StreamConfig, QueueConfig

    class MyQueue(BaseRedisQueue[MyJob]):
        def __init__(self, redis_url: str):
            super().__init__(
                redis_url=redis_url,
                queue_config=QueueConfig(
                    idle_timeout_ms=30_000,
                    max_delivery_attempts=3,
                ),
            )

        def _parse_job(self, message_id, fields) -> MyJob:
            return MyJob(message_id=message_id, data=fields["data"])

        def _get_stream_config(self) -> StreamConfig:
            return StreamConfig(
                stream_name="my_stream",
                consumer_group="my_workers",
                dlq_stream_name="my_stream:dlq",
            )
"""

from src.queues.base import BaseRedisQueue, StreamConfig
from src.queues.config import QueueConfig

__all__ = ["BaseRedisQueue", "StreamConfig", "QueueConfig"]
