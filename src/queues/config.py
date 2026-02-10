"""
Queue configuration for Redis Streams message reclaim behavior.

Provides settings for idle message detection and retry limits to ensure
at-least-once delivery semantics with dead letter queue handling.
"""

from dataclasses import dataclass


@dataclass
class QueueConfig:
    """
    Configuration for queue message reclaim behavior.

    Attributes:
        idle_timeout_ms: Time in milliseconds after which an unacknowledged
            message is considered idle and eligible for reclaim by another
            consumer. Default 30 seconds strikes a balance between quick
            recovery and avoiding premature reclaims during slow processing.

        max_delivery_attempts: Maximum number of times a message can be
            delivered before being moved to the dead letter queue.
            After this many attempts, the message is considered unprocessable.

        reclaim_batch_size: Number of pending messages to attempt to reclaim
            in a single XAUTOCLAIM call. Larger batches are more efficient
            but may delay processing new messages.
    """

    idle_timeout_ms: int = 30_000  # 30 seconds
    max_delivery_attempts: int = 3
    reclaim_batch_size: int = 10

    # Backoff settings for consume() error recovery
    backoff_base_delay: float = 1.0
    backoff_max_delay: float = 60.0
