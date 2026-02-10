"""
Exponential backoff utility for retry loops.

Provides configurable delay calculation with jitter for workers and
queue consumers that need to reconnect after transient failures.
"""

import random


class ExponentialBackoff:
    """
    Exponential backoff with jitter.

    Computes delays as: min(base * multiplier^attempt, max_delay) + jitter.
    Call reset() after a successful operation to zero the attempt counter.

    Usage:
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0)
        while True:
            try:
                await do_work()
                backoff.reset()
            except Exception:
                delay = backoff.next_delay()
                await asyncio.sleep(delay)
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter_range: float = 0.5,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter_range = jitter_range
        self._attempt = 0

    @property
    def attempt(self) -> int:
        """Current attempt count."""
        return self._attempt

    def next_delay(self) -> float:
        """Calculate and return the next backoff delay, incrementing the attempt counter."""
        delay = min(
            self.base_delay * (self.multiplier ** self._attempt),
            self.max_delay,
        )
        # Add random jitter: ±jitter_range fraction of delay
        jitter = delay * random.uniform(-self.jitter_range, self.jitter_range)
        delay = max(0, delay + jitter)
        self._attempt += 1
        return delay

    def reset(self) -> None:
        """Reset the attempt counter after a successful operation."""
        self._attempt = 0
