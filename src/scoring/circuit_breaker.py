"""Generic circuit breaker for wrapping any async callable.

Provides the same CLOSED → OPEN → HALF_OPEN → CLOSED state machine as
``src/alerts/channels.CircuitBreaker``, but decoupled from NotificationChannel.
Can wrap any ``async def`` — LLM API calls, HTTP requests, etc.

Usage:
    breaker = GenericCircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    try:
        result = await breaker.call(some_async_fn, arg1, arg2)
    except CircuitOpenError:
        # Fallback logic
"""

import enum
import logging
import time
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when calling through an open circuit breaker."""


class GenericCircuitBreaker:
    """Wraps any async callable with circuit breaker protection.

    State machine: CLOSED → OPEN → HALF_OPEN → CLOSED.

    - CLOSED: Calls pass through. Consecutive failures tracked.
    - OPEN: Calls rejected with CircuitOpenError. After recovery_timeout,
      moves to HALF_OPEN.
    - HALF_OPEN: Single probe call allowed. Success → CLOSED, failure → OPEN.

    Args:
        failure_threshold: Consecutive failures before opening circuit.
        recovery_timeout: Seconds before attempting a recovery probe.
        name: Optional name for logging.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "circuit_breaker",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive failures."""
        return self._consecutive_failures

    async def call(
        self,
        fn: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async function through the circuit breaker.

        Args:
            fn: Async callable to execute.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            Result from fn.

        Raises:
            CircuitOpenError: If circuit is open and recovery timeout
                has not elapsed.
        """
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit breaker %s: OPEN → HALF_OPEN (recovery probe)",
                    self._name,
                )
            else:
                raise CircuitOpenError(
                    f"Circuit breaker {self._name} is OPEN"
                )

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise

        # Success path
        if self._state == CircuitState.HALF_OPEN:
            logger.info(
                "Circuit breaker %s: HALF_OPEN → CLOSED (probe succeeded)",
                self._name,
            )
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        return result

    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker %s: HALF_OPEN → OPEN (probe failed)",
                self._name,
            )
        elif self._consecutive_failures >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker %s: CLOSED → OPEN after %d failures",
                self._name,
                self._consecutive_failures,
            )
