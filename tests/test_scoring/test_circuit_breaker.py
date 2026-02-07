"""Tests for the generic circuit breaker."""

import time

import pytest

from src.scoring.circuit_breaker import (
    CircuitOpenError,
    CircuitState,
    GenericCircuitBreaker,
)


async def _success() -> str:
    return "ok"


async def _failure() -> str:
    raise RuntimeError("boom")


class TestClosedState:
    """Circuit in CLOSED state passes calls through."""

    async def test_passthrough_success(self) -> None:
        breaker = GenericCircuitBreaker(failure_threshold=3)
        result = await breaker.call(_success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 0

    async def test_single_failure_stays_closed(self) -> None:
        breaker = GenericCircuitBreaker(failure_threshold=3)
        with pytest.raises(RuntimeError, match="boom"):
            await breaker.call(_failure)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 1

    async def test_success_resets_failure_count(self) -> None:
        breaker = GenericCircuitBreaker(failure_threshold=3)
        # Two failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_failure)
        assert breaker.consecutive_failures == 2

        # One success resets
        await breaker.call(_success)
        assert breaker.consecutive_failures == 0
        assert breaker.state == CircuitState.CLOSED


class TestOpenState:
    """Circuit opens after threshold failures and rejects calls."""

    async def test_opens_after_threshold(self) -> None:
        breaker = GenericCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(_failure)
        assert breaker.state == CircuitState.OPEN

    async def test_rejects_when_open(self) -> None:
        breaker = GenericCircuitBreaker(failure_threshold=2)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_failure)
        assert breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError, match="OPEN"):
            await breaker.call(_success)


class TestHalfOpenRecovery:
    """Circuit transitions to HALF_OPEN after recovery timeout."""

    async def test_recovery_probe_success(self) -> None:
        breaker = GenericCircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.01,  # Very short for testing
        )
        # Open the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_failure)
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)

        # Probe should succeed and close circuit
        result = await breaker.call(_success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 0

    async def test_recovery_probe_failure(self) -> None:
        breaker = GenericCircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.01,
        )
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_failure)

        time.sleep(0.02)

        # Failed probe should re-open circuit
        with pytest.raises(RuntimeError, match="boom"):
            await breaker.call(_failure)
        assert breaker.state == CircuitState.OPEN


class TestCallWithArgs:
    """Circuit breaker passes args/kwargs to the wrapped function."""

    async def test_args_passthrough(self) -> None:
        async def _add(a: int, b: int) -> int:
            return a + b

        breaker = GenericCircuitBreaker()
        result = await breaker.call(_add, 3, 7)
        assert result == 10

    async def test_kwargs_passthrough(self) -> None:
        async def _greet(name: str, prefix: str = "Hello") -> str:
            return f"{prefix} {name}"

        breaker = GenericCircuitBreaker()
        result = await breaker.call(_greet, "world", prefix="Hi")
        assert result == "Hi world"
