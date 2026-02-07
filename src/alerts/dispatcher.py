"""Notification dispatcher orchestrating alert delivery across channels.

Handles retry logic, circuit breaker wrapping, and Redis fallback queuing
for failed deliveries. Notification failures never block alert persistence
(graceful degradation).

Pattern: Orchestrator (like AlertService), delegates to stateless channels.
"""

import asyncio
import json
import logging
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.alerts.channels import CircuitBreaker, NotificationChannel
from src.alerts.schemas import Alert

logger = logging.getLogger(__name__)


class NotificationConfig(BaseSettings):
    """Configuration for notification dispatch."""

    model_config = SettingsConfigDict(
        env_prefix="NOTIFICATIONS_",
        case_sensitive=False,
        extra="ignore",
    )

    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum send attempts per channel per alert",
    )
    retry_delays: list[float] = Field(
        default=[1.0, 5.0, 30.0],
        description="Per-attempt delay in seconds before each retry",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive failures before circuit opens",
    )
    circuit_breaker_recovery_seconds: float = Field(
        default=60.0,
        ge=5.0,
        description="Seconds before circuit breaker probes recovery",
    )
    queue_key_prefix: str = Field(
        default="notify:retry",
        description="Redis key prefix for retry queue",
    )
    queue_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="TTL for queued retry items",
    )


class NotificationDispatcher:
    """Orchestrates alert delivery across notification channels.

    Wraps each channel in a CircuitBreaker and provides retry logic
    with Redis fallback queuing for persistent failures.
    """

    def __init__(
        self,
        channels: list[NotificationChannel],
        config: NotificationConfig | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self._config = config or NotificationConfig()
        self._redis = redis_client

        # Wrap each channel in a circuit breaker
        self._channels: list[CircuitBreaker] = []
        for ch in channels:
            if isinstance(ch, CircuitBreaker):
                self._channels.append(ch)
            else:
                self._channels.append(
                    CircuitBreaker(
                        channel=ch,
                        failure_threshold=self._config.circuit_breaker_threshold,
                        recovery_timeout=self._config.circuit_breaker_recovery_seconds,
                    )
                )

    @property
    def channels(self) -> list[CircuitBreaker]:
        """Access wrapped channels (for inspection/testing)."""
        return self._channels

    async def dispatch(self, alert: Alert) -> list[tuple[str, bool]]:
        """Send an alert to all configured channels.

        Args:
            alert: Alert to deliver.

        Returns:
            List of (channel_name, success) tuples.
        """
        results: list[tuple[str, bool]] = []

        for channel in self._channels:
            success = await self._send_with_retry(channel, alert)
            results.append((channel.name, success))

            if not success:
                await self._queue_for_retry(alert, channel.name)

        self._record_delivery(alert, results)
        return results

    async def dispatch_batch(self, alerts: list[Alert]) -> None:
        """Send a batch of alerts, isolating failures per-alert.

        Each alert is dispatched independently — a failure for one alert
        does not affect delivery of others.

        Args:
            alerts: Alerts to deliver.
        """
        for alert in alerts:
            try:
                await self.dispatch(alert)
            except Exception as e:
                logger.error(
                    "Unexpected error dispatching alert %s: %s",
                    alert.alert_id, e,
                )

    async def _send_with_retry(
        self,
        channel: NotificationChannel,
        alert: Alert,
    ) -> bool:
        """Attempt to send with configured retries.

        Args:
            channel: Target notification channel.
            alert: Alert to deliver.

        Returns:
            True if any attempt succeeded.
        """
        delays = self._config.retry_delays
        max_attempts = self._config.retry_max_attempts

        for attempt in range(max_attempts):
            try:
                success = await channel.send(alert)
                if success:
                    if attempt > 0:
                        logger.info(
                            "Alert %s delivered to %s on attempt %d",
                            alert.alert_id, channel.name, attempt + 1,
                        )
                    return True
            except Exception as e:
                logger.warning(
                    "Channel %s send error (attempt %d): %s",
                    channel.name, attempt + 1, e,
                )

            # Wait before retry (if not the last attempt)
            if attempt < max_attempts - 1:
                delay = delays[attempt] if attempt < len(delays) else delays[-1]
                await asyncio.sleep(delay)

        logger.warning(
            "All %d attempts exhausted for alert %s on channel %s",
            max_attempts, alert.alert_id, channel.name,
        )
        return False

    async def _queue_for_retry(self, alert: Alert, channel_name: str) -> None:
        """Push a failed alert to Redis for later retry.

        Uses LPUSH with a per-channel key. Gracefully degrades if Redis
        is unavailable.

        Args:
            alert: Alert that failed delivery.
            channel_name: Channel that failed.
        """
        if self._redis is None:
            return

        key = f"{self._config.queue_key_prefix}:{channel_name}"
        payload = json.dumps(alert.to_dict())

        try:
            await self._redis.lpush(key, payload)
            ttl_seconds = self._config.queue_ttl_hours * 3600
            await self._redis.expire(key, ttl_seconds)
            logger.info(
                "Queued alert %s for retry on channel %s",
                alert.alert_id, channel_name,
            )
        except Exception as e:
            logger.warning(
                "Failed to queue alert %s for retry: %s",
                alert.alert_id, e,
            )

    def _record_delivery(
        self,
        alert: Alert,
        results: list[tuple[str, bool]],
    ) -> None:
        """Log delivery results. Extensible hook for future metrics.

        Args:
            alert: Delivered alert.
            results: Per-channel delivery outcomes.
        """
        successes = [name for name, ok in results if ok]
        failures = [name for name, ok in results if not ok]

        if failures and not successes:
            logger.error(
                "Alert %s (%s) failed ALL channels: %s",
                alert.alert_id, alert.severity, failures,
            )
        elif failures:
            logger.warning(
                "Alert %s partial delivery: ok=%s failed=%s",
                alert.alert_id, successes, failures,
            )
        else:
            logger.debug(
                "Alert %s delivered to all channels: %s",
                alert.alert_id, successes,
            )
