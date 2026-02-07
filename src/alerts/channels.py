"""Notification channel implementations for alert delivery.

Provides an ABC for notification channels plus concrete implementations
for webhooks and Slack. A CircuitBreaker decorator wraps any channel
to prevent cascading failures when downstream services are unhealthy.

Pattern: Decorator (CircuitBreaker wraps any NotificationChannel).
"""

import enum
import logging
import time
from abc import ABC, abstractmethod

import httpx

from src.alerts.schemas import Alert

logger = logging.getLogger(__name__)


class NotificationChannel(ABC):
    """Abstract base for notification delivery channels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this channel (e.g. 'webhook', 'slack')."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Deliver an alert through this channel.

        Args:
            alert: Alert to deliver.

        Returns:
            True if delivery succeeded, False otherwise.
        """


class WebhookChannel(NotificationChannel):
    """Delivers alerts as JSON POST to an arbitrary HTTP endpoint.

    Creates a new ``httpx.AsyncClient`` per call (short-lived, no pooling)
    matching the project's HTTP pattern.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "webhook"

    def _build_payload(self, alert: Alert) -> dict:
        """Build the webhook JSON payload from an alert.

        Matches the spec: alert_id, theme_id, trigger_type, severity,
        title, message, timestamp, metadata.
        """
        return {
            "alert_id": alert.alert_id,
            "theme_id": alert.theme_id,
            "trigger_type": alert.trigger_type,
            "severity": alert.severity,
            "title": alert.title,
            "message": alert.message,
            "timestamp": alert.created_at.isoformat(),
            "metadata": alert.trigger_data,
        }

    async def send(self, alert: Alert) -> bool:
        payload = self._build_payload(alert)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._url,
                    json=payload,
                    headers=self._headers,
                )
                if resp.is_success:
                    return True
                logger.warning(
                    "Webhook %s returned %d for alert %s",
                    self._url, resp.status_code, alert.alert_id,
                )
                return False
        except httpx.TimeoutException:
            logger.warning(
                "Webhook %s timed out for alert %s",
                self._url, alert.alert_id,
            )
            return False
        except Exception as e:
            logger.warning(
                "Webhook %s failed for alert %s: %s",
                self._url, alert.alert_id, e,
            )
            return False


class SlackChannel(NotificationChannel):
    """Delivers alerts to a Slack channel via incoming webhook.

    Formats alerts using Slack Block Kit for rich display.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._webhook_url = webhook_url
        self._channel = channel
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "slack"

    def _format_message(self, alert: Alert) -> dict:
        """Build Slack Block Kit payload from an alert."""
        severity_emoji = {
            "critical": ":red_circle:",
            "warning": ":large_orange_circle:",
            "info": ":large_blue_circle:",
        }
        emoji = severity_emoji.get(alert.severity, ":white_circle:")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {alert.title}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message,
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"*Severity:* {alert.severity} | "
                            f"*Trigger:* {alert.trigger_type} | "
                            f"*Theme:* {alert.theme_id}"
                        ),
                    },
                ],
            },
        ]

        payload: dict = {"blocks": blocks}
        if self._channel:
            payload["channel"] = self._channel
        return payload

    async def send(self, alert: Alert) -> bool:
        payload = self._format_message(alert)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._webhook_url, json=payload)
                if resp.is_success:
                    return True
                logger.warning(
                    "Slack webhook returned %d for alert %s",
                    resp.status_code, alert.alert_id,
                )
                return False
        except httpx.TimeoutException:
            logger.warning(
                "Slack webhook timed out for alert %s", alert.alert_id,
            )
            return False
        except Exception as e:
            logger.warning(
                "Slack webhook failed for alert %s: %s", alert.alert_id, e,
            )
            return False


class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker(NotificationChannel):
    """Wraps a NotificationChannel with circuit breaker protection.

    State machine: CLOSED → OPEN → HALF_OPEN → CLOSED.

    - CLOSED: All requests pass through. Consecutive failures tracked.
    - OPEN: Requests rejected immediately. After recovery_timeout, moves
      to HALF_OPEN.
    - HALF_OPEN: Single probe request allowed. Success → CLOSED, failure → OPEN.
    """

    def __init__(
        self,
        channel: NotificationChannel,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self._channel = channel
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time: float = 0.0

    @property
    def name(self) -> str:
        return self._channel.name

    @property
    def state(self) -> CircuitState:
        return self._state

    async def send(self, alert: Alert) -> bool:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit breaker %s: OPEN → HALF_OPEN (recovery probe)",
                    self.name,
                )
            else:
                logger.debug(
                    "Circuit breaker %s: OPEN, rejecting alert %s",
                    self.name, alert.alert_id,
                )
                return False

        success = await self._channel.send(alert)

        if success:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    "Circuit breaker %s: HALF_OPEN → CLOSED (probe succeeded)",
                    self.name,
                )
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker %s: HALF_OPEN → OPEN (probe failed)",
                    self.name,
                )
            elif self._consecutive_failures >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker %s: CLOSED → OPEN after %d failures",
                    self.name, self._consecutive_failures,
                )

        return success
