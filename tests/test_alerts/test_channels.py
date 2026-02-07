"""Tests for notification channels, circuit breaker, and dispatcher."""

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.alerts.channels import (
    CircuitBreaker,
    CircuitState,
    NotificationChannel,
    SlackChannel,
    WebhookChannel,
)
from src.alerts.dispatcher import NotificationConfig, NotificationDispatcher
from src.alerts.schemas import Alert


# ── Fixtures ────────────────────────────────────────────


@pytest.fixture
def sample_alert():
    return Alert(
        alert_id="test-alert-001",
        theme_id="theme_abc123",
        trigger_type="volume_surge",
        severity="critical",
        title="Volume surge: AI Chips",
        message="Theme 'AI Chips' volume z-score is 4.5σ above normal",
        trigger_data={"volume_zscore": 4.5, "document_count": 120},
        created_at=datetime(2026, 2, 7, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def info_alert():
    return Alert(
        alert_id="test-alert-002",
        theme_id="theme_def456",
        trigger_type="new_theme",
        severity="info",
        title="New theme: Supply Chain",
        message="New theme 'Supply Chain' detected",
    )


def _mock_response(status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response."""
    return httpx.Response(status_code=status_code, request=httpx.Request("POST", "http://test"))


# ── WebhookChannel ──────────────────────────────────────


class TestWebhookChannel:
    """Tests for WebhookChannel."""

    @pytest.mark.asyncio
    async def test_successful_send(self, sample_alert):
        channel = WebhookChannel(url="https://example.com/webhook")

        with patch("src.alerts.channels.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_response(200)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await channel.send(sample_alert)

        assert result is True
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["alert_id"] == "test-alert-001"
        assert payload["severity"] == "critical"
        assert payload["metadata"]["volume_zscore"] == 4.5

    @pytest.mark.asyncio
    async def test_failure_on_500(self, sample_alert):
        channel = WebhookChannel(url="https://example.com/webhook")

        with patch("src.alerts.channels.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_response(500)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await channel.send(sample_alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_alert):
        channel = WebhookChannel(url="https://example.com/webhook", timeout=1.0)

        with patch("src.alerts.channels.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await channel.send(sample_alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_payload_shape(self, sample_alert):
        channel = WebhookChannel(url="https://example.com/webhook")
        payload = channel._build_payload(sample_alert)

        expected_keys = {
            "alert_id", "theme_id", "trigger_type", "severity",
            "title", "message", "timestamp", "metadata",
        }
        assert set(payload.keys()) == expected_keys
        assert payload["timestamp"] == "2026-02-07T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_custom_headers(self, sample_alert):
        headers = {"Authorization": "Bearer test-token"}
        channel = WebhookChannel(
            url="https://example.com/webhook", headers=headers,
        )

        with patch("src.alerts.channels.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_response(200)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await channel.send(sample_alert)

        call_kwargs = mock_client.post.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert sent_headers["Authorization"] == "Bearer test-token"

    def test_name(self):
        channel = WebhookChannel(url="https://example.com/webhook")
        assert channel.name == "webhook"


# ── SlackChannel ────────────────────────────────────────


class TestSlackChannel:
    """Tests for SlackChannel."""

    @pytest.mark.asyncio
    async def test_successful_send(self, sample_alert):
        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")

        with patch("src.alerts.channels.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_response(200)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await channel.send(sample_alert)

        assert result is True

    @pytest.mark.asyncio
    async def test_message_formatting(self, sample_alert):
        channel = SlackChannel(
            webhook_url="https://hooks.slack.com/test",
            channel="#alerts",
        )
        payload = channel._format_message(sample_alert)

        assert "blocks" in payload
        assert payload["channel"] == "#alerts"
        assert len(payload["blocks"]) == 3

        # Header block contains title
        header = payload["blocks"][0]
        assert header["type"] == "header"
        assert "Volume surge" in header["text"]["text"]

        # Context block contains severity
        context = payload["blocks"][2]
        assert "critical" in context["elements"][0]["text"]

    @pytest.mark.asyncio
    async def test_non_200_response(self, sample_alert):
        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")

        with patch("src.alerts.channels.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_response(403)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await channel.send(sample_alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_no_channel_in_payload(self, sample_alert):
        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        payload = channel._format_message(sample_alert)
        assert "channel" not in payload

    def test_name(self):
        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        assert channel.name == "slack"

    @pytest.mark.asyncio
    async def test_severity_emojis(self):
        """Each severity level maps to a different emoji."""
        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")

        for severity, expected_emoji in [
            ("critical", ":red_circle:"),
            ("warning", ":large_orange_circle:"),
            ("info", ":large_blue_circle:"),
        ]:
            alert = Alert(
                theme_id="t1",
                trigger_type="volume_surge",
                severity=severity,
                title="Test",
                message="Test message",
            )
            payload = channel._format_message(alert)
            assert expected_emoji in payload["blocks"][0]["text"]["text"]


# ── CircuitBreaker ──────────────────────────────────────


class TestCircuitBreaker:
    """Tests for CircuitBreaker state machine."""

    @pytest.mark.asyncio
    async def test_passes_through_when_closed(self, sample_alert):
        inner = AsyncMock(spec=NotificationChannel)
        inner.name = "test"
        inner.send.return_value = True

        breaker = CircuitBreaker(inner, failure_threshold=3)
        result = await breaker.send(sample_alert)

        assert result is True
        assert breaker.state == CircuitState.CLOSED
        inner.send.assert_called_once_with(sample_alert)

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, sample_alert):
        inner = AsyncMock(spec=NotificationChannel)
        inner.name = "test"
        inner.send.return_value = False

        breaker = CircuitBreaker(inner, failure_threshold=3, recovery_timeout=60.0)

        for _ in range(3):
            await breaker.send(sample_alert)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_when_open(self, sample_alert):
        inner = AsyncMock(spec=NotificationChannel)
        inner.name = "test"
        inner.send.return_value = False

        breaker = CircuitBreaker(inner, failure_threshold=2, recovery_timeout=60.0)

        # Trigger open
        await breaker.send(sample_alert)
        await breaker.send(sample_alert)
        assert breaker.state == CircuitState.OPEN

        # Reset mock to track new calls
        inner.send.reset_mock()

        # Should reject without calling inner
        result = await breaker.send(sample_alert)
        assert result is False
        inner.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self, sample_alert):
        inner = AsyncMock(spec=NotificationChannel)
        inner.name = "test"
        inner.send.return_value = False

        breaker = CircuitBreaker(inner, failure_threshold=2, recovery_timeout=0.01)

        # Trigger open
        await breaker.send(sample_alert)
        await breaker.send(sample_alert)
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Next call should probe (HALF_OPEN)
        inner.send.return_value = True
        result = await breaker.send(sample_alert)
        assert result is True
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, sample_alert):
        inner = AsyncMock(spec=NotificationChannel)
        inner.name = "test"
        inner.send.return_value = False

        breaker = CircuitBreaker(inner, failure_threshold=2, recovery_timeout=0.01)

        # Trigger open
        await breaker.send(sample_alert)
        await breaker.send(sample_alert)
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.02)

        # Probe fails → back to OPEN
        result = await breaker.send(sample_alert)
        assert result is False
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, sample_alert):
        inner = AsyncMock(spec=NotificationChannel)
        inner.name = "test"

        breaker = CircuitBreaker(inner, failure_threshold=3)

        # Two failures, then a success
        inner.send.return_value = False
        await breaker.send(sample_alert)
        await breaker.send(sample_alert)
        assert breaker._consecutive_failures == 2

        inner.send.return_value = True
        await breaker.send(sample_alert)
        assert breaker._consecutive_failures == 0
        assert breaker.state == CircuitState.CLOSED

    def test_name_delegates(self):
        inner = AsyncMock(spec=NotificationChannel)
        inner.name = "webhook"
        breaker = CircuitBreaker(inner)
        assert breaker.name == "webhook"


# ── NotificationDispatcher ──────────────────────────────


class TestNotificationDispatcher:
    """Tests for dispatch orchestration."""

    @pytest.mark.asyncio
    async def test_dispatch_to_multiple_channels(self, sample_alert):
        ch1 = AsyncMock(spec=NotificationChannel)
        ch1.name = "webhook"
        ch1.send.return_value = True

        ch2 = AsyncMock(spec=NotificationChannel)
        ch2.name = "slack"
        ch2.send.return_value = True

        config = NotificationConfig(retry_max_attempts=1, retry_delays=[0.0])
        dispatcher = NotificationDispatcher(
            channels=[ch1, ch2], config=config,
        )

        results = await dispatcher.dispatch(sample_alert)

        assert len(results) == 2
        assert all(ok for _, ok in results)

    @pytest.mark.asyncio
    async def test_partial_failure(self, sample_alert):
        ch1 = AsyncMock(spec=NotificationChannel)
        ch1.name = "webhook"
        ch1.send.return_value = True

        ch2 = AsyncMock(spec=NotificationChannel)
        ch2.name = "slack"
        ch2.send.return_value = False

        config = NotificationConfig(retry_max_attempts=1, retry_delays=[0.0])
        dispatcher = NotificationDispatcher(
            channels=[ch1, ch2], config=config,
        )

        results = await dispatcher.dispatch(sample_alert)

        assert results[0] == ("webhook", True)
        assert results[1] == ("slack", False)

    @pytest.mark.asyncio
    async def test_batch_dispatch_isolates_errors(self, sample_alert, info_alert):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "webhook"
        # First alert fails, second succeeds
        ch.send.side_effect = [False, True]

        config = NotificationConfig(retry_max_attempts=1, retry_delays=[0.0])
        dispatcher = NotificationDispatcher(channels=[ch], config=config)

        # Should not raise even though first alert fails
        await dispatcher.dispatch_batch([sample_alert, info_alert])

        assert ch.send.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_catches_unexpected_errors(self, sample_alert, info_alert):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "webhook"
        ch.send.side_effect = [RuntimeError("unexpected"), True]

        config = NotificationConfig(retry_max_attempts=1, retry_delays=[0.0])
        dispatcher = NotificationDispatcher(channels=[ch], config=config)

        # Should not propagate the error
        await dispatcher.dispatch_batch([sample_alert, info_alert])

    @pytest.mark.asyncio
    async def test_wraps_channels_in_circuit_breaker(self):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "test"

        dispatcher = NotificationDispatcher(channels=[ch])
        assert len(dispatcher.channels) == 1
        assert isinstance(dispatcher.channels[0], CircuitBreaker)

    @pytest.mark.asyncio
    async def test_does_not_double_wrap_circuit_breaker(self):
        inner = AsyncMock(spec=NotificationChannel)
        inner.name = "test"
        breaker = CircuitBreaker(inner)

        dispatcher = NotificationDispatcher(channels=[breaker])
        assert dispatcher.channels[0] is breaker


# ── Retry Logic ─────────────────────────────────────────


class TestRetryLogic:
    """Tests for send retry behavior."""

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self, sample_alert):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "webhook"
        ch.send.side_effect = [False, True]

        config = NotificationConfig(
            retry_max_attempts=3, retry_delays=[0.0, 0.0, 0.0],
        )
        dispatcher = NotificationDispatcher(channels=[ch], config=config)
        results = await dispatcher.dispatch(sample_alert)

        assert results[0] == ("webhook", True)
        assert ch.send.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, sample_alert):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "webhook"
        ch.send.return_value = False

        config = NotificationConfig(
            retry_max_attempts=3, retry_delays=[0.0, 0.0, 0.0],
        )
        dispatcher = NotificationDispatcher(channels=[ch], config=config)
        results = await dispatcher.dispatch(sample_alert)

        assert results[0] == ("webhook", False)
        assert ch.send.call_count == 3

    @pytest.mark.asyncio
    async def test_redis_queue_fallback(self, sample_alert):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "webhook"
        ch.send.return_value = False

        mock_redis = AsyncMock()
        config = NotificationConfig(
            retry_max_attempts=1, retry_delays=[0.0],
        )
        dispatcher = NotificationDispatcher(
            channels=[ch], config=config, redis_client=mock_redis,
        )

        await dispatcher.dispatch(sample_alert)

        mock_redis.lpush.assert_called_once()
        key = mock_redis.lpush.call_args[0][0]
        assert key == "notify:retry:webhook"

        # Verify the queued payload is valid JSON
        payload_json = mock_redis.lpush.call_args[0][1]
        payload = json.loads(payload_json)
        assert payload["alert_id"] == "test-alert-001"

    @pytest.mark.asyncio
    async def test_redis_queue_graceful_without_redis(self, sample_alert):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "webhook"
        ch.send.return_value = False

        config = NotificationConfig(
            retry_max_attempts=1, retry_delays=[0.0],
        )
        dispatcher = NotificationDispatcher(
            channels=[ch], config=config, redis_client=None,
        )

        # Should not raise
        results = await dispatcher.dispatch(sample_alert)
        assert results[0] == ("webhook", False)

    @pytest.mark.asyncio
    async def test_redis_queue_error_handled(self, sample_alert):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "webhook"
        ch.send.return_value = False

        mock_redis = AsyncMock()
        mock_redis.lpush.side_effect = ConnectionError("Redis down")

        config = NotificationConfig(
            retry_max_attempts=1, retry_delays=[0.0],
        )
        dispatcher = NotificationDispatcher(
            channels=[ch], config=config, redis_client=mock_redis,
        )

        # Should not raise
        results = await dispatcher.dispatch(sample_alert)
        assert results[0] == ("webhook", False)


# ── Delivery Recording ──────────────────────────────────


class TestDeliveryRecording:
    """Tests for delivery result logging."""

    @pytest.mark.asyncio
    async def test_all_channels_succeed_logs_debug(self, sample_alert):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "webhook"
        ch.send.return_value = True

        config = NotificationConfig(retry_max_attempts=1, retry_delays=[0.0])
        dispatcher = NotificationDispatcher(channels=[ch], config=config)

        with patch("src.alerts.dispatcher.logger") as mock_logger:
            await dispatcher.dispatch(sample_alert)
            mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_all_channels_fail_logs_error(self, sample_alert):
        ch = AsyncMock(spec=NotificationChannel)
        ch.name = "webhook"
        ch.send.return_value = False

        config = NotificationConfig(retry_max_attempts=1, retry_delays=[0.0])
        dispatcher = NotificationDispatcher(channels=[ch], config=config)

        with patch("src.alerts.dispatcher.logger") as mock_logger:
            await dispatcher.dispatch(sample_alert)
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_partial_failure_logs_warning(self, sample_alert):
        ch1 = AsyncMock(spec=NotificationChannel)
        ch1.name = "webhook"
        ch1.send.return_value = True

        ch2 = AsyncMock(spec=NotificationChannel)
        ch2.name = "slack"
        ch2.send.return_value = False

        config = NotificationConfig(retry_max_attempts=1, retry_delays=[0.0])
        dispatcher = NotificationDispatcher(
            channels=[ch1, ch2], config=config,
        )

        with patch("src.alerts.dispatcher.logger") as mock_logger:
            await dispatcher.dispatch(sample_alert)
            mock_logger.warning.assert_called()
