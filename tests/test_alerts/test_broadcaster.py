"""Tests for AlertBroadcaster — connection management, filtering, and publish."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.alerts.broadcaster import CHANNEL_NAME, AlertBroadcaster, ClientConnection
from src.alerts.schemas import Alert


def _make_alert(
    alert_id: str = "alert_001",
    theme_id: str = "theme_abc",
    severity: str = "warning",
) -> Alert:
    return Alert(
        alert_id=alert_id,
        theme_id=theme_id,
        trigger_type="volume_surge",
        severity=severity,
        title="Volume surge detected",
        message="Theme volume z-score exceeds threshold",
        created_at=datetime(2026, 2, 7, 10, 0, 0, tzinfo=timezone.utc),
    )


def _mock_ws(state="connected"):
    """Create a mock WebSocket."""
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


# ── Connection Management ────────────────────────────────


class TestConnectionManagement:
    """Test connect/disconnect and max connections."""

    def test_connect_registers_client(self):
        broadcaster = AlertBroadcaster(max_connections=10)
        ws = _mock_ws()

        result = broadcaster.connect(ws, severity="critical")

        assert result is True
        assert broadcaster.active_connections == 1

    def test_connect_with_filters(self):
        broadcaster = AlertBroadcaster(max_connections=10)
        ws = _mock_ws()

        broadcaster.connect(ws, severity="warning", theme_id="theme_xyz")

        assert broadcaster.active_connections == 1
        client = broadcaster._clients[ws]
        assert client.severity == "warning"
        assert client.theme_id == "theme_xyz"

    def test_connect_no_filters(self):
        broadcaster = AlertBroadcaster(max_connections=10)
        ws = _mock_ws()

        broadcaster.connect(ws)

        client = broadcaster._clients[ws]
        assert client.severity is None
        assert client.theme_id is None

    def test_disconnect_removes_client(self):
        broadcaster = AlertBroadcaster(max_connections=10)
        ws = _mock_ws()
        broadcaster.connect(ws)

        broadcaster.disconnect(ws)

        assert broadcaster.active_connections == 0

    def test_disconnect_idempotent(self):
        broadcaster = AlertBroadcaster(max_connections=10)
        ws = _mock_ws()

        # Disconnect without connect — should not raise
        broadcaster.disconnect(ws)
        assert broadcaster.active_connections == 0

    def test_max_connections_rejected(self):
        broadcaster = AlertBroadcaster(max_connections=2)
        ws1, ws2, ws3 = _mock_ws(), _mock_ws(), _mock_ws()

        assert broadcaster.connect(ws1) is True
        assert broadcaster.connect(ws2) is True
        assert broadcaster.connect(ws3) is False
        assert broadcaster.active_connections == 2

    def test_disconnect_frees_slot(self):
        broadcaster = AlertBroadcaster(max_connections=1)
        ws1, ws2 = _mock_ws(), _mock_ws()

        broadcaster.connect(ws1)
        broadcaster.disconnect(ws1)

        assert broadcaster.connect(ws2) is True
        assert broadcaster.active_connections == 1


# ── Filter Matching ──────────────────────────────────────


class TestFilterMatching:
    """Test _matches_filters static method."""

    def test_no_filters_matches_everything(self):
        client = ClientConnection(ws=_mock_ws())
        assert AlertBroadcaster._matches_filters(client, "warning", "theme_abc") is True

    def test_severity_filter_matches(self):
        client = ClientConnection(ws=_mock_ws(), severity="critical")
        assert AlertBroadcaster._matches_filters(client, "critical", "theme_abc") is True

    def test_severity_filter_rejects(self):
        client = ClientConnection(ws=_mock_ws(), severity="critical")
        assert AlertBroadcaster._matches_filters(client, "warning", "theme_abc") is False

    def test_theme_id_filter_matches(self):
        client = ClientConnection(ws=_mock_ws(), theme_id="theme_abc")
        assert AlertBroadcaster._matches_filters(client, "warning", "theme_abc") is True

    def test_theme_id_filter_rejects(self):
        client = ClientConnection(ws=_mock_ws(), theme_id="theme_abc")
        assert AlertBroadcaster._matches_filters(client, "warning", "theme_xyz") is False

    def test_both_filters_match(self):
        client = ClientConnection(ws=_mock_ws(), severity="critical", theme_id="theme_abc")
        assert AlertBroadcaster._matches_filters(client, "critical", "theme_abc") is True

    def test_both_filters_severity_rejects(self):
        client = ClientConnection(ws=_mock_ws(), severity="critical", theme_id="theme_abc")
        assert AlertBroadcaster._matches_filters(client, "warning", "theme_abc") is False

    def test_both_filters_theme_rejects(self):
        client = ClientConnection(ws=_mock_ws(), severity="critical", theme_id="theme_abc")
        assert AlertBroadcaster._matches_filters(client, "critical", "theme_xyz") is False


# ── Publish ──────────────────────────────────────────────


class TestPublish:
    """Test static publish method."""

    @pytest.mark.asyncio
    async def test_publish_sends_json(self):
        mock_redis = AsyncMock()
        alert = _make_alert()

        result = await AlertBroadcaster.publish(mock_redis, alert)

        assert result is True
        mock_redis.publish.assert_called_once()

        channel, payload = mock_redis.publish.call_args.args
        assert channel == CHANNEL_NAME
        parsed = json.loads(payload)
        assert parsed["type"] == "alert"
        assert parsed["data"]["alert_id"] == "alert_001"
        assert parsed["data"]["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_publish_failure_returns_false(self):
        mock_redis = AsyncMock()
        mock_redis.publish.side_effect = ConnectionError("Redis down")
        alert = _make_alert()

        result = await AlertBroadcaster.publish(mock_redis, alert)

        assert result is False


# ── Dispatch Message ─────────────────────────────────────


class TestDispatchMessage:
    """Test _dispatch_message internal method."""

    @pytest.mark.asyncio
    async def test_sends_to_matching_client(self):
        broadcaster = AlertBroadcaster()
        ws = _mock_ws()
        broadcaster.connect(ws)

        payload = json.dumps({
            "type": "alert",
            "data": {"severity": "warning", "theme_id": "t1"},
        })
        await broadcaster._dispatch_message(payload)

        ws.send_text.assert_called_once()
        sent = json.loads(ws.send_text.call_args.args[0])
        assert sent["type"] == "alert"

    @pytest.mark.asyncio
    async def test_filters_non_matching_client(self):
        broadcaster = AlertBroadcaster()
        ws = _mock_ws()
        broadcaster.connect(ws, severity="critical")

        payload = json.dumps({
            "type": "alert",
            "data": {"severity": "info", "theme_id": "t1"},
        })
        await broadcaster._dispatch_message(payload)

        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnects_failed_client(self):
        broadcaster = AlertBroadcaster()
        ws = _mock_ws()
        ws.send_text.side_effect = RuntimeError("Connection lost")
        broadcaster.connect(ws)

        payload = json.dumps({
            "type": "alert",
            "data": {"severity": "warning", "theme_id": "t1"},
        })
        await broadcaster._dispatch_message(payload)

        assert broadcaster.active_connections == 0

    @pytest.mark.asyncio
    async def test_handles_bytes_payload(self):
        broadcaster = AlertBroadcaster()
        ws = _mock_ws()
        broadcaster.connect(ws)

        payload = json.dumps({
            "type": "alert",
            "data": {"severity": "warning", "theme_id": "t1"},
        }).encode("utf-8")
        await broadcaster._dispatch_message(payload)

        ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        broadcaster = AlertBroadcaster()
        ws = _mock_ws()
        broadcaster.connect(ws)

        await broadcaster._dispatch_message("not-json{{{")

        ws.send_text.assert_not_called()
        assert broadcaster.active_connections == 1  # Client not disconnected

    @pytest.mark.asyncio
    async def test_multi_client_selective_delivery(self):
        broadcaster = AlertBroadcaster()
        ws_critical = _mock_ws()
        ws_warning = _mock_ws()
        ws_all = _mock_ws()

        broadcaster.connect(ws_critical, severity="critical")
        broadcaster.connect(ws_warning, severity="warning")
        broadcaster.connect(ws_all)

        payload = json.dumps({
            "type": "alert",
            "data": {"severity": "warning", "theme_id": "t1"},
        })
        await broadcaster._dispatch_message(payload)

        ws_critical.send_text.assert_not_called()
        ws_warning.send_text.assert_called_once()
        ws_all.send_text.assert_called_once()
