"""Integration tests for the /ws/alerts WebSocket endpoint."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.alerts.broadcaster import AlertBroadcaster
from src.alerts.schemas import Alert
from src.api.app import create_app
from src.api.routes.ws_alerts import set_broadcaster


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


@pytest.fixture
def broadcaster():
    """A real AlertBroadcaster (no Redis needed — we test fan-out directly)."""
    return AlertBroadcaster(max_connections=5, heartbeat_interval=300)


@pytest.fixture
def app_enabled(broadcaster):
    """FastAPI app with ws_alerts_enabled=True."""
    app = create_app()
    set_broadcaster(broadcaster)

    with patch("src.api.routes.ws_alerts.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.ws_alerts_enabled = True
        settings.api_keys = None  # dev mode
        yield app

    set_broadcaster(None)
    app.dependency_overrides.clear()


@pytest.fixture
def app_disabled():
    """FastAPI app with ws_alerts_enabled=False."""
    app = create_app()
    set_broadcaster(None)

    with patch("src.api.routes.ws_alerts.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.ws_alerts_enabled = False
        settings.api_keys = None
        yield app

    app.dependency_overrides.clear()


@pytest.fixture
def app_auth_required(broadcaster):
    """FastAPI app with API key auth enforced."""
    app = create_app()
    set_broadcaster(broadcaster)

    with patch("src.api.routes.ws_alerts.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.ws_alerts_enabled = True
        settings.api_keys = "secret-key-1,secret-key-2"
        yield app

    set_broadcaster(None)
    app.dependency_overrides.clear()


# ── Connection Tests ─────────────────────────────────────


class TestWebSocketConnection:
    """Test basic WebSocket connection and alert delivery."""

    def test_connect_and_receive_alert(self, app_enabled, broadcaster):
        client = TestClient(app_enabled)

        with client.websocket_connect("/ws/alerts") as ws:
            assert broadcaster.active_connections == 1

            # Simulate an alert being dispatched
            alert = _make_alert()
            payload = json.dumps({
                "type": "alert",
                "data": alert.to_dict(),
            })

            # Directly call _dispatch_message (simulates Redis pub/sub delivery)
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(broadcaster._dispatch_message(payload))
            loop.close()

            data = ws.receive_json()
            assert data["type"] == "alert"
            assert data["data"]["alert_id"] == "alert_001"
            assert data["data"]["severity"] == "warning"

    def test_disconnect_cleans_up(self, app_enabled, broadcaster):
        client = TestClient(app_enabled)

        with client.websocket_connect("/ws/alerts"):
            assert broadcaster.active_connections == 1

        # After context exit, disconnect handler fires
        assert broadcaster.active_connections == 0

    def test_client_ping_pong(self, app_enabled, broadcaster):
        client = TestClient(app_enabled)

        with client.websocket_connect("/ws/alerts") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"


# ── Filter Tests ─────────────────────────────────────────


class TestWebSocketFiltering:
    """Test severity and theme_id filtering via query params."""

    def test_severity_filter_receives_matching(self, app_enabled, broadcaster):
        client = TestClient(app_enabled)

        with client.websocket_connect("/ws/alerts?severity=critical") as ws:
            # Send critical alert
            payload = json.dumps({
                "type": "alert",
                "data": {"severity": "critical", "theme_id": "t1", "alert_id": "a1"},
            })
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(broadcaster._dispatch_message(payload))
            loop.close()

            data = ws.receive_json()
            assert data["data"]["severity"] == "critical"

    def test_severity_filter_blocks_non_matching(self, app_enabled, broadcaster):
        client = TestClient(app_enabled)

        with client.websocket_connect("/ws/alerts?severity=critical") as ws:
            # Send warning alert — should NOT be received
            payload = json.dumps({
                "type": "alert",
                "data": {"severity": "warning", "theme_id": "t1", "alert_id": "a2"},
            })
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(broadcaster._dispatch_message(payload))
            loop.close()

            # Send a matching alert so we can verify the first wasn't received
            payload2 = json.dumps({
                "type": "alert",
                "data": {"severity": "critical", "theme_id": "t2", "alert_id": "a3"},
            })
            loop = asyncio.new_event_loop()
            loop.run_until_complete(broadcaster._dispatch_message(payload2))
            loop.close()

            data = ws.receive_json()
            # The first message we receive should be the critical one, not the warning
            assert data["data"]["alert_id"] == "a3"

    def test_theme_id_filter(self, app_enabled, broadcaster):
        client = TestClient(app_enabled)

        with client.websocket_connect("/ws/alerts?theme_id=theme_xyz") as ws:
            # Non-matching
            payload1 = json.dumps({
                "type": "alert",
                "data": {"severity": "warning", "theme_id": "theme_other", "alert_id": "a1"},
            })
            # Matching
            payload2 = json.dumps({
                "type": "alert",
                "data": {"severity": "warning", "theme_id": "theme_xyz", "alert_id": "a2"},
            })

            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(broadcaster._dispatch_message(payload1))
            loop.run_until_complete(broadcaster._dispatch_message(payload2))
            loop.close()

            data = ws.receive_json()
            assert data["data"]["alert_id"] == "a2"


# ── Auth Tests ───────────────────────────────────────────


class TestWebSocketAuth:
    """Test API key authentication for WebSocket connections."""

    def test_valid_api_key_connects(self, app_auth_required, broadcaster):
        client = TestClient(app_auth_required)

        with client.websocket_connect("/ws/alerts?api_key=secret-key-1") as ws:
            assert broadcaster.active_connections == 1

    def test_invalid_api_key_rejected(self, app_auth_required):
        client = TestClient(app_auth_required)

        with pytest.raises(Exception):
            with client.websocket_connect("/ws/alerts?api_key=wrong-key"):
                pass

    def test_missing_api_key_rejected(self, app_auth_required):
        client = TestClient(app_auth_required)

        with pytest.raises(Exception):
            with client.websocket_connect("/ws/alerts"):
                pass

    def test_dev_mode_no_key_needed(self, app_enabled, broadcaster):
        """When api_keys is None (dev mode), connections are allowed."""
        client = TestClient(app_enabled)

        with client.websocket_connect("/ws/alerts") as ws:
            assert broadcaster.active_connections == 1


# ── Feature Flag Tests ───────────────────────────────────


class TestWebSocketFeatureFlag:
    """Test ws_alerts_enabled feature flag."""

    def test_disabled_rejects_connection(self, app_disabled):
        client = TestClient(app_disabled)

        with pytest.raises(Exception):
            with client.websocket_connect("/ws/alerts"):
                pass


# ── Max Connections ──────────────────────────────────────


class TestWebSocketMaxConnections:
    """Test max connections enforcement."""

    def test_max_connections_exceeded(self, app_enabled, broadcaster):
        """Broadcaster has max_connections=5 in fixture.

        When max connections is reached, the server accepts the WS upgrade
        then immediately sends a close frame with code 1008 (Policy Violation).
        """
        client = TestClient(app_enabled)

        # Fill up connections by connecting directly to broadcaster
        fake_clients = [AsyncMock() for _ in range(5)]
        for fc in fake_clients:
            broadcaster.connect(fc)

        assert broadcaster.active_connections == 5

        # Connection is accepted then closed — Starlette TestClient
        # raises WebSocketDisconnect when server closes the connection
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/alerts") as ws:
                # Try to receive — should get a close frame
                ws.receive_json()

        # Clean up
        for fc in fake_clients:
            broadcaster.disconnect(fc)
