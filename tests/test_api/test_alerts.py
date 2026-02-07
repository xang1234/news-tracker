"""Tests for alert REST API endpoints."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.alerts.repository import AlertRepository
from src.alerts.schemas import Alert
from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_alert_repository


def _make_alert(
    alert_id: str = "alert_001",
    theme_id: str = "theme_abc123",
    trigger_type: str = "volume_surge",
    severity: str = "warning",
    acknowledged: bool = False,
    **kwargs,
) -> Alert:
    """Helper to create an Alert with sensible defaults."""
    return Alert(
        alert_id=alert_id,
        theme_id=theme_id,
        trigger_type=trigger_type,
        severity=severity,
        title=kwargs.pop("title", "Volume surge detected"),
        message=kwargs.pop("message", "Theme theme_abc123 volume z-score 3.5 exceeds threshold"),
        trigger_data=kwargs.pop("trigger_data", {"volume_zscore": 3.5, "threshold": 3.0}),
        acknowledged=acknowledged,
        created_at=kwargs.pop(
            "created_at", datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc)
        ),
        **kwargs,
    )


@pytest.fixture
def mock_alert_repo():
    """Mock AlertRepository."""
    repo = AsyncMock(spec=AlertRepository)
    repo.get_recent = AsyncMock(return_value=[])
    repo.acknowledge = AsyncMock(return_value=True)
    repo.get_by_id = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def client(mock_alert_repo):
    """FastAPI TestClient with dependency overrides for alerts."""
    app = create_app()

    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_alert_repository] = lambda: mock_alert_repo

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── GET /alerts ──────────────────────────────────────────


class TestListAlerts:
    """Tests for the list alerts endpoint."""

    def test_empty_list(self, client, mock_alert_repo):
        mock_alert_repo.get_recent.return_value = []
        resp = client.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alerts"] == []
        assert data["total"] == 0
        assert "latency_ms" in data

    def test_returns_alerts(self, client, mock_alert_repo):
        alerts = [
            _make_alert("a1", severity="critical"),
            _make_alert("a2", severity="warning"),
        ]
        mock_alert_repo.get_recent.return_value = alerts

        resp = client.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["alerts"][0]["alert_id"] == "a1"
        assert data["alerts"][1]["alert_id"] == "a2"

    def test_alert_fields(self, client, mock_alert_repo):
        mock_alert_repo.get_recent.return_value = [_make_alert()]

        resp = client.get("/alerts")
        item = resp.json()["alerts"][0]
        assert item["alert_id"] == "alert_001"
        assert item["theme_id"] == "theme_abc123"
        assert item["trigger_type"] == "volume_surge"
        assert item["severity"] == "warning"
        assert item["title"] == "Volume surge detected"
        assert item["acknowledged"] is False
        assert item["trigger_data"] == {"volume_zscore": 3.5, "threshold": 3.0}
        assert "created_at" in item

    def test_severity_filter(self, client, mock_alert_repo):
        mock_alert_repo.get_recent.return_value = []

        resp = client.get("/alerts?severity=critical")
        assert resp.status_code == 200
        mock_alert_repo.get_recent.assert_called_once_with(
            severity="critical",
            trigger_type=None,
            theme_id=None,
            acknowledged=None,
            limit=50,
            offset=0,
        )

    def test_trigger_type_filter(self, client, mock_alert_repo):
        mock_alert_repo.get_recent.return_value = []

        resp = client.get("/alerts?trigger_type=sentiment_velocity")
        assert resp.status_code == 200
        mock_alert_repo.get_recent.assert_called_once_with(
            severity=None,
            trigger_type="sentiment_velocity",
            theme_id=None,
            acknowledged=None,
            limit=50,
            offset=0,
        )

    def test_theme_id_filter(self, client, mock_alert_repo):
        mock_alert_repo.get_recent.return_value = []

        resp = client.get("/alerts?theme_id=theme_xyz")
        assert resp.status_code == 200
        mock_alert_repo.get_recent.assert_called_once_with(
            severity=None,
            trigger_type=None,
            theme_id="theme_xyz",
            acknowledged=None,
            limit=50,
            offset=0,
        )

    def test_acknowledged_filter(self, client, mock_alert_repo):
        mock_alert_repo.get_recent.return_value = []

        resp = client.get("/alerts?acknowledged=true")
        assert resp.status_code == 200
        mock_alert_repo.get_recent.assert_called_once_with(
            severity=None,
            trigger_type=None,
            theme_id=None,
            acknowledged=True,
            limit=50,
            offset=0,
        )

    def test_multiple_filters(self, client, mock_alert_repo):
        mock_alert_repo.get_recent.return_value = []

        resp = client.get("/alerts?severity=warning&trigger_type=volume_surge&theme_id=t1")
        assert resp.status_code == 200
        mock_alert_repo.get_recent.assert_called_once_with(
            severity="warning",
            trigger_type="volume_surge",
            theme_id="t1",
            acknowledged=None,
            limit=50,
            offset=0,
        )

    def test_pagination(self, client, mock_alert_repo):
        mock_alert_repo.get_recent.return_value = []

        resp = client.get("/alerts?limit=10&offset=20")
        assert resp.status_code == 200
        mock_alert_repo.get_recent.assert_called_once_with(
            severity=None,
            trigger_type=None,
            theme_id=None,
            acknowledged=None,
            limit=10,
            offset=20,
        )

    def test_invalid_severity(self, client, mock_alert_repo):
        resp = client.get("/alerts?severity=bogus")
        assert resp.status_code == 422
        assert "Invalid severity" in resp.json()["detail"]

    def test_invalid_trigger_type(self, client, mock_alert_repo):
        resp = client.get("/alerts?trigger_type=bogus")
        assert resp.status_code == 422
        assert "Invalid trigger_type" in resp.json()["detail"]

    def test_server_error(self, client, mock_alert_repo):
        mock_alert_repo.get_recent.side_effect = RuntimeError("DB down")

        resp = client.get("/alerts")
        assert resp.status_code == 500
        assert "Failed to list alerts" in resp.json()["detail"]


# ── PATCH /alerts/{alert_id}/acknowledge ─────────────────


class TestAcknowledgeAlert:
    """Tests for the acknowledge alert endpoint."""

    def test_acknowledge_success(self, client, mock_alert_repo):
        mock_alert_repo.acknowledge.return_value = True

        resp = client.patch("/alerts/alert_001/acknowledge")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alert_id"] == "alert_001"
        assert data["acknowledged"] is True
        assert "latency_ms" in data
        mock_alert_repo.acknowledge.assert_called_once_with("alert_001")

    def test_acknowledge_not_found(self, client, mock_alert_repo):
        mock_alert_repo.acknowledge.return_value = False

        resp = client.patch("/alerts/nonexistent/acknowledge")
        assert resp.status_code == 404
        assert "not found or already acknowledged" in resp.json()["detail"]

    def test_acknowledge_already_acknowledged(self, client, mock_alert_repo):
        # Repository returns False when alert is already acknowledged
        mock_alert_repo.acknowledge.return_value = False

        resp = client.patch("/alerts/alert_001/acknowledge")
        assert resp.status_code == 404
        assert "already acknowledged" in resp.json()["detail"]

    def test_acknowledge_server_error(self, client, mock_alert_repo):
        mock_alert_repo.acknowledge.side_effect = RuntimeError("DB down")

        resp = client.patch("/alerts/alert_001/acknowledge")
        assert resp.status_code == 500
        assert "Failed to acknowledge" in resp.json()["detail"]
