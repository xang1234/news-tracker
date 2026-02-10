"""Tests for securities REST API endpoints."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_security_master_repository
from src.security_master.schemas import Security


def _make_security(
    ticker: str = "NVDA",
    exchange: str = "US",
    name: str = "NVIDIA Corporation",
    is_active: bool = True,
    **kwargs,
) -> Security:
    """Helper to create a Security with sensible defaults."""
    return Security(
        ticker=ticker,
        exchange=exchange,
        name=name,
        aliases=kwargs.pop("aliases", ["NVIDIA", "Jensen Huang's company"]),
        sector=kwargs.pop("sector", "Semiconductors"),
        country=kwargs.pop("country", "US"),
        currency=kwargs.pop("currency", "USD"),
        is_active=is_active,
        created_at=kwargs.pop("created_at", datetime(2026, 1, 1, tzinfo=timezone.utc)),
        updated_at=kwargs.pop("updated_at", datetime(2026, 2, 5, tzinfo=timezone.utc)),
        **kwargs,
    )


@pytest.fixture
def mock_security_repo():
    """Mock SecurityMasterRepository."""
    repo = AsyncMock()
    repo.list_securities = AsyncMock(return_value=([], 0))
    repo.upsert = AsyncMock(return_value=None)
    repo.get_by_ticker = AsyncMock(return_value=None)
    repo.deactivate = AsyncMock(return_value=False)
    return repo


@pytest.fixture
def client(mock_security_repo):
    """FastAPI TestClient with dependency overrides for securities."""
    app = create_app()

    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_security_master_repository] = lambda: mock_security_repo

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── Feature gate ─────────────────────────────────


class TestFeatureGate:
    """Securities endpoints return 404 when security_master_enabled=false."""

    @patch("src.api.routes.securities._get_settings")
    def test_list_disabled(self, mock_settings, client):
        settings = MagicMock()
        settings.security_master_enabled = False
        mock_settings.return_value = settings

        resp = client.get("/securities")
        assert resp.status_code == 404
        assert "security_master_enabled" in resp.json()["detail"]

    @patch("src.api.routes.securities._get_settings")
    def test_create_disabled(self, mock_settings, client):
        settings = MagicMock()
        settings.security_master_enabled = False
        mock_settings.return_value = settings

        resp = client.post("/securities", json={"ticker": "NVDA", "exchange": "US", "name": "NVIDIA"})
        assert resp.status_code == 404


# ── GET /securities ──────────────────────────────


class TestListSecurities:
    """Tests for the list securities endpoint."""

    @patch("src.api.routes.securities._get_settings")
    def test_empty_list(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.list_securities.return_value = ([], 0)

        resp = client.get("/securities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["securities"] == []
        assert data["total"] == 0
        assert data["has_more"] is False
        assert "latency_ms" in data

    @patch("src.api.routes.securities._get_settings")
    def test_returns_securities(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.list_securities.return_value = (
            [_make_security("NVDA"), _make_security("AMD", name="AMD Inc")],
            2,
        )

        resp = client.get("/securities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["securities"][0]["ticker"] == "NVDA"
        assert data["securities"][1]["ticker"] == "AMD"

    @patch("src.api.routes.securities._get_settings")
    def test_security_fields(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.list_securities.return_value = ([_make_security()], 1)

        resp = client.get("/securities")
        item = resp.json()["securities"][0]
        assert item["ticker"] == "NVDA"
        assert item["exchange"] == "US"
        assert item["name"] == "NVIDIA Corporation"
        assert item["sector"] == "Semiconductors"
        assert item["is_active"] is True
        assert "created_at" in item

    @patch("src.api.routes.securities._get_settings")
    def test_search_filter(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.list_securities.return_value = ([], 0)

        resp = client.get("/securities?search=nvidia&active_only=true&exchange=US")
        assert resp.status_code == 200
        mock_security_repo.list_securities.assert_called_once_with(
            search="nvidia",
            active_only=True,
            exchange="US",
            limit=50,
            offset=0,
        )

    @patch("src.api.routes.securities._get_settings")
    def test_pagination(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.list_securities.return_value = ([], 100)

        resp = client.get("/securities?limit=10&offset=20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_more"] is True

    @patch("src.api.routes.securities._get_settings")
    def test_server_error(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.list_securities.side_effect = RuntimeError("DB down")

        resp = client.get("/securities")
        assert resp.status_code == 500


# ── POST /securities ─────────────────────────────


class TestCreateSecurity:
    """Tests for the create security endpoint."""

    @patch("src.api.routes.securities._get_settings")
    def test_create_success(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        created = _make_security()
        mock_security_repo.get_by_ticker.return_value = created

        resp = client.post(
            "/securities",
            json={"ticker": "nvda", "exchange": "us", "name": "NVIDIA Corporation"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["ticker"] == "NVDA"
        assert data["name"] == "NVIDIA Corporation"
        mock_security_repo.upsert.assert_called_once()

    @patch("src.api.routes.securities._get_settings")
    def test_create_with_all_fields(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        created = _make_security(sector="Tech", country="US", currency="USD")
        mock_security_repo.get_by_ticker.return_value = created

        resp = client.post(
            "/securities",
            json={
                "ticker": "NVDA",
                "exchange": "US",
                "name": "NVIDIA Corporation",
                "aliases": ["NVIDIA"],
                "sector": "Tech",
                "country": "US",
                "currency": "USD",
            },
        )
        assert resp.status_code == 201

    @patch("src.api.routes.securities._get_settings")
    def test_create_missing_required_fields(self, mock_settings, client):
        mock_settings.return_value = MagicMock(security_master_enabled=True)

        resp = client.post("/securities", json={"ticker": "NVDA"})
        assert resp.status_code == 422

    @patch("src.api.routes.securities._get_settings")
    def test_create_refetch_fails(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.get_by_ticker.return_value = None

        resp = client.post(
            "/securities",
            json={"ticker": "NVDA", "exchange": "US", "name": "NVIDIA"},
        )
        assert resp.status_code == 500
        assert "failed" in resp.json()["detail"].lower()


# ── PUT /securities/{ticker}/{exchange} ──────────


class TestUpdateSecurity:
    """Tests for the update security endpoint."""

    @patch("src.api.routes.securities._get_settings")
    def test_update_success(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        existing = _make_security()
        updated = _make_security(name="NVIDIA Corp Updated")
        mock_security_repo.get_by_ticker.side_effect = [existing, updated]

        resp = client.put(
            "/securities/NVDA/US",
            json={"name": "NVIDIA Corp Updated"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "NVIDIA Corp Updated"

    @patch("src.api.routes.securities._get_settings")
    def test_update_not_found(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.get_by_ticker.return_value = None

        resp = client.put(
            "/securities/AAPL/US",
            json={"name": "Apple Inc"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    @patch("src.api.routes.securities._get_settings")
    def test_update_partial(self, mock_settings, client, mock_security_repo):
        """Only provided fields should be updated."""
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        existing = _make_security(sector="Semiconductors")
        updated = _make_security(sector="Semiconductors", name="NVIDIA Updated")
        mock_security_repo.get_by_ticker.side_effect = [existing, updated]

        resp = client.put(
            "/securities/NVDA/US",
            json={"name": "NVIDIA Updated"},
        )
        assert resp.status_code == 200
        # The upserted Security should preserve the original sector
        call_args = mock_security_repo.upsert.call_args[0][0]
        assert call_args.sector == "Semiconductors"
        assert call_args.name == "NVIDIA Updated"


# ── DELETE /securities/{ticker}/{exchange} ───────


class TestDeactivateSecurity:
    """Tests for the deactivate security endpoint."""

    @patch("src.api.routes.securities._get_settings")
    def test_deactivate_success(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.deactivate.return_value = True

        resp = client.delete("/securities/NVDA/US")
        assert resp.status_code == 204
        mock_security_repo.deactivate.assert_called_once_with("NVDA", "US")

    @patch("src.api.routes.securities._get_settings")
    def test_deactivate_not_found(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.deactivate.return_value = False

        resp = client.delete("/securities/AAPL/US")
        assert resp.status_code == 404
        assert "not found or already inactive" in resp.json()["detail"]

    @patch("src.api.routes.securities._get_settings")
    def test_deactivate_server_error(self, mock_settings, client, mock_security_repo):
        mock_settings.return_value = MagicMock(security_master_enabled=True)
        mock_security_repo.deactivate.side_effect = RuntimeError("DB down")

        resp = client.delete("/securities/NVDA/US")
        assert resp.status_code == 500
