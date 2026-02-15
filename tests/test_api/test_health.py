"""Tests for comprehensive health endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_database, get_embedding_service


def _mock_embedding_service():
    """Create a mock embedding service."""
    service = MagicMock()
    service.is_model_initialized = MagicMock(return_value=False)
    service.is_cache_available = AsyncMock(return_value=True)
    service.get_stats = MagicMock(return_value={"requests": 0})
    return service


def _mock_db(healthy: bool = True):
    """Create a mock database."""
    db = AsyncMock()
    if healthy:
        db.health_check = AsyncMock(return_value=True)
    else:
        db.health_check = AsyncMock(side_effect=Exception("Connection refused"))
    return db


def _make_client(db_healthy: bool = True, redis_healthy: bool = True):
    """Create a test client with configurable health states."""
    app = create_app()

    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    app.dependency_overrides[get_database] = lambda: _mock_db(db_healthy)

    # Patch redis.asyncio.from_url in health module to return our mock client
    mock_redis = AsyncMock()
    if redis_healthy:
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.xlen = AsyncMock(return_value=100)

        def _fake_xinfo_groups(stream):
            groups_by_stream = {
                "embedding_queue": [{"name": "embedding_workers", "consumers": 1, "pel-count": 2, "entries-read": 98, "lag": 0}],
                "sentiment_queue": [{"name": "sentiment_workers", "consumers": 1, "pel-count": 0, "entries-read": 100, "lag": 0}],
                "clustering_queue": [{"name": "clustering_workers", "consumers": 1, "pel-count": 5, "entries-read": 50, "lag": 10}],
            }
            return groups_by_stream.get(stream, [])

        mock_redis.xinfo_groups = AsyncMock(side_effect=_fake_xinfo_groups)
        mock_redis.aclose = AsyncMock()
    else:
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection refused"))
        mock_redis.aclose = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with TestClient(app) as client:
            yield client

    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Test /health endpoint with infrastructure checks."""

    def test_all_healthy(self):
        """DB + Redis healthy -> status=healthy, components present."""
        for client in _make_client(db_healthy=True, redis_healthy=True):
            resp = client.get("/health")
            assert resp.status_code == 200

            data = resp.json()
            assert data["status"] == "healthy"
            assert data["version"] == "0.1.0"
            assert "database" in data["components"]
            assert data["components"]["database"]["status"] == "healthy"
            assert "redis" in data["components"]
            assert data["components"]["redis"]["status"] == "healthy"
            assert "embedding_queue" in data["queue_depths"]
            eq = data["queue_depths"]["embedding_queue"]
            assert eq["pending"] == 2  # lag(0) + pel-count(2)
            assert eq["processed"] == 96  # entries-read(98) - pel-count(2)

            sq = data["queue_depths"]["sentiment_queue"]
            assert sq["pending"] == 0  # lag(0) + pel-count(0)
            assert sq["processed"] == 100  # entries-read(100) - pel-count(0)

            cq = data["queue_depths"]["clustering_queue"]
            assert cq["pending"] == 15  # lag(10) + pel-count(5)
            assert cq["processed"] == 45  # entries-read(50) - pel-count(5)

    def test_db_unhealthy(self):
        """DB fails -> status=unhealthy."""
        for client in _make_client(db_healthy=False, redis_healthy=True):
            resp = client.get("/health")
            assert resp.status_code == 200

            data = resp.json()
            assert data["status"] == "unhealthy"
            assert data["components"]["database"]["status"] == "unhealthy"

    def test_redis_unhealthy(self):
        """Redis fails -> status=degraded (service still works without cache)."""
        for client in _make_client(db_healthy=True, redis_healthy=False):
            resp = client.get("/health")
            assert resp.status_code == 200

            data = resp.json()
            assert data["status"] == "degraded"
            assert data["components"]["database"]["status"] == "healthy"
            assert data["components"]["redis"]["status"] == "unhealthy"

    def test_backward_compatible_fields(self):
        """Original fields (models_loaded, cache_available, etc.) still present."""
        for client in _make_client(db_healthy=True, redis_healthy=True):
            resp = client.get("/health")
            data = resp.json()

            assert "models_loaded" in data
            assert "cache_available" in data
            assert "gpu_available" in data
            assert "service_stats" in data
