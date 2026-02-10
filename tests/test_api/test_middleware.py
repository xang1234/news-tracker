"""Tests for request/correlation ID middleware."""

import re

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_database, get_embedding_service

# UUID v4 regex pattern
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _make_client():
    """Create a test client with minimal dependency overrides."""
    from unittest.mock import AsyncMock, MagicMock

    app = create_app()

    # Bypass auth
    app.dependency_overrides[verify_api_key] = lambda: "test-key"

    # Mock embedding service for /health
    mock_service = MagicMock()
    mock_service.is_model_initialized = MagicMock(return_value=False)
    mock_service.is_cache_available = AsyncMock(return_value=False)
    mock_service.get_stats = MagicMock(return_value={})
    app.dependency_overrides[get_embedding_service] = lambda: mock_service

    # Mock database for /health
    mock_db = AsyncMock()
    mock_db.health_check = AsyncMock(return_value=True)
    app.dependency_overrides[get_database] = lambda: mock_db

    return app


class TestCorrelationIdMiddleware:
    """Test X-Request-ID middleware behavior."""

    def test_generates_uuid_when_no_header(self):
        """Request without header gets a UUID v4 in response."""
        app = _make_client()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            request_id = resp.headers.get("X-Request-ID")
            assert request_id is not None
            assert UUID_RE.match(request_id), f"Expected UUID v4, got: {request_id}"

    def test_echoes_custom_request_id(self):
        """Request with X-Request-ID header echoes it back."""
        app = _make_client()
        with TestClient(app) as client:
            resp = client.get("/health", headers={"X-Request-ID": "custom-id-123"})
            assert resp.status_code == 200
            assert resp.headers.get("X-Request-ID") == "custom-id-123"

    def test_echoes_correlation_id(self):
        """Request with X-Correlation-ID header echoes it back as X-Request-ID."""
        app = _make_client()
        with TestClient(app) as client:
            resp = client.get("/health", headers={"X-Correlation-ID": "corr-456"})
            assert resp.status_code == 200
            assert resp.headers.get("X-Request-ID") == "corr-456"

    def test_request_id_takes_priority_over_correlation_id(self):
        """X-Request-ID takes priority over X-Correlation-ID."""
        app = _make_client()
        with TestClient(app) as client:
            resp = client.get(
                "/health",
                headers={
                    "X-Request-ID": "req-id",
                    "X-Correlation-ID": "corr-id",
                },
            )
            assert resp.status_code == 200
            assert resp.headers.get("X-Request-ID") == "req-id"
