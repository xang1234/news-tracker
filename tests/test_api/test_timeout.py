"""Tests for request timeout middleware."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware.timeout import TimeoutMiddleware


def _create_test_app(timeout: float = 1.0) -> FastAPI:
    """Create a minimal FastAPI app with timeout middleware for testing."""
    app = FastAPI()
    app.add_middleware(TimeoutMiddleware, timeout_seconds=timeout)

    @app.get("/fast")
    async def fast():
        return {"status": "ok"}

    @app.get("/slow")
    async def slow():
        await asyncio.sleep(10)
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        # Health should be excluded from timeout
        return {"status": "healthy"}

    return app


class TestTimeoutMiddleware:
    """Tests for TimeoutMiddleware behavior."""

    def test_fast_request_succeeds(self):
        """Normal requests within timeout should succeed."""
        app = _create_test_app(timeout=5.0)
        client = TestClient(app)
        response = client.get("/fast")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_slow_request_returns_504(self):
        """Requests exceeding timeout should return 504."""
        app = _create_test_app(timeout=0.1)
        client = TestClient(app)
        response = client.get("/slow")
        assert response.status_code == 504
        data = response.json()
        assert "timed out" in data["detail"]
        assert data["timeout_seconds"] == 0.1

    def test_health_excluded_from_timeout(self):
        """Health endpoint should bypass timeout enforcement."""
        app = _create_test_app(timeout=5.0)
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
