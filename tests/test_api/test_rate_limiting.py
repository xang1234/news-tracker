"""Tests for API rate limiting configuration."""

from types import SimpleNamespace
from unittest.mock import patch

from src.api.rate_limit import _get_rate_limit_key, create_limiter


class TestRateLimitKeyExtraction:
    """Tests for rate limit key function."""

    def test_uses_api_key_when_present(self):
        """Should use X-API-KEY header as rate limit key."""
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/embed",
            "headers": [(b"x-api-key", b"test-key-123")],
        }
        request = Request(scope)
        key = _get_rate_limit_key(request)
        assert key == "test-key-123"

    def test_falls_back_to_ip(self):
        """Should fall back to client IP when no API key."""
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/embed",
            "headers": [],
            "client": ("192.168.1.100", 12345),
        }
        request = Request(scope)
        key = _get_rate_limit_key(request)
        assert key == "192.168.1.100"


class TestLimiterConfiguration:
    """Tests for limiter construction."""

    def test_create_limiter_disables_slowapi_checks_when_feature_is_off(self):
        settings = SimpleNamespace(
            rate_limit_enabled=False,
            rate_limit_default="60/minute",
            redis_url="redis://localhost:6379/0",
        )

        with patch("src.api.rate_limit.get_settings", return_value=settings):
            limiter = create_limiter()

        assert limiter.enabled is False
