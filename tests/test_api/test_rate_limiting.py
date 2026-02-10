"""Tests for API rate limiting configuration."""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from src.api.rate_limit import _get_rate_limit_key


class TestRateLimitKeyExtraction:
    """Tests for rate limit key function."""

    def test_uses_api_key_when_present(self):
        """Should use X-API-KEY header as rate limit key."""
        from starlette.requests import Request
        from starlette.datastructures import Headers

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
