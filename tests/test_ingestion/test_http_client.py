"""Tests for HTTP client infrastructure layer."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from src.ingestion.http_client import (
    APIKeyRotator,
    HTTPClient,
    HTTPClientError,
    RateLimitError,
    RetryConfig,
)


class TestAPIKeyRotator:
    """Tests for APIKeyRotator."""

    def test_from_env_var_with_multiple_keys(self):
        """Should parse comma-separated keys."""
        rotator = APIKeyRotator.from_env_var("key1,key2,key3")

        assert rotator is not None
        assert rotator.keys == ["key1", "key2", "key3"]
        assert rotator.key_count == 3

    def test_from_env_var_with_single_key(self):
        """Should handle single key without comma."""
        rotator = APIKeyRotator.from_env_var("single_key")

        assert rotator is not None
        assert rotator.keys == ["single_key"]
        assert rotator.key_count == 1

    def test_from_env_var_with_whitespace(self):
        """Should strip whitespace from keys."""
        rotator = APIKeyRotator.from_env_var("  key1  ,  key2  ,  key3  ")

        assert rotator is not None
        assert rotator.keys == ["key1", "key2", "key3"]

    def test_from_env_var_with_none(self):
        """Should return None for None input."""
        rotator = APIKeyRotator.from_env_var(None)
        assert rotator is None

    def test_from_env_var_with_empty_string(self):
        """Should return None for empty string."""
        rotator = APIKeyRotator.from_env_var("")
        assert rotator is None

    def test_from_env_var_with_only_whitespace(self):
        """Should return None for whitespace-only string."""
        rotator = APIKeyRotator.from_env_var("   ")
        assert rotator is None

    def test_from_env_var_with_empty_segments(self):
        """Should filter out empty segments."""
        rotator = APIKeyRotator.from_env_var("key1,,key2,  ,key3")

        assert rotator is not None
        assert rotator.keys == ["key1", "key2", "key3"]

    def test_get_key_sync_rotation(self):
        """Should rotate through keys in order."""
        rotator = APIKeyRotator(keys=["a", "b", "c"])

        assert rotator.get_key_sync() == "a"
        assert rotator.get_key_sync() == "b"
        assert rotator.get_key_sync() == "c"
        assert rotator.get_key_sync() == "a"  # Wraps around

    @pytest.mark.asyncio
    async def test_get_key_async_rotation(self):
        """Should rotate through keys asynchronously."""
        rotator = APIKeyRotator(keys=["x", "y", "z"])

        assert await rotator.get_key() == "x"
        assert await rotator.get_key() == "y"
        assert await rotator.get_key() == "z"
        assert await rotator.get_key() == "x"

    @pytest.mark.asyncio
    async def test_get_key_concurrent_access(self):
        """Should handle concurrent key requests safely."""
        rotator = APIKeyRotator(keys=["1", "2", "3"])

        # Request 9 keys concurrently
        tasks = [rotator.get_key() for _ in range(9)]
        results = await asyncio.gather(*tasks)

        # Each key should appear exactly 3 times
        assert results.count("1") == 3
        assert results.count("2") == 3
        assert results.count("3") == 3


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.max_backoff_seconds == 60.0
        assert config.base_delay == 1.0
        assert config.jitter_factor == 0.1

    def test_calculate_backoff_exponential(self):
        """Should calculate exponential backoff."""
        config = RetryConfig(base_delay=1.0, jitter_factor=0.0)

        # Without jitter, should be exact powers of 2
        assert config.calculate_backoff(0) == 1.0   # 1 * 2^0
        assert config.calculate_backoff(1) == 2.0   # 1 * 2^1
        assert config.calculate_backoff(2) == 4.0   # 1 * 2^2
        assert config.calculate_backoff(3) == 8.0   # 1 * 2^3

    def test_calculate_backoff_respects_max(self):
        """Should cap backoff at max_backoff_seconds."""
        config = RetryConfig(
            base_delay=1.0,
            max_backoff_seconds=5.0,
            jitter_factor=0.0,
        )

        assert config.calculate_backoff(0) == 1.0
        assert config.calculate_backoff(1) == 2.0
        assert config.calculate_backoff(2) == 4.0
        assert config.calculate_backoff(3) == 5.0  # Capped
        assert config.calculate_backoff(10) == 5.0  # Still capped

    def test_calculate_backoff_with_jitter(self):
        """Should add jitter within expected range."""
        config = RetryConfig(
            base_delay=1.0,
            jitter_factor=0.1,
            max_backoff_seconds=60.0,
        )

        # Run multiple times to verify jitter is applied
        backoffs = [config.calculate_backoff(0) for _ in range(100)]

        # Base is 1.0, jitter adds 0-0.1, so range is [1.0, 1.1)
        assert all(1.0 <= b < 1.1 for b in backoffs)
        # Should have some variation (not all identical)
        assert len(set(backoffs)) > 1

    def test_is_retryable_status_rate_limit(self):
        """Should identify 429 as retryable."""
        config = RetryConfig()
        assert config.is_retryable_status(429) is True

    def test_is_retryable_status_server_errors(self):
        """Should identify 5xx as retryable."""
        config = RetryConfig()

        assert config.is_retryable_status(500) is True
        assert config.is_retryable_status(502) is True
        assert config.is_retryable_status(503) is True
        assert config.is_retryable_status(504) is True

    def test_is_retryable_status_client_errors(self):
        """Should not retry 4xx (except 429)."""
        config = RetryConfig()

        assert config.is_retryable_status(400) is False
        assert config.is_retryable_status(401) is False
        assert config.is_retryable_status(403) is False
        assert config.is_retryable_status(404) is False

    def test_is_retryable_status_success(self):
        """Should not retry success codes."""
        config = RetryConfig()

        assert config.is_retryable_status(200) is False
        assert config.is_retryable_status(201) is False
        assert config.is_retryable_status(204) is False

    def test_is_retryable_exception_timeout(self):
        """Should retry on timeout."""
        config = RetryConfig()
        assert config.is_retryable_exception(httpx.TimeoutException("timeout")) is True

    def test_is_retryable_exception_connect_error(self):
        """Should retry on connection error."""
        config = RetryConfig()
        assert config.is_retryable_exception(httpx.ConnectError("connection failed")) is True

    def test_is_retryable_exception_read_error(self):
        """Should retry on read error."""
        config = RetryConfig()
        assert config.is_retryable_exception(httpx.ReadError("read failed")) is True

    def test_is_retryable_exception_other(self):
        """Should not retry on other exceptions."""
        config = RetryConfig()

        assert config.is_retryable_exception(ValueError("bad value")) is False
        assert config.is_retryable_exception(httpx.HTTPStatusError("error", request=None, response=None)) is False


class TestHTTPClient:
    """Tests for HTTPClient."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_success(self):
        """Should return response on successful GET."""
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={"result": "success"})
        )

        async with HTTPClient() as client:
            response = await client.get("https://api.example.com/data")

        assert response.status_code == 200
        assert response.json() == {"result": "success"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_with_params(self):
        """Should include query parameters in GET request."""
        route = respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={})
        )

        async with HTTPClient() as client:
            await client.get(
                "https://api.example.com/data",
                params={"q": "search", "limit": 10},
            )

        assert route.called
        request = route.calls.last.request
        assert "q=search" in str(request.url)
        assert "limit=10" in str(request.url)

    @pytest.mark.asyncio
    @respx.mock
    async def test_post_success(self):
        """Should return response on successful POST."""
        respx.post("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={"id": 123})
        )

        async with HTTPClient() as client:
            response = await client.post(
                "https://api.example.com/data",
                json_body={"name": "test"},
            )

        assert response.status_code == 200
        assert response.json() == {"id": 123}

    @pytest.mark.asyncio
    @respx.mock
    async def test_post_with_json_body(self):
        """Should send JSON body in POST request."""
        route = respx.post("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={})
        )

        async with HTTPClient() as client:
            await client.post(
                "https://api.example.com/data",
                json_body={"name": "test", "value": 42},
            )

        assert route.called
        request = route.calls.last.request
        assert request.headers.get("content-type") == "application/json"

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_key_in_header(self):
        """Should add API key to request header."""
        route = respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={})
        )

        rotator = APIKeyRotator(keys=["test_api_key"])

        async with HTTPClient() as client:
            await client.get(
                "https://api.example.com/data",
                api_key_rotator=rotator,
                api_key_header="Authorization",
                api_key_prefix="Bearer ",
            )

        request = route.calls.last.request
        assert request.headers.get("Authorization") == "Bearer test_api_key"

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_key_in_query_param(self):
        """Should add API key to query parameter."""
        route = respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={})
        )

        rotator = APIKeyRotator(keys=["my_secret_key"])

        async with HTTPClient() as client:
            await client.get(
                "https://api.example.com/data",
                api_key_rotator=rotator,
                api_key_param="api_token",
            )

        request = route.calls.last.request
        assert "api_token=my_secret_key" in str(request.url)

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_429_with_success(self):
        """Should retry on 429 and succeed on subsequent attempt."""
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, text="Rate limited")
            return httpx.Response(200, json={"success": True})

        respx.get("https://api.example.com/data").mock(side_effect=side_effect)

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)
        async with HTTPClient(retry_config=config) as client:
            response = await client.get("https://api.example.com/data")

        assert response.status_code == 200
        assert call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_500_with_success(self):
        """Should retry on 500 and succeed on subsequent attempt."""
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(500, text="Server error")
            return httpx.Response(200, json={"success": True})

        respx.get("https://api.example.com/data").mock(side_effect=side_effect)

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)
        async with HTTPClient(retry_config=config) as client:
            response = await client.get("https://api.example.com/data")

        assert response.status_code == 200
        assert call_count == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_rate_limit_error_after_retries_exhausted(self):
        """Should raise RateLimitError after all retries fail with 429."""
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(429, text="Rate limited")
        )

        config = RetryConfig(max_retries=2, base_delay=0.01, jitter_factor=0.0)
        async with HTTPClient(retry_config=config) as client:
            with pytest.raises(RateLimitError) as exc_info:
                await client.get("https://api.example.com/data")

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_error_after_retries_exhausted(self):
        """Should raise HTTPClientError after all retries fail with 5xx."""
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(503, text="Service unavailable")
        )

        config = RetryConfig(max_retries=2, base_delay=0.01, jitter_factor=0.0)
        async with HTTPClient(retry_config=config) as client:
            with pytest.raises(HTTPClientError) as exc_info:
                await client.get("https://api.example.com/data")

        assert exc_info.value.status_code == 503
        assert "failed with status 503" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_retry_on_4xx(self):
        """Should not retry on non-429 4xx errors."""
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(404, text="Not found")

        respx.get("https://api.example.com/data").mock(side_effect=side_effect)

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)
        async with HTTPClient(retry_config=config) as client:
            with pytest.raises(HTTPClientError) as exc_info:
                await client.get("https://api.example.com/data")

        assert call_count == 1  # No retries
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_timeout(self):
        """Should retry on timeout exception."""
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("Request timed out")
            return httpx.Response(200, json={"success": True})

        respx.get("https://api.example.com/data").mock(side_effect=side_effect)

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)
        async with HTTPClient(retry_config=config) as client:
            response = await client.get("https://api.example.com/data")

        assert response.status_code == 200
        assert call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_connect_error(self):
        """Should retry on connection error."""
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            return httpx.Response(200, json={"success": True})

        respx.get("https://api.example.com/data").mock(side_effect=side_effect)

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)
        async with HTTPClient(retry_config=config) as client:
            response = await client.get("https://api.example.com/data")

        assert response.status_code == 200
        assert call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_key_rotation_on_retry(self):
        """Should rotate API keys on retry."""
        call_count = 0
        used_keys = []

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            # Extract key from header
            auth_header = request.headers.get("Authorization", "")
            key = auth_header.replace("Bearer ", "")
            used_keys.append(key)

            if call_count <= 2:
                return httpx.Response(429, text="Rate limited")
            return httpx.Response(200, json={"success": True})

        respx.get("https://api.example.com/data").mock(side_effect=side_effect)

        rotator = APIKeyRotator(keys=["key_a", "key_b", "key_c"])
        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)

        async with HTTPClient(retry_config=config) as client:
            await client.get(
                "https://api.example.com/data",
                api_key_rotator=rotator,
                api_key_header="Authorization",
                api_key_prefix="Bearer ",
            )

        # Should have used different keys
        assert used_keys == ["key_a", "key_b", "key_c"]

    @pytest.mark.asyncio
    async def test_client_not_used_as_context_manager(self):
        """Should raise error if client not used as context manager."""
        client = HTTPClient()

        with pytest.raises(RuntimeError, match="must be used as async context manager"):
            await client.get("https://api.example.com/data")

    @pytest.mark.asyncio
    @respx.mock
    async def test_custom_headers(self):
        """Should include custom headers in request."""
        route = respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={})
        )

        async with HTTPClient() as client:
            await client.get(
                "https://api.example.com/data",
                headers={"X-Custom-Header": "custom_value"},
            )

        request = route.calls.last.request
        assert request.headers.get("X-Custom-Header") == "custom_value"

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_key_header_without_prefix(self):
        """Should add API key without prefix when prefix is empty."""
        route = respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json={})
        )

        rotator = APIKeyRotator(keys=["raw_key"])

        async with HTTPClient() as client:
            await client.get(
                "https://api.example.com/data",
                api_key_rotator=rotator,
                api_key_header="X-API-KEY",
            )

        request = route.calls.last.request
        assert request.headers.get("X-API-KEY") == "raw_key"

    @pytest.mark.asyncio
    @respx.mock
    async def test_exponential_backoff_timing(self):
        """Should wait appropriate time between retries."""
        call_times = []

        def side_effect(request):
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 3:
                return httpx.Response(500, text="Error")
            return httpx.Response(200, json={"success": True})

        respx.get("https://api.example.com/data").mock(side_effect=side_effect)

        # Use base_delay=0.1 for testable timing
        config = RetryConfig(max_retries=3, base_delay=0.1, jitter_factor=0.0)

        async with HTTPClient(retry_config=config) as client:
            await client.get("https://api.example.com/data")

        # Check delays between calls
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # First delay should be ~0.1s (2^0 * 0.1)
        assert 0.08 <= delay1 <= 0.15
        # Second delay should be ~0.2s (2^1 * 0.1)
        assert 0.15 <= delay2 <= 0.30
