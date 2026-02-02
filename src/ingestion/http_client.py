"""
HTTP infrastructure layer with retry logic and API key rotation.

Provides:
- APIKeyRotator: Round-robin rotation for comma-separated API keys
- RetryConfig: Exponential backoff configuration
- HTTPClient: Async HTTP client with automatic retry and key rotation

This layer separates HTTP concerns (retries, backoff, key rotation) from
domain logic (article transformation) in the adapters.
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class APIKeyRotator:
    """
    Round-robin API key rotation from comma-separated environment variable.

    Supports multiple API keys for the same service, rotating through them
    to distribute load and avoid rate limits hitting a single key.

    Example:
        rotator = APIKeyRotator.from_env_var("key1,key2,key3")
        key = rotator.get_key()  # Returns keys in round-robin order
    """

    keys: list[str]
    _current_index: int = field(default=0, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @classmethod
    def from_env_var(cls, value: str | None) -> "APIKeyRotator | None":
        """
        Create rotator from comma-separated environment variable value.

        Args:
            value: Comma-separated API keys or single key, or None

        Returns:
            APIKeyRotator instance or None if no keys provided
        """
        if not value:
            return None

        # Split by comma and strip whitespace, filter empty strings
        keys = [k.strip() for k in value.split(",") if k.strip()]

        if not keys:
            return None

        return cls(keys=keys)

    async def get_key(self) -> str:
        """
        Get the next API key in round-robin rotation.

        Thread-safe via asyncio lock to handle concurrent requests.

        Returns:
            The next API key in rotation
        """
        async with self._lock:
            key = self.keys[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.keys)
            return key

    def get_key_sync(self) -> str:
        """
        Get the next API key synchronously (for non-async contexts).

        Note: Not thread-safe. Use get_key() for async contexts.

        Returns:
            The next API key in rotation
        """
        key = self.keys[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.keys)
        return key

    @property
    def key_count(self) -> int:
        """Return the number of available keys."""
        return len(self.keys)


@dataclass
class RetryConfig:
    """
    Exponential backoff configuration for HTTP retries.

    Implements exponential backoff with jitter to prevent thundering herd
    problems when multiple clients retry simultaneously.

    Formula: min(max_backoff, base_delay * 2^attempt) * (1 + random(0, jitter_factor))
    """

    max_retries: int = 3
    max_backoff_seconds: float = 60.0
    base_delay: float = 1.0
    jitter_factor: float = 0.1

    def calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff duration for a given retry attempt.

        Args:
            attempt: The retry attempt number (0-indexed)

        Returns:
            Backoff duration in seconds with jitter applied
        """
        # Exponential backoff: base * 2^attempt
        delay = self.base_delay * (2**attempt)

        # Cap at max backoff
        delay = min(delay, self.max_backoff_seconds)

        # Add jitter to prevent thundering herd
        jitter = delay * self.jitter_factor * random.random()
        delay += jitter

        return delay

    def is_retryable_status(self, status_code: int) -> bool:
        """
        Check if an HTTP status code should trigger a retry.

        Retryable status codes:
        - 429: Too Many Requests (rate limited)
        - 500: Internal Server Error
        - 502: Bad Gateway
        - 503: Service Unavailable
        - 504: Gateway Timeout

        Args:
            status_code: HTTP response status code

        Returns:
            True if the request should be retried
        """
        return status_code in {429, 500, 502, 503, 504}

    def is_retryable_exception(self, exc: Exception) -> bool:
        """
        Check if an exception should trigger a retry.

        Retryable exceptions:
        - httpx.TimeoutException: Request timed out
        - httpx.ConnectError: Connection failed
        - httpx.ReadError: Error reading response

        Args:
            exc: The exception that was raised

        Returns:
            True if the request should be retried
        """
        return isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
            ),
        )


class HTTPClientError(Exception):
    """Base exception for HTTP client errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RateLimitError(HTTPClientError):
    """Raised when rate limit is hit and all retries exhausted."""

    pass


class HTTPClient:
    """
    Async HTTP client with retry logic and API key rotation.

    Features:
    - Exponential backoff with jitter on retryable errors
    - Automatic retry on 429, 5xx status codes
    - Automatic retry on timeout/connection errors
    - Optional API key rotation for each request
    - Context manager for proper resource cleanup

    Example:
        config = RetryConfig(max_retries=3)
        async with HTTPClient(config) as client:
            response = await client.get(
                "https://api.example.com/data",
                params={"q": "search"},
                api_key_rotator=rotator,
                api_key_header="Authorization",
                api_key_prefix="Bearer ",
            )
    """

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize HTTP client.

        Args:
            retry_config: Configuration for retry behavior. Uses defaults if None.
            timeout: Request timeout in seconds.
        """
        self.retry_config = retry_config or RetryConfig()
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "HTTPClient":
        """Enter async context manager, create client."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager, close client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        api_key_rotator: APIKeyRotator | None = None,
        api_key_header: str | None = None,
        api_key_prefix: str = "",
        api_key_param: str | None = None,
    ) -> httpx.Response:
        """
        Perform GET request with retry logic.

        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            api_key_rotator: Optional key rotator for authentication
            api_key_header: Header name for API key (e.g., "Authorization", "X-API-KEY")
            api_key_prefix: Prefix for API key in header (e.g., "Bearer ")
            api_key_param: Query parameter name for API key (alternative to header)

        Returns:
            httpx.Response on success

        Raises:
            HTTPClientError: On non-retryable errors or after retries exhausted
            RateLimitError: When rate limited and retries exhausted
        """
        return await self._request_with_retry(
            method="GET",
            url=url,
            params=params,
            headers=headers,
            api_key_rotator=api_key_rotator,
            api_key_header=api_key_header,
            api_key_prefix=api_key_prefix,
            api_key_param=api_key_param,
        )

    async def post(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        api_key_rotator: APIKeyRotator | None = None,
        api_key_header: str | None = None,
        api_key_prefix: str = "",
        api_key_param: str | None = None,
    ) -> httpx.Response:
        """
        Perform POST request with retry logic.

        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            json_body: JSON body to send
            api_key_rotator: Optional key rotator for authentication
            api_key_header: Header name for API key
            api_key_prefix: Prefix for API key in header
            api_key_param: Query parameter name for API key

        Returns:
            httpx.Response on success

        Raises:
            HTTPClientError: On non-retryable errors or after retries exhausted
            RateLimitError: When rate limited and retries exhausted
        """
        return await self._request_with_retry(
            method="POST",
            url=url,
            params=params,
            headers=headers,
            json_body=json_body,
            api_key_rotator=api_key_rotator,
            api_key_header=api_key_header,
            api_key_prefix=api_key_prefix,
            api_key_param=api_key_param,
        )

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        api_key_rotator: APIKeyRotator | None = None,
        api_key_header: str | None = None,
        api_key_prefix: str = "",
        api_key_param: str | None = None,
    ) -> httpx.Response:
        """
        Execute HTTP request with retry logic.

        Implements exponential backoff with jitter on retryable errors.
        Rotates API keys on each retry if rotator is provided.
        """
        if not self._client:
            raise RuntimeError("HTTPClient must be used as async context manager")

        last_exception: Exception | None = None
        last_status_code: int | None = None
        last_response_body: str | None = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Prepare headers
                request_headers = dict(headers) if headers else {}
                request_params = dict(params) if params else {}

                # Add API key if rotator provided
                if api_key_rotator:
                    api_key = await api_key_rotator.get_key()

                    if api_key_header:
                        request_headers[api_key_header] = f"{api_key_prefix}{api_key}"
                    elif api_key_param:
                        request_params[api_key_param] = api_key

                # Make request
                if method == "GET":
                    response = await self._client.get(
                        url,
                        params=request_params if request_params else None,
                        headers=request_headers if request_headers else None,
                    )
                elif method == "POST":
                    response = await self._client.post(
                        url,
                        params=request_params if request_params else None,
                        headers=request_headers if request_headers else None,
                        json=json_body,
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Check for retryable status codes
                if self.retry_config.is_retryable_status(response.status_code):
                    last_status_code = response.status_code
                    last_response_body = response.text

                    if attempt < self.retry_config.max_retries:
                        backoff = self.retry_config.calculate_backoff(attempt)
                        logger.warning(
                            f"Retryable status {response.status_code} from {url}, "
                            f"attempt {attempt + 1}/{self.retry_config.max_retries + 1}, "
                            f"backing off {backoff:.2f}s"
                        )
                        await asyncio.sleep(backoff)
                        continue

                    # Retries exhausted
                    if response.status_code == 429:
                        raise RateLimitError(
                            f"Rate limit exceeded for {url} after {attempt + 1} attempts",
                            status_code=response.status_code,
                            response_body=last_response_body,
                        )
                    raise HTTPClientError(
                        f"Request failed with status {response.status_code} after {attempt + 1} attempts",
                        status_code=response.status_code,
                        response_body=last_response_body,
                    )

                # Non-retryable error status
                if response.status_code >= 400:
                    raise HTTPClientError(
                        f"Request failed with status {response.status_code}",
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                # Success
                return response

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                last_exception = e

                if attempt < self.retry_config.max_retries:
                    backoff = self.retry_config.calculate_backoff(attempt)
                    logger.warning(
                        f"Retryable error {type(e).__name__} for {url}, "
                        f"attempt {attempt + 1}/{self.retry_config.max_retries + 1}, "
                        f"backing off {backoff:.2f}s"
                    )
                    await asyncio.sleep(backoff)
                    continue

                # Retries exhausted
                raise HTTPClientError(
                    f"Request failed after {attempt + 1} attempts: {e}",
                    status_code=last_status_code,
                ) from e

        # Should not reach here, but just in case
        raise HTTPClientError(
            f"Request failed after {self.retry_config.max_retries + 1} attempts",
            status_code=last_status_code,
            response_body=last_response_body,
        )
