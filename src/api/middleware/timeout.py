"""
Request timeout middleware.

Wraps each request in an asyncio timeout to prevent long-running requests
from consuming server resources indefinitely. Returns 504 Gateway Timeout
on expiration. Health and WebSocket endpoints are excluded.
"""

import asyncio

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)

# Paths excluded from timeout enforcement
_EXCLUDED_PREFIXES = ("/health", "/ws/")


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Enforce a maximum request duration, returning 504 on timeout."""

    def __init__(self, app, timeout_seconds: float = 30.0):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        # Skip WebSocket upgrades and excluded paths
        if (
            request.headers.get("upgrade", "").lower() == "websocket"
            or any(request.url.path.startswith(p) for p in _EXCLUDED_PREFIXES)
        ):
            return await call_next(request)

        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await call_next(request)
        except TimeoutError:
            logger.warning(
                "Request timed out",
                path=request.url.path,
                method=request.method,
                timeout_seconds=self.timeout_seconds,
            )
            return JSONResponse(
                status_code=504,
                content={
                    "detail": "Request timed out",
                    "timeout_seconds": self.timeout_seconds,
                },
            )
