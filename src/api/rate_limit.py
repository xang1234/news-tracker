"""
API rate limiting using slowapi.

Provides a shared Limiter instance keyed by API key (if present)
or client IP address. Enable via RATE_LIMIT_ENABLED=true.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from src.config.settings import get_settings


def _get_rate_limit_key(request: Request) -> str:
    """Extract rate limit key: API key header or remote IP."""
    api_key = request.headers.get("X-API-KEY")
    if api_key:
        return api_key
    return get_remote_address(request)


def create_limiter() -> Limiter:
    """Create a configured Limiter instance."""
    settings = get_settings()
    return Limiter(
        key_func=_get_rate_limit_key,
        default_limits=[settings.rate_limit_default],
        storage_uri=str(settings.redis_url),
    )


limiter = create_limiter()
