"""
API authentication using X-API-KEY header.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config.settings import get_settings

# API key header scheme
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """
    Verify API key from X-API-KEY header.

    Args:
        api_key: API key from header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    settings = get_settings()

    # If no API keys configured, allow all requests (dev mode)
    if not settings.api_keys:
        return "dev-mode"

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-KEY header.",
        )

    # Check if key is in the list of valid keys (filter out empty strings)
    valid_keys = [k.strip() for k in settings.api_keys.split(",") if k.strip()]
    if not valid_keys or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key
