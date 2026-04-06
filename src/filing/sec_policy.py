"""Centralized SEC identity and rate-limit policy.

All filing providers MUST use these settings when accessing SEC EDGAR.
The SEC requires:
    1. A declared User-Agent with company name and contact email.
    2. No more than 10 requests per second to EDGAR endpoints.
    3. Compliance with the SEC fair access policy.

This module is the single source of truth for SEC access parameters.
Provider implementations import from here rather than hardcoding values.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class SECPolicy(BaseSettings):
    """SEC EDGAR access policy configuration.

    All fields are configurable via environment variables with the
    SEC_ prefix. Defaults reflect SEC fair access requirements.

    Attributes:
        user_agent: User-Agent string for SEC requests. Must include
            company name and contact email per SEC guidelines.
        rate_limit_per_second: Maximum requests per second to EDGAR.
            SEC enforces 10 req/s; we default to 8 for safety margin.
        max_retries: Maximum retry attempts for failed requests.
        retry_base_delay: Base delay in seconds for exponential backoff.
        retry_max_delay: Maximum delay in seconds between retries.
        request_timeout: HTTP request timeout in seconds.
        base_url: SEC EDGAR base URL.
        filing_base_url: SEC EDGAR full-text search/filing URL.
    """

    model_config = SettingsConfigDict(
        env_prefix="SEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Identity (required by SEC) ----------------------------------------
    user_agent: str = "news-tracker support@example.com"

    # -- Rate limiting -----------------------------------------------------
    rate_limit_per_second: int = 8

    # -- Retry policy ------------------------------------------------------
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    # -- HTTP settings -----------------------------------------------------
    request_timeout: float = 30.0

    # -- SEC EDGAR URLs ----------------------------------------------------
    base_url: str = "https://efts.sec.gov/LATEST"
    filing_base_url: str = "https://www.sec.gov/cgi-bin/browse-edgar"
    archives_url: str = "https://www.sec.gov/Archives/edgar/data"

    @property
    def headers(self) -> dict[str, str]:
        """HTTP headers required for SEC EDGAR requests.

        Does not set Host — HTTP clients derive it from the request URL.
        """
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
