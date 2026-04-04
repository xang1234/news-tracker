"""FilingProvider interface — the contract for all filing data sources.

Every filing provider (edgartools, direct SEC API, etc.) must implement
this interface. The interface guarantees:
    1. Consistent output shape (FilingResult with FilingIdentity).
    2. Centralized SEC policy compliance (rate limits, identity).
    3. Lineage fields that downstream claim extraction can trace.

Provider implementations should:
    - Use SECPolicy for all HTTP requests to SEC endpoints.
    - Emit FilingResult with fully populated FilingIdentity.
    - Set provider name for lineage attribution.
    - Handle errors gracefully (return FilingResult with status='failed').
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from src.filing.schemas import FilingIdentity, FilingResult
from src.filing.sec_policy import SECPolicy

logger = logging.getLogger(__name__)


class SECRateLimiter:
    """Rate limiter tuned for SEC EDGAR's per-second limit.

    Unlike the ingestion RateLimiter (per-minute token bucket), SEC
    requires per-second enforcement. This uses a simple sliding window.
    """

    def __init__(self, requests_per_second: int = 8) -> None:
        self._rps = requests_per_second
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        async with self._lock:
            now = time.monotonic()
            # Prune timestamps older than 1 second
            self._timestamps = [
                t for t in self._timestamps if now - t < 1.0
            ]
            if len(self._timestamps) >= self._rps:
                # Wait until the oldest request falls outside the window
                sleep_time = 1.0 - (now - self._timestamps[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self._timestamps = [
                    t for t in self._timestamps if time.monotonic() - t < 1.0
                ]
            self._timestamps.append(time.monotonic())


class FilingProvider(ABC):
    """Abstract base class for filing data providers.

    Subclasses must implement:
        - name: Provider identifier for lineage attribution.
        - fetch_filing(): Fetch and parse a single filing by accession number.
        - search_filings(): Search for filings by CIK and/or filing type.

    The base class provides:
        - SEC rate limiter (shared across all requests).
        - SEC policy access (User-Agent, URLs, etc.).
        - Error wrapping into FilingResult with status='failed'.
    """

    def __init__(self, policy: SECPolicy | None = None) -> None:
        self._policy = policy or SECPolicy()
        self._rate_limiter = SECRateLimiter(
            requests_per_second=self._policy.rate_limit_per_second
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for lineage attribution (e.g., 'edgartools', 'sec_api')."""
        ...

    @abstractmethod
    async def fetch_filing(
        self, accession_number: str
    ) -> FilingResult:
        """Fetch and parse a single filing by accession number.

        Args:
            accession_number: SEC accession number (e.g., "0001234567-24-000123").

        Returns:
            FilingResult with populated identity, sections, and status.
            On error, returns FilingResult with status='failed' and error_message.
        """
        ...

    @abstractmethod
    async def search_filings(
        self,
        *,
        cik: str | None = None,
        ticker: str | None = None,
        filing_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 20,
    ) -> list[FilingIdentity]:
        """Search for filings matching the given criteria.

        At least one of cik or ticker must be provided.

        Args:
            cik: SEC Central Index Key.
            ticker: Ticker symbol (resolved to CIK internally).
            filing_type: Filter by form type (10-K, 10-Q, etc.).
            date_from: Start date for filed_date filter (YYYY-MM-DD).
            date_to: End date for filed_date filter (YYYY-MM-DD).
            limit: Maximum number of results.

        Returns:
            List of FilingIdentity objects matching the criteria.
        """
        ...

    async def _acquire_rate_limit(self) -> None:
        """Acquire a rate limit slot before making an SEC request."""
        await self._rate_limiter.acquire()

    @property
    def policy(self) -> SECPolicy:
        """Access the SEC policy configuration."""
        return self._policy
