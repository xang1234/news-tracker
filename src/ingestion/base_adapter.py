"""
Base adapter interface and shared functionality for platform adapters.

Each platform adapter must implement the fetch() method which yields
NormalizedDocument instances. The base class provides:
- Rate limiting
- Error handling and retry logic
- Metrics tracking
- Common preprocessing utilities
"""

import asyncio
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from src.ingestion.schemas import NormalizedDocument, Platform

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """
    Simple token bucket rate limiter.

    Allows `rate` requests per minute with burst capacity.
    """

    rate: int  # requests per minute
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.rate)
        self._last_update = time.monotonic()

    async def acquire(self) -> None:
        """
        Wait until a token is available, then consume it.

        This implements a smooth rate limiter that refills tokens continuously
        rather than in bursts.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            # Refill tokens based on elapsed time
            self._tokens = min(
                float(self.rate),
                self._tokens + elapsed * (self.rate / 60.0),
            )

            if self._tokens < 1:
                # Wait for token to become available
                wait_time = (1 - self._tokens) * 60.0 / self.rate
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


@dataclass
class AdapterStats:
    """Statistics for an adapter run."""

    documents_fetched: int = 0
    documents_filtered: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def documents_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0
        return self.documents_fetched / elapsed


class BaseAdapter(ABC):
    """
    Abstract base class for platform adapters.

    Subclasses must implement:
        - platform: Platform enum value
        - _fetch_raw(): Async generator yielding raw platform data
        - _transform(): Convert raw data to NormalizedDocument

    The base class handles:
        - Rate limiting (via RateLimiter)
        - Error handling with exponential backoff
        - Basic preprocessing (URL extraction, etc.)
        - Logging and metrics
    """

    def __init__(self, rate_limit: int = 60):
        """
        Initialize adapter with rate limiting.

        Args:
            rate_limit: Maximum requests per minute
        """
        self._rate_limiter = RateLimiter(rate=rate_limit)
        self._stats = AdapterStats()

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """Return the platform this adapter handles."""
        ...

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        return f"{self.platform.value}_adapter"

    @abstractmethod
    async def _fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch raw data from the platform API.

        Yields:
            Raw platform responses as dictionaries

        IMPORTANT: Subclasses MUST call `await self._rate_limiter.acquire()`
        before each HTTP request to enforce rate limiting at the correct
        granularity. Rate limiting should be applied per-request, not per-item.

        Example:
            async def _fetch_raw(self):
                async with httpx.AsyncClient() as client:
                    for ticker in tickers:
                        await self._rate_limiter.acquire()  # Before HTTP call
                        response = await client.get(url, params={...})
                        for item in response.json():
                            yield item  # No rate limit here
        """
        ...

    @abstractmethod
    def _transform(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        """
        Transform raw platform data to NormalizedDocument.

        Args:
            raw: Raw data from platform API

        Returns:
            NormalizedDocument or None if data should be filtered

        This method should NOT raise exceptions - return None for invalid data.
        """
        ...

    async def fetch(self) -> AsyncIterator[NormalizedDocument]:
        """
        Fetch and transform documents from the platform.

        This is the main entry point called by the ingestion service.
        Handles rate limiting, error handling, and logging.

        Yields:
            NormalizedDocument instances
        """
        self._stats = AdapterStats()  # Reset stats for this run

        logger.info(f"Starting fetch for {self.name}")

        try:
            async for raw in self._fetch_raw():
                # Note: Rate limiting is applied in _fetch_raw() before HTTP calls,
                # not here. This ensures rate limiting is per-request, not per-item.
                try:
                    doc = self._transform(raw)
                    if doc is None:
                        self._stats.documents_filtered += 1
                        continue

                    # Apply common preprocessing
                    doc = self._preprocess(doc)
                    self._stats.documents_fetched += 1

                    yield doc

                except Exception as e:
                    self._stats.errors += 1
                    logger.error(
                        f"Error transforming document in {self.name}: {e}",
                        exc_info=True,
                    )
                    continue

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Error in {self.name} fetch: {e}", exc_info=True)
            raise

        finally:
            logger.info(
                f"{self.name} completed: "
                f"fetched={self._stats.documents_fetched}, "
                f"filtered={self._stats.documents_filtered}, "
                f"errors={self._stats.errors}, "
                f"elapsed={self._stats.elapsed_seconds:.2f}s"
            )

    def _preprocess(self, doc: NormalizedDocument) -> NormalizedDocument:
        """
        Apply common preprocessing to all documents.

        - Extract URLs from content
        - Basic content cleaning

        Platform-specific preprocessing should be done in _transform().
        """
        # Extract URLs from content
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, doc.content)
        if urls:
            doc.urls_mentioned = list(set(urls))

        return doc

    @property
    def stats(self) -> AdapterStats:
        """Get current adapter statistics."""
        return self._stats

    async def health_check(self) -> bool:
        """
        Check if the adapter can connect to its platform.

        Override in subclasses for platform-specific health checks.
        """
        return True


# Common preprocessing utilities used across adapters

def extract_cashtags(text: str) -> list[str]:
    """
    Extract $TICKER cashtags from text.

    This is a simple extraction that only finds $TICKER patterns.
    For full ticker extraction including company names and fuzzy matching,
    use extract_tickers() instead.

    Args:
        text: Text to search

    Returns:
        List of ticker symbols (without $ prefix)
    """
    pattern = r'\$([A-Z]{1,5})\b'
    matches = re.findall(pattern, text.upper())
    return list(set(matches))


# Lazy-loaded ticker extractor singleton
_ticker_extractor = None


def get_ticker_extractor():
    """Get the singleton TickerExtractor instance."""
    global _ticker_extractor
    if _ticker_extractor is None:
        # Import here to avoid circular imports
        from src.ingestion.preprocessor import TickerExtractor
        _ticker_extractor = TickerExtractor()
    return _ticker_extractor


def extract_tickers(text: str) -> list[str]:
    """
    Extract ticker symbols from text using comprehensive extraction.

    Uses the full TickerExtractor which includes:
    1. Cashtag pattern matching ($NVDA)
    2. Direct ticker mentions (NVDA)
    3. Company name lookup (Nvidia -> NVDA)
    4. Fuzzy matching for variations (NVIDIA Corp -> NVDA)

    This is the recommended function for ticker extraction across all adapters.

    Args:
        text: Text to search

    Returns:
        List of unique ticker symbols
    """
    return get_ticker_extractor().extract(text)


def clean_text(text: str) -> str:
    """
    Clean text content by removing excessive whitespace and normalizing.

    Args:
        text: Raw text content

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = " ".join(text.split())

    # Remove null bytes and other control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text.strip()


def stable_hash(value: str) -> str:
    """
    Generate a stable, deterministic hash from a string.

    Uses SHA256 truncated to 16 hex characters (64 bits) for a compact but
    collision-resistant ID. Unlike Python's built-in hash(), this is
    deterministic across process restarts and Python versions.

    Args:
        value: String to hash (typically a URL or identifier)

    Returns:
        16-character hex string (e.g., "a1b2c3d4e5f67890")
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def expand_twitter_abbreviations(text: str) -> str:
    """
    Expand common Twitter/finance abbreviations.

    Args:
        text: Text with abbreviations

    Returns:
        Text with abbreviations expanded (for better NLP)
    """
    abbreviations = {
        r'\bLFG\b': 'lets go',
        r'\bHODL\b': 'hold',
        r'\bFOMO\b': 'fear of missing out',
        r'\bFUD\b': 'fear uncertainty doubt',
        r'\bDD\b': 'due diligence',
        r'\bIMO\b': 'in my opinion',
        r'\bIMHO\b': 'in my humble opinion',
        r'\bTLDR\b': 'too long didnt read',
        r'\bATH\b': 'all time high',
        r'\bATL\b': 'all time low',
        r'\bYOLO\b': 'you only live once',
        r'\bDCA\b': 'dollar cost averaging',
    }

    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def translate_emoji_sentiment(text: str) -> str:
    """
    Translate common financial emojis to text sentiment markers.

    Args:
        text: Text with emojis

    Returns:
        Text with emojis translated
    """
    emoji_map = {
        '🚀': ' bullish ',
        '📈': ' bullish ',
        '🌙': ' bullish ',
        '💎': ' holding ',
        '🙌': ' holding ',
        '📉': ' bearish ',
        '🐻': ' bearish ',
        '🐂': ' bullish ',
        '💰': ' profit ',
        '🔥': ' hot ',
        '⚠️': ' warning ',
        '🎯': ' target ',
    }

    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)

    return text
