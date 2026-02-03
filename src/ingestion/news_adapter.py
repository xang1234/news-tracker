"""
News API adapter with multi-source fallback.

Aggregates financial news from multiple providers:
- Finnhub (primary): Focused on financial news
- NewsAPI (fallback): Broader coverage
- Alpha Vantage (fallback): Additional financial news
- Newsfilter.io: Real-time SEC filings and news
- Marketaux: Global financial news aggregation
- Finlight.me: AI-curated financial news

Handles:
- Multi-source fallback logic
- Deduplication across providers
- Source authority tiering
- Exponential backoff with API key rotation

CRITICAL: News often LAGS social media by 24-72 hours for emerging themes.
Its value is CONFIRMATION, not discovery.
"""

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from src.config.settings import get_settings
from src.config.tickers import SEMICONDUCTOR_TICKERS
from src.ingestion.base_adapter import BaseAdapter, clean_text, extract_tickers, stable_hash
from src.ingestion.http_client import (
    APIKeyRotator,
    HTTPClient,
    HTTPClientError,
    RateLimitError,
    RetryConfig,
)
from src.ingestion.schemas import (
    EngagementMetrics,
    NormalizedDocument,
    Platform,
)

logger = logging.getLogger(__name__)

# API endpoints - Original sources
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"
ALPHA_VANTAGE_NEWS_URL = "https://www.alphavantage.co/query"

# API endpoints - New sources
NEWSFILTER_URL = "https://api.newsfilter.io/search"
MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
FINLIGHT_URL = "https://api.finlight.me/v2/articles"

# Source authority tiers for weighting
SOURCE_AUTHORITY = {
    # Tier 1 (weight 3x): Premium financial sources
    "wsj.com": 3,
    "bloomberg.com": 3,
    "reuters.com": 3,
    "ft.com": 3,
    "barrons.com": 3,
    # Tier 2 (weight 2x): Major financial outlets
    "cnbc.com": 2,
    "yahoo.com": 2,
    "seekingalpha.com": 2,
    "marketwatch.com": 2,
    "investors.com": 2,
    # Tier 3 (weight 1x): Everything else
}


def get_source_weight(url: str) -> int:
    """Get authority weight for a news source."""
    if not url:
        return 1
    url_lower = url.lower()
    for domain, weight in SOURCE_AUTHORITY.items():
        if domain in url_lower:
            return weight
    return 1


# Timestamp parsing utilities

def _parse_unix_timestamp(value: Any) -> datetime:
    """Parse Unix timestamp to datetime."""
    return datetime.fromtimestamp(value or 0, tz=timezone.utc)


def _parse_iso_timestamp(value: str) -> datetime:
    """Parse ISO format timestamp (with Z suffix) to datetime."""
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_alpha_vantage_timestamp(value: str) -> datetime:
    """Parse Alpha Vantage format (20231215T143022) to datetime."""
    if not value:
        return datetime.now(timezone.utc)
    return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


@dataclass
class ArticleSourceConfig:
    """Configuration for transforming articles from different API sources."""

    api_source: str
    id_prefix: str
    title_field: str
    content_field: str
    content_fallback_field: str | None
    timestamp_field: str
    timestamp_parser: Callable[[Any], datetime]
    source_field: str
    source_default: str
    extra_raw_data_fields: list[str]


# Source-specific configurations
_FINNHUB_CONFIG = ArticleSourceConfig(
    api_source="finnhub",
    id_prefix="news_finnhub_",
    title_field="headline",
    content_field="summary",
    content_fallback_field=None,
    timestamp_field="datetime",
    timestamp_parser=_parse_unix_timestamp,
    source_field="source",
    source_default="finnhub",
    extra_raw_data_fields=["category"],
)

_NEWSAPI_CONFIG = ArticleSourceConfig(
    api_source="newsapi",
    id_prefix="news_newsapi_",
    title_field="title",
    content_field="content",
    content_fallback_field="description",
    timestamp_field="publishedAt",
    timestamp_parser=_parse_iso_timestamp,
    source_field="source",
    source_default="Unknown",
    extra_raw_data_fields=["author"],
)

_ALPHA_VANTAGE_CONFIG = ArticleSourceConfig(
    api_source="alpha_vantage",
    id_prefix="news_av_",
    title_field="title",
    content_field="summary",
    content_fallback_field=None,
    timestamp_field="time_published",
    timestamp_parser=_parse_alpha_vantage_timestamp,
    source_field="source",
    source_default="alpha_vantage",
    extra_raw_data_fields=["overall_sentiment_score", "overall_sentiment_label"],
)

# New source configurations
_NEWSFILTER_CONFIG = ArticleSourceConfig(
    api_source="newsfilter",
    id_prefix="news_newsfilter_",
    title_field="title",
    content_field="description",
    content_fallback_field=None,
    timestamp_field="publishedAt",
    timestamp_parser=_parse_iso_timestamp,
    source_field="source",
    source_default="newsfilter",
    extra_raw_data_fields=["symbols", "type"],
)

_MARKETAUX_CONFIG = ArticleSourceConfig(
    api_source="marketaux",
    id_prefix="news_marketaux_",
    title_field="title",
    content_field="description",
    content_fallback_field="snippet",
    timestamp_field="published_at",
    timestamp_parser=_parse_iso_timestamp,
    source_field="source",
    source_default="marketaux",
    extra_raw_data_fields=["entities", "relevance_score"],
)

_FINLIGHT_CONFIG = ArticleSourceConfig(
    api_source="finlight",
    id_prefix="news_finlight_",
    title_field="title",
    content_field="content",
    content_fallback_field="summary",
    timestamp_field="publishedAt",
    timestamp_parser=_parse_iso_timestamp,
    source_field="source",
    source_default="finlight",
    extra_raw_data_fields=["sentiment", "topics"],
)


class NewsAdapter(BaseAdapter):
    """
    Multi-source news adapter with fallback logic.

    Queries multiple news APIs and aggregates results. Uses fallback
    logic: Finnhub -> NewsAPI -> Alpha Vantage. Deduplicates articles
    across sources.

    Source Authority Tiers:
        - Tier 1 (3x): WSJ, Bloomberg, Reuters, FT
        - Tier 2 (2x): CNBC, Yahoo Finance, Seeking Alpha
        - Tier 3 (1x): Other financial news
    """

    def __init__(
        self,
        finnhub_key: str | None = None,
        newsapi_key: str | None = None,
        alpha_vantage_key: str | None = None,
        newsfilter_keys: str | None = None,
        marketaux_keys: str | None = None,
        finlight_keys: str | None = None,
        tickers: set[str] | None = None,
        rate_limit: int = 60,
        articles_per_ticker: int = 10,
    ):
        """
        Initialize news adapter.

        Args:
            finnhub_key: Finnhub API key
            newsapi_key: NewsAPI key
            alpha_vantage_key: Alpha Vantage key
            newsfilter_keys: Newsfilter.io API keys (comma-separated for rotation)
            marketaux_keys: Marketaux API keys (comma-separated for rotation)
            finlight_keys: Finlight.me API keys (comma-separated for rotation)
            tickers: Tickers to query news for
            rate_limit: Requests per minute
            articles_per_ticker: Max articles per ticker
        """
        super().__init__(rate_limit=rate_limit)

        settings = get_settings()

        # Original single-key sources
        self._finnhub_key = finnhub_key or settings.finnhub_api_key
        self._newsapi_key = newsapi_key or settings.newsapi_api_key
        self._alpha_vantage_key = alpha_vantage_key or settings.alpha_vantage_api_key

        # New sources with key rotation support
        self._newsfilter_rotator = APIKeyRotator.from_env_var(
            newsfilter_keys or settings.newsfilter_api_keys
        )
        self._marketaux_rotator = APIKeyRotator.from_env_var(
            marketaux_keys or settings.marketaux_api_keys
        )
        self._finlight_rotator = APIKeyRotator.from_env_var(
            finlight_keys or settings.finlight_api_keys
        )

        # HTTP retry configuration from settings
        self._retry_config = RetryConfig(
            max_retries=settings.max_http_retries,
            max_backoff_seconds=settings.max_backoff_seconds,
        )

        self._tickers = tickers or SEMICONDUCTOR_TICKERS
        self._articles_per_ticker = articles_per_ticker

        # Track seen URLs for deduplication
        self._seen_urls: set[str] = set()

        if not self._has_any_api_configured():
            logger.warning(
                "No news API keys configured. Adapter will not fetch data."
            )

    def _has_any_api_configured(self) -> bool:
        """Check if at least one API source is configured."""
        return any([
            self._finnhub_key,
            self._newsapi_key,
            self._alpha_vantage_key,
            self._newsfilter_rotator,
            self._marketaux_rotator,
            self._finlight_rotator,
        ])

    @property
    def platform(self) -> Platform:
        return Platform.NEWS

    async def _fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch news from multiple sources with fallback.

        Yields raw article data from whichever APIs are configured.
        Uses HTTPClient with retry logic for new sources.
        """
        self._seen_urls.clear()

        # Use HTTPClient for new sources (with retry and key rotation)
        async with HTTPClient(retry_config=self._retry_config) as http_client:
            # Also create a plain httpx client for legacy sources
            async with httpx.AsyncClient(timeout=30.0) as legacy_client:
                # Prioritize Finnhub (most focused on financial news)
                if self._finnhub_key:
                    async for article in self._fetch_finnhub(legacy_client):
                        yield article

                # NewsAPI as fallback/supplement
                if self._newsapi_key:
                    async for article in self._fetch_newsapi(legacy_client):
                        yield article

                # Alpha Vantage as tertiary source
                if self._alpha_vantage_key:
                    async for article in self._fetch_alpha_vantage(legacy_client):
                        yield article

                # New sources with HTTPClient (retry + key rotation)
                if self._newsfilter_rotator:
                    async for article in self._fetch_newsfilter(http_client):
                        yield article

                if self._marketaux_rotator:
                    async for article in self._fetch_marketaux(http_client):
                        yield article

                if self._finlight_rotator:
                    async for article in self._fetch_finlight(http_client):
                        yield article

    async def _fetch_finnhub(
        self,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict[str, Any]]:
        """Fetch news from Finnhub API."""
        # Finnhub requires querying by symbol
        # Limit to top 10 tickers to manage rate limits
        top_tickers = list(self._tickers)[:10]

        for ticker in top_tickers:
            try:
                # Rate limit before each API call
                await self._rate_limiter.acquire()

                # Get news from last 7 days
                from datetime import timedelta
                to_date = datetime.now()
                from_date = to_date - timedelta(days=7)

                response = await client.get(
                    FINNHUB_NEWS_URL,
                    params={
                        "symbol": ticker,
                        "from": from_date.strftime("%Y-%m-%d"),
                        "to": to_date.strftime("%Y-%m-%d"),
                        "token": self._finnhub_key,
                    },
                )

                if response.status_code == 429:
                    logger.warning("Finnhub rate limit hit")
                    break

                response.raise_for_status()
                articles = response.json()

            except Exception as e:
                logger.error(f"Finnhub error for {ticker}: {e}")
                continue

            # Yield articles
            for article in articles[: self._articles_per_ticker]:
                url = article.get("url", "")
                if url in self._seen_urls:
                    continue
                self._seen_urls.add(url)

                yield {
                    "source": "finnhub",
                    "ticker": ticker,
                    "article": article,
                }

    async def _fetch_newsapi(
        self,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict[str, Any]]:
        """Fetch news from NewsAPI."""
        # Build query with ticker symbols
        query = " OR ".join(list(self._tickers)[:5])  # Limit query length

        try:
            # Rate limit before API call
            await self._rate_limiter.acquire()

            response = await client.get(
                NEWSAPI_EVERYTHING_URL,
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 50,
                    "apiKey": self._newsapi_key,
                },
            )

            if response.status_code == 429:
                logger.warning("NewsAPI rate limit hit")
                return

            response.raise_for_status()
            data = response.json()

        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return

        # Yield articles
        articles = data.get("articles", [])
        for article in articles:
            url = article.get("url", "")
            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            yield {
                "source": "newsapi",
                "article": article,
            }

    async def _fetch_alpha_vantage(
        self,
        client: httpx.AsyncClient,
    ) -> AsyncIterator[dict[str, Any]]:
        """Fetch news from Alpha Vantage."""
        # Alpha Vantage news sentiment endpoint
        tickers_str = ",".join(list(self._tickers)[:5])

        try:
            # Rate limit before API call
            await self._rate_limiter.acquire()

            response = await client.get(
                ALPHA_VANTAGE_NEWS_URL,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": tickers_str,
                    "apikey": self._alpha_vantage_key,
                },
            )

            if response.status_code == 429:
                logger.warning("Alpha Vantage rate limit hit")
                return

            response.raise_for_status()
            data = response.json()

        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return

        # Yield articles
        articles = data.get("feed", [])
        for article in articles[:50]:
            url = article.get("url", "")
            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            yield {
                "source": "alpha_vantage",
                "article": article,
            }

    async def _fetch_newsfilter(
        self,
        client: HTTPClient,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch news from Newsfilter.io API.

        Uses POST request with Bearer token authentication.
        Supports real-time SEC filings and financial news.
        """
        if not self._newsfilter_rotator:
            return

        # Build search query for tickers
        tickers_query = " OR ".join(list(self._tickers)[:10])

        try:
            # Rate limit before API call
            await self._rate_limiter.acquire()

            response = await client.post(
                NEWSFILTER_URL,
                json_body={
                    "queryString": tickers_query,
                    "from": 0,
                    "size": 50,
                },
                api_key_rotator=self._newsfilter_rotator,
                api_key_header="Authorization",
                api_key_prefix="Bearer ",
            )

            data = response.json()

        except RateLimitError:
            logger.warning("Newsfilter rate limit hit after retries")
            return
        except HTTPClientError as e:
            logger.error(f"Newsfilter error: {e}")
            return
        except Exception as e:
            logger.error(f"Newsfilter unexpected error: {e}")
            return

        # Yield articles from hits
        articles = data.get("articles", [])
        for article in articles[:50]:
            url = article.get("url", "")
            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            yield {
                "source": "newsfilter",
                "article": article,
            }

    async def _fetch_marketaux(
        self,
        client: HTTPClient,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch news from Marketaux API.

        Uses GET request with api_token query parameter.
        Provides global financial news with entity recognition.
        """
        if not self._marketaux_rotator:
            return

        # Build symbols parameter
        symbols = ",".join(list(self._tickers)[:10])

        try:
            # Rate limit before API call
            await self._rate_limiter.acquire()

            response = await client.get(
                MARKETAUX_URL,
                params={
                    "symbols": symbols,
                    "filter_entities": "true",
                    "language": "en",
                    "limit": 50,
                },
                api_key_rotator=self._marketaux_rotator,
                api_key_param="api_token",
            )

            data = response.json()

        except RateLimitError:
            logger.warning("Marketaux rate limit hit after retries")
            return
        except HTTPClientError as e:
            logger.error(f"Marketaux error: {e}")
            return
        except Exception as e:
            logger.error(f"Marketaux unexpected error: {e}")
            return

        # Yield articles from data array
        articles = data.get("data", [])
        for article in articles:
            url = article.get("url", "")
            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            yield {
                "source": "marketaux",
                "article": article,
            }

    async def _fetch_finlight(
        self,
        client: HTTPClient,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch news from Finlight.me API.

        Uses POST request with X-API-KEY header authentication.
        Provides AI-curated financial news with sentiment analysis.
        """
        if not self._finlight_rotator:
            return

        # Build query for tickers
        tickers_list = list(self._tickers)[:10]

        try:
            # Rate limit before API call
            await self._rate_limiter.acquire()

            response = await client.post(
                FINLIGHT_URL,
                json_body={
                    "tickers": tickers_list,
                    "limit": 50,
                    "language": "en",
                },
                api_key_rotator=self._finlight_rotator,
                api_key_header="X-API-KEY",
            )

            data = response.json()

        except RateLimitError:
            logger.warning("Finlight rate limit hit after retries")
            return
        except HTTPClientError as e:
            logger.error(f"Finlight error: {e}")
            return
        except Exception as e:
            logger.error(f"Finlight unexpected error: {e}")
            return

        # Yield articles
        articles = data.get("articles", [])
        for article in articles:
            url = article.get("url", "")
            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            yield {
                "source": "finlight",
                "article": article,
            }

    def _transform(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        """Transform news article to NormalizedDocument."""
        source = raw.get("source", "")

        configs = {
            "finnhub": _FINNHUB_CONFIG,
            "newsapi": _NEWSAPI_CONFIG,
            "alpha_vantage": _ALPHA_VANTAGE_CONFIG,
            "newsfilter": _NEWSFILTER_CONFIG,
            "marketaux": _MARKETAUX_CONFIG,
            "finlight": _FINLIGHT_CONFIG,
        }

        config = configs.get(source)
        if not config:
            return None

        return self._transform_article(raw, config)

    def _transform_article(
        self,
        raw: dict[str, Any],
        config: ArticleSourceConfig,
    ) -> NormalizedDocument | None:
        """
        Transform article using configuration.

        This consolidated method handles all news sources with their differences
        specified in ArticleSourceConfig.
        """
        try:
            article = raw["article"]

            # Parse timestamp using configured parser
            timestamp = config.timestamp_parser(article.get(config.timestamp_field))

            # Extract title and content using configured field names
            title = article.get(config.title_field, "")
            content = article.get(config.content_field, "")
            if not content and config.content_fallback_field:
                content = article.get(config.content_fallback_field, "")

            url = article.get("url", "")

            if not title and not content:
                return None

            # Build and clean full text
            full_text = f"{title}\n\n{content}" if content else title
            full_text = clean_text(full_text)

            # Extract tickers from text
            tickers = extract_tickers(full_text)

            # Source-specific ticker additions
            if config.api_source == "finnhub":
                # Finnhub: add ticker from query if not already found
                ticker = raw.get("ticker", "")
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
            elif config.api_source == "alpha_vantage":
                # Alpha Vantage: extract tickers from sentiment data
                for ticker_data in article.get("ticker_sentiment", []):
                    ticker = ticker_data.get("ticker", "")
                    if ticker and ticker not in tickers:
                        tickers.append(ticker)
            elif config.api_source == "newsfilter":
                # Newsfilter: extract from symbols array
                for symbol in article.get("symbols", []):
                    if isinstance(symbol, str) and symbol not in tickers:
                        tickers.append(symbol)
            elif config.api_source == "marketaux":
                # Marketaux: extract from entities array
                for entity in article.get("entities", []):
                    symbol = entity.get("symbol", "")
                    if symbol and symbol not in tickers:
                        tickers.append(symbol)

            # Get source info - NewsAPI has nested source dict
            if config.api_source == "newsapi":
                source_info = article.get(config.source_field, {})
                author_name = source_info.get("name", config.source_default)
                author_id = source_info.get("id", author_name.lower())
            else:
                author_name = article.get(config.source_field, config.source_default)
                author_id = author_name

            # Generate document ID (use stable hash for deterministic IDs across runs)
            article_id = article.get("id") or stable_hash(url)
            doc_id = f"{config.id_prefix}{article_id}"

            # Get source authority weight
            weight = get_source_weight(url)

            # Build raw_data with common and source-specific fields
            raw_data: dict[str, Any] = {
                "api_source": config.api_source,
                "authority_weight": weight,
            }
            for field in config.extra_raw_data_fields:
                raw_data[field] = article.get(field)

            return NormalizedDocument(
                id=doc_id,
                platform=Platform.NEWS,
                url=url,
                timestamp=timestamp,
                author_id=author_id,
                author_name=author_name,
                content=full_text,
                content_type="article",
                title=title,
                engagement=EngagementMetrics(),
                tickers_mentioned=tickers,
                raw_data=raw_data,
            )

        except Exception as e:
            logger.debug(f"Failed to transform {config.api_source} article: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if at least one news API is accessible."""
        return self._has_any_api_configured()
