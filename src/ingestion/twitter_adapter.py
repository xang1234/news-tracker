"""
Twitter API v2 adapter for financial content ingestion.

Uses the Twitter API v2 to fetch tweets containing financial cashtags
and relevant keywords. Handles:
- Rate limiting (450 requests/15 min for Essential tier)
- Pagination via next_token
- Tweet expansion (author, engagement metrics)
- Content preprocessing (emoji translation, abbreviation expansion)
"""

import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import httpx

from src.config.settings import get_settings
from src.config.tickers import SEMICONDUCTOR_TICKERS
from src.ingestion.base_adapter import (
    BaseAdapter,
    clean_text,
    expand_twitter_abbreviations,
    extract_cashtags,
    translate_emoji_sentiment,
)
from src.ingestion.schemas import (
    EngagementMetrics,
    NormalizedDocument,
    Platform,
)

logger = logging.getLogger(__name__)

# Twitter API v2 endpoints
TWITTER_API_BASE = "https://api.twitter.com/2"
TWEETS_SEARCH_RECENT = f"{TWITTER_API_BASE}/tweets/search/recent"


class TwitterAdapter(BaseAdapter):
    """
    Twitter API v2 adapter for fetching financial tweets.

    Uses the recent search endpoint to find tweets containing
    semiconductor cashtags. Filters out retweets and non-English content.

    Rate Limits (Essential tier):
        - 450 requests per 15-minute window
        - 100 tweets per request (max)

    Query Strategy:
        - Search for $TICKER cashtags (e.g., "$NVDA OR $AMD")
        - Filter: -is:retweet lang:en
        - Batch by sector to maximize coverage
    """

    def __init__(
        self,
        bearer_token: str | None = None,
        tickers: set[str] | None = None,
        rate_limit: int = 30,  # Conservative default
        max_results_per_request: int = 100,
    ):
        """
        Initialize Twitter adapter.

        Args:
            bearer_token: Twitter API bearer token (or from env)
            tickers: Set of tickers to track (default: semiconductors)
            rate_limit: Requests per minute
            max_results_per_request: Max tweets per API call (10-100)
        """
        super().__init__(rate_limit=rate_limit)

        settings = get_settings()
        self._bearer_token = bearer_token or settings.twitter_bearer_token
        self._tickers = tickers or SEMICONDUCTOR_TICKERS
        self._max_results = min(max_results_per_request, 100)

        if not self._bearer_token:
            logger.warning(
                "Twitter bearer token not configured. "
                "Adapter will not be able to fetch data."
            )

    @property
    def platform(self) -> Platform:
        return Platform.TWITTER

    def _build_query(self, tickers: list[str]) -> str:
        """
        Build Twitter search query for given tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Twitter search query string
        """
        # Build OR query for cashtags
        cashtag_query = " OR ".join(f"${t}" for t in tickers)

        # Add filters
        # -is:retweet: Exclude retweets (reduce noise)
        # lang:en: English only
        query = f"({cashtag_query}) -is:retweet lang:en"

        return query

    async def _fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch tweets from Twitter API.

        Yields raw tweet data with author and metrics expansions.
        """
        if not self._bearer_token:
            logger.error("Twitter bearer token not configured")
            return

        headers = {
            "Authorization": f"Bearer {self._bearer_token}",
            "User-Agent": "NewsTracker/1.0",
        }

        # Split tickers into batches to stay within query length limits
        # Twitter allows ~512 chars in query
        ticker_list = list(self._tickers)
        batch_size = 10  # ~10 tickers per query to be safe

        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(0, len(ticker_list), batch_size):
                batch = ticker_list[i : i + batch_size]
                query = self._build_query(batch)

                logger.debug(f"Twitter query: {query}")

                params = {
                    "query": query,
                    "max_results": self._max_results,
                    "tweet.fields": "created_at,public_metrics,author_id",
                    "expansions": "author_id",
                    "user.fields": "name,username,public_metrics,verified",
                }

                next_token = None
                pages_fetched = 0
                max_pages = 3  # Limit pages per batch

                while pages_fetched < max_pages:
                    if next_token:
                        params["next_token"] = next_token

                    try:
                        # Rate limit before each API call (pagination)
                        await self._rate_limiter.acquire()

                        response = await client.get(
                            TWEETS_SEARCH_RECENT,
                            headers=headers,
                            params=params,
                        )

                        if response.status_code == 429:
                            # Rate limited
                            logger.warning("Twitter rate limit hit")
                            return

                        response.raise_for_status()
                        data = response.json()

                    except httpx.HTTPStatusError as e:
                        logger.error(f"Twitter API error: {e.response.status_code}")
                        if e.response.status_code == 401:
                            logger.error("Twitter authentication failed")
                            return
                        continue

                    except Exception as e:
                        logger.error(f"Twitter request failed: {e}")
                        continue

                    # Build author lookup from includes
                    authors = {}
                    if "includes" in data and "users" in data["includes"]:
                        for user in data["includes"]["users"]:
                            authors[user["id"]] = user

                    # Yield tweets with author info
                    tweets = data.get("data", [])
                    for tweet in tweets:
                        author = authors.get(tweet.get("author_id"), {})
                        yield {
                            "tweet": tweet,
                            "author": author,
                        }

                    # Check for pagination
                    meta = data.get("meta", {})
                    next_token = meta.get("next_token")
                    pages_fetched += 1

                    if not next_token:
                        break

                    logger.debug(
                        f"Fetched page {pages_fetched}, "
                        f"tweets={len(tweets)}, has_more={bool(next_token)}"
                    )

    def _transform(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        """Transform Twitter API response to NormalizedDocument."""
        try:
            tweet = raw["tweet"]
            author = raw.get("author", {})

            # Parse timestamp
            created_at = tweet.get("created_at", "")
            if created_at:
                timestamp = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                )
            else:
                timestamp = datetime.now(timezone.utc)

            # Get content and preprocess
            content = tweet.get("text", "")
            if not content:
                return None

            # Apply Twitter-specific preprocessing
            content = translate_emoji_sentiment(content)
            content = expand_twitter_abbreviations(content)
            content = clean_text(content)

            # Extract tickers from content
            tickers = extract_cashtags(tweet.get("text", ""))

            # Get engagement metrics
            metrics = tweet.get("public_metrics", {})
            engagement = EngagementMetrics(
                likes=metrics.get("like_count", 0),
                shares=metrics.get("retweet_count", 0),
                comments=metrics.get("reply_count", 0),
                views=metrics.get("impression_count"),
            )

            # Get author metrics
            author_metrics = author.get("public_metrics", {})
            author_followers = author_metrics.get("followers_count")

            return NormalizedDocument(
                id=f"twitter_{tweet['id']}",
                platform=Platform.TWITTER,
                url=f"https://twitter.com/i/status/{tweet['id']}",
                timestamp=timestamp,
                author_id=str(tweet.get("author_id", "")),
                author_name=author.get("username", "unknown"),
                author_followers=author_followers,
                author_verified=author.get("verified", False),
                content=content,
                content_type="post",
                engagement=engagement,
                tickers_mentioned=tickers,
                raw_data=raw,
            )

        except Exception as e:
            logger.debug(f"Failed to transform tweet: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if Twitter API is accessible."""
        if not self._bearer_token:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{TWITTER_API_BASE}/users/me",
                    headers={"Authorization": f"Bearer {self._bearer_token}"},
                )
                # 401 means invalid token, but API is reachable
                # 403 means endpoint not authorized (common for app-only auth)
                return response.status_code in (200, 401, 403)
        except Exception:
            return False
