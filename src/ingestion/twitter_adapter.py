"""
Twitter API v2 adapter for financial content ingestion.

Uses the Twitter API v2 to fetch tweets containing financial cashtags
and relevant keywords. Falls back to Sotwe.com scraping when no API
key is configured.

Handles:
- Rate limiting (450 requests/15 min for Essential tier)
- Pagination via next_token
- Tweet expansion (author, engagement metrics)
- Content preprocessing (emoji translation, abbreviation expansion)
- Sotwe fallback for API-less operation
"""

import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import httpx

from src.config.settings import get_settings
from src.config.tickers import SEMICONDUCTOR_TICKERS
from src.config.twitter_accounts import parse_usernames
from src.ingestion.base_adapter import (
    BaseAdapter,
    RateLimiter,
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
from src.ingestion.sotwe_client import SotweClient, SotweClientError, SotweTweet

logger = logging.getLogger(__name__)

# Twitter API v2 endpoints
TWITTER_API_BASE = "https://api.twitter.com/2"
TWEETS_SEARCH_RECENT = f"{TWITTER_API_BASE}/tweets/search/recent"


class TwitterAdapter(BaseAdapter):
    """
    Twitter API v2 adapter for fetching financial tweets.

    Uses the recent search endpoint to find tweets containing
    semiconductor cashtags. Falls back to Sotwe.com scraping when
    no API key is configured.

    Rate Limits (Essential tier):
        - 450 requests per 15-minute window
        - 100 tweets per request (max)

    Query Strategy (API mode):
        - Search for $TICKER cashtags (e.g., "$NVDA OR $AMD")
        - Filter: -is:retweet lang:en
        - Batch by sector to maximize coverage

    Sotwe Fallback:
        - Scrapes tweets from curated semiconductor-focused accounts
        - Uses browser impersonation to bypass Cloudflare
        - Requires Node.js for parsing NUXT JavaScript
    """

    def __init__(
        self,
        bearer_token: str | None = None,
        tickers: set[str] | None = None,
        rate_limit: int = 30,  # Conservative default
        max_results_per_request: int = 100,
        sotwe_usernames: list[str] | None = None,
        sotwe_rate_limit: int | None = None,
    ):
        """
        Initialize Twitter adapter.

        Args:
            bearer_token: Twitter API bearer token (or from env)
            tickers: Set of tickers to track (default: semiconductors)
            rate_limit: Requests per minute for Twitter API
            max_results_per_request: Max tweets per API call (10-100)
            sotwe_usernames: Twitter usernames to track via Sotwe fallback
            sotwe_rate_limit: Requests per minute for Sotwe scraping
        """
        super().__init__(rate_limit=rate_limit)

        settings = get_settings()
        self._bearer_token = bearer_token or settings.twitter_bearer_token
        self._tickers = tickers or SEMICONDUCTOR_TICKERS
        self._max_results = min(max_results_per_request, 100)

        # Sotwe fallback configuration
        self._sotwe_enabled = settings.sotwe_enabled
        self._sotwe_usernames = sotwe_usernames if sotwe_usernames is not None else parse_usernames(
            settings.sotwe_usernames
        )
        self._sotwe_rate_limiter = RateLimiter(
            rate=sotwe_rate_limit or settings.sotwe_rate_limit
        )
        self._sotwe_client: SotweClient | None = None

        # Track seen tweet IDs to avoid duplicates
        self._seen_tweet_ids: set[str] = set()

        if not self._bearer_token:
            if self._sotwe_enabled:
                logger.info(
                    "Twitter bearer token not configured. "
                    "Using Sotwe fallback for tweet ingestion."
                )
            else:
                logger.warning(
                    "Twitter bearer token not configured and Sotwe disabled. "
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
        Fetch tweets from Twitter API or Sotwe fallback.

        Prefers Twitter API when bearer token is configured.
        Falls back to Sotwe scraping otherwise.

        Yields raw tweet data with author and metrics expansions.
        """
        # Clear seen IDs for this fetch cycle
        self._seen_tweet_ids.clear()

        if self._bearer_token:
            # Use Twitter API (primary)
            async for item in self._fetch_twitter_api():
                yield item
        elif self._sotwe_enabled:
            # Use Sotwe fallback
            async for item in self._fetch_sotwe():
                yield item
        else:
            logger.error(
                "Twitter bearer token not configured and Sotwe disabled"
            )

    async def _fetch_twitter_api(self) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch tweets from Twitter API v2.

        Yields raw tweet data with author and metrics expansions.
        """
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
                        tweet_id = tweet.get("id")
                        if tweet_id in self._seen_tweet_ids:
                            continue
                        self._seen_tweet_ids.add(tweet_id)

                        author = authors.get(tweet.get("author_id"), {})
                        yield {
                            "source": "twitter_api",
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

    async def _fetch_sotwe(self) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch tweets from Sotwe.com as fallback.

        Iterates through configured usernames and fetches their recent tweets.
        Rate limits before each user fetch.

        Yields raw tweet data in a format compatible with _transform().
        """
        # Lazy initialize Sotwe client
        if self._sotwe_client is None:
            self._sotwe_client = SotweClient()

        # Check availability before starting
        if not await self._sotwe_client.check_available():
            logger.error(
                "Sotwe fallback unavailable. Ensure Node.js is installed."
            )
            return

        logger.info(
            f"Fetching tweets via Sotwe for {len(self._sotwe_usernames)} accounts"
        )

        for username in self._sotwe_usernames:
            try:
                # Rate limit before each scrape request
                await self._sotwe_rate_limiter.acquire()

                tweets = await self._sotwe_client.fetch_user_tweets(
                    username=username,
                    max_tweets=50,
                )

                logger.debug(f"Sotwe: fetched {len(tweets)} tweets from @{username}")

                for tweet in tweets:
                    # Skip duplicates
                    if tweet.id in self._seen_tweet_ids:
                        continue
                    self._seen_tweet_ids.add(tweet.id)

                    yield {
                        "source": "sotwe",
                        "tweet": tweet,
                        "username": username,
                    }

            except SotweClientError as e:
                logger.warning(f"Sotwe error for @{username}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error fetching @{username}: {e}")
                continue

    def _transform(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        """
        Transform raw tweet data to NormalizedDocument.

        Routes to appropriate transform method based on data source.
        """
        source = raw.get("source", "twitter_api")

        if source == "sotwe":
            return self._transform_sotwe(raw)
        else:
            return self._transform_twitter_api(raw)

    def _transform_twitter_api(self, raw: dict[str, Any]) -> NormalizedDocument | None:
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
                raw_data={"source": "twitter_api"},
            )

        except Exception as e:
            logger.debug(f"Failed to transform Twitter API tweet: {e}")
            return None

    def _transform_sotwe(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        """
        Transform Sotwe tweet data to NormalizedDocument.

        Args:
            raw: Dict containing 'tweet' (SotweTweet) and 'username'

        Returns:
            NormalizedDocument or None if transformation fails
        """
        try:
            tweet: SotweTweet = raw["tweet"]
            username = raw.get("username", tweet.username)

            # Get content and preprocess
            content = tweet.text
            if not content:
                return None

            # Apply Twitter-specific preprocessing
            content = translate_emoji_sentiment(content)
            content = expand_twitter_abbreviations(content)
            content = clean_text(content)

            # Extract tickers from original content
            tickers = extract_cashtags(tweet.text)

            # Build engagement metrics
            engagement = EngagementMetrics(
                likes=tweet.likes,
                shares=tweet.retweets,
                comments=tweet.replies,
                views=tweet.views,
            )

            return NormalizedDocument(
                id=f"twitter_{tweet.id}",
                platform=Platform.TWITTER,
                url=f"https://twitter.com/{username}/status/{tweet.id}",
                timestamp=tweet.created_at,
                author_id=username,
                author_name=tweet.author_name,
                author_followers=None,  # Not available from Sotwe
                author_verified=False,  # Unknown from Sotwe
                content=content,
                content_type="post",
                engagement=engagement,
                tickers_mentioned=tickers,
                raw_data={"source": "sotwe"},
            )

        except Exception as e:
            logger.debug(f"Failed to transform Sotwe tweet: {e}")
            return None

    async def health_check(self) -> bool:
        """
        Check if Twitter data source is accessible.

        Returns True if either Twitter API or Sotwe fallback is available.
        """
        # Check Twitter API first (preferred)
        if self._bearer_token:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"{TWITTER_API_BASE}/users/me",
                        headers={"Authorization": f"Bearer {self._bearer_token}"},
                    )
                    # 401 means invalid token, but API is reachable
                    # 403 means endpoint not authorized (common for app-only auth)
                    if response.status_code in (200, 401, 403):
                        return True
            except Exception:
                pass  # Fall through to Sotwe check

        # Check Sotwe fallback
        if self._sotwe_enabled:
            if self._sotwe_client is None:
                self._sotwe_client = SotweClient()
            return await self._sotwe_client.check_available()

        return False
