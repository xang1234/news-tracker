"""
Reddit API adapter for financial content ingestion.

Uses PRAW (Python Reddit API Wrapper) to fetch posts and comments
from financial subreddits. Handles:
- OAuth2 authentication
- Rate limiting (60 requests/min)
- Subreddit monitoring
- Ticker extraction from natural language (no cashtags)
"""

import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import httpx

from src.config.settings import get_settings
from src.ingestion.base_adapter import BaseAdapter, clean_text, extract_tickers
from src.ingestion.schemas import (
    EngagementMetrics,
    NormalizedDocument,
    Platform,
)

logger = logging.getLogger(__name__)

# Subreddits to monitor for financial content
FINANCIAL_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "semiconductors",
    "options",
    "stockmarket",
    "AMD_Stock",
    "nvidia",
    "intel",
]

# Reddit API endpoints
REDDIT_API_BASE = "https://oauth.reddit.com"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"


class RedditAdapter(BaseAdapter):
    """
    Reddit API adapter for fetching financial posts and comments.

    Monitors specified subreddits for content mentioning tracked tickers.
    Extracts tickers from natural language (Reddit doesn't use cashtags).

    Rate Limits:
        - 60 requests per minute (OAuth)
        - 100 posts per request

    Content Handling:
        - Posts: title + selftext
        - Handles Reddit markdown formatting
        - upvote_ratio is a critical signal
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        subreddits: list[str] | None = None,
        rate_limit: int = 60,
        posts_per_subreddit: int = 25,
    ):
        """
        Initialize Reddit adapter.

        Args:
            client_id: Reddit OAuth client ID
            client_secret: Reddit OAuth client secret
            user_agent: User agent string
            subreddits: List of subreddits to monitor
            rate_limit: Requests per minute
            posts_per_subreddit: Posts to fetch per subreddit
        """
        super().__init__(rate_limit=rate_limit)

        settings = get_settings()
        self._client_id = client_id or settings.reddit_client_id
        self._client_secret = client_secret or settings.reddit_client_secret
        self._user_agent = user_agent or settings.reddit_user_agent
        self._subreddits = subreddits if subreddits is not None else FINANCIAL_SUBREDDITS
        self._posts_per_subreddit = posts_per_subreddit

        self._access_token: str | None = None

        if not self._client_id or not self._client_secret:
            logger.warning(
                "Reddit API credentials not configured. "
                "Adapter will not be able to fetch data."
            )

    @property
    def platform(self) -> Platform:
        return Platform.REDDIT

    async def _get_access_token(self, client: httpx.AsyncClient) -> str | None:
        """
        Get OAuth access token from Reddit.

        Returns:
            Access token string or None on failure
        """
        if self._access_token:
            return self._access_token

        try:
            # Rate limit before token request
            await self._rate_limiter.acquire()

            response = await client.post(
                REDDIT_TOKEN_URL,
                auth=(self._client_id, self._client_secret),
                data={
                    "grant_type": "client_credentials",
                },
                headers={
                    "User-Agent": self._user_agent,
                },
            )
            response.raise_for_status()

            data = response.json()
            self._access_token = data.get("access_token")
            return self._access_token

        except Exception as e:
            logger.error(f"Failed to get Reddit access token: {e}")
            return None

    async def _fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch posts from Reddit API.

        Yields raw post data from configured subreddits.
        """
        if not self._client_id or not self._client_secret:
            logger.error("Reddit credentials not configured")
            return

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get access token
            token = await self._get_access_token(client)
            if not token:
                return

            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": self._user_agent,
            }

            for subreddit in self._subreddits:
                try:
                    # Rate limit before each subreddit API call
                    await self._rate_limiter.acquire()

                    # Fetch hot posts from subreddit
                    response = await client.get(
                        f"{REDDIT_API_BASE}/r/{subreddit}/hot",
                        headers=headers,
                        params={
                            "limit": self._posts_per_subreddit,
                        },
                    )

                    if response.status_code == 429:
                        logger.warning("Reddit rate limit hit")
                        continue

                    response.raise_for_status()
                    data = response.json()

                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"Reddit API error for r/{subreddit}: {e.response.status_code}"
                    )
                    continue
                except Exception as e:
                    logger.error(f"Reddit request failed for r/{subreddit}: {e}")
                    continue

                # Yield posts
                posts = data.get("data", {}).get("children", [])
                for post in posts:
                    yield {
                        "subreddit": subreddit,
                        "post": post.get("data", {}),
                    }

                logger.debug(f"Fetched {len(posts)} posts from r/{subreddit}")

    def _transform(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        """Transform Reddit post to NormalizedDocument."""
        try:
            post = raw["post"]
            subreddit = raw["subreddit"]

            # Skip stickied/pinned posts
            if post.get("stickied"):
                return None

            # Skip removed/deleted posts
            if post.get("removed_by_category"):
                return None

            # Parse timestamp
            created_utc = post.get("created_utc", 0)
            timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc)

            # Build content from title + selftext
            title = post.get("title", "")
            selftext = post.get("selftext", "")

            # Clean Reddit markdown
            content = self._clean_reddit_markdown(f"{title}\n\n{selftext}")
            content = clean_text(content)

            if len(content) < 10:
                return None

            # Extract tickers from content (Reddit doesn't use cashtags)
            tickers = extract_tickers(content)

            # Get engagement metrics
            engagement = EngagementMetrics(
                likes=post.get("ups", 0),
                shares=post.get("num_crossposts", 0),
                comments=post.get("num_comments", 0),
                upvote_ratio=post.get("upvote_ratio"),
            )

            # Build URL
            permalink = post.get("permalink", "")
            url = f"https://reddit.com{permalink}" if permalink else None

            return NormalizedDocument(
                id=f"reddit_{post.get('id', '')}",
                platform=Platform.REDDIT,
                url=url,
                timestamp=timestamp,
                author_id=post.get("author_fullname", ""),
                author_name=post.get("author", "unknown"),
                author_verified=False,  # Reddit doesn't have verified users like Twitter
                content=content,
                content_type="post",
                title=title,
                engagement=engagement,
                tickers_mentioned=tickers,
                raw_data={
                    "subreddit": subreddit,
                    "flair": post.get("link_flair_text"),
                    "is_self": post.get("is_self"),
                    "domain": post.get("domain"),
                },
            )

        except Exception as e:
            logger.debug(f"Failed to transform Reddit post: {e}")
            return None

    def _clean_reddit_markdown(self, text: str) -> str:
        """
        Clean Reddit-specific markdown formatting.

        Args:
            text: Raw Reddit text with markdown

        Returns:
            Cleaned plain text
        """
        # Remove blockquotes
        text = re.sub(r'^>+\s*', '', text, flags=re.MULTILINE)

        # Remove bold/italic markers
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)

        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Remove strikethrough
        text = re.sub(r'~~([^~]+)~~', r'\1', text)

        # Remove heading markers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

        return text

    async def health_check(self) -> bool:
        """Check if Reddit API is accessible."""
        if not self._client_id or not self._client_secret:
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                token = await self._get_access_token(client)
                return token is not None
        except Exception:
            return False
