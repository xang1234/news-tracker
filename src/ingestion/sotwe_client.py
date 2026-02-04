"""
Sotwe.com scraping client for Twitter data fallback.

Sotwe is a Twitter mirror that provides public tweet data without API access.
This client uses browser impersonation to bypass Cloudflare protection and
BeautifulSoup for HTML parsing.

Requirements:
- curl_cffi: HTTP client with browser impersonation
- beautifulsoup4: HTML parsing (already a project dependency)

Usage:
    client = SotweClient()
    if await client.check_available():
        tweets = await client.fetch_user_tweets("SemiAnalysis")
        for tweet in tweets:
            print(tweet.text)
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Sotwe URL pattern
SOTWE_USER_URL = "https://sotwe.com/{username}"


@dataclass
class SotweTweet:
    """
    Parsed tweet data from Sotwe.

    Attributes:
        id: Tweet ID (numeric string)
        text: Tweet content
        created_at: Tweet timestamp
        username: Author's Twitter username
        author_name: Author's display name
        likes: Like count
        retweets: Retweet count
        replies: Reply count
        views: View count (may be None)
    """

    id: str
    text: str
    created_at: datetime
    username: str
    author_name: str
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    views: int | None = None


def check_node_available() -> bool:
    """
    Check if Node.js is available in the system PATH.

    Note: Node.js is no longer required for Sotwe scraping.
    This function is kept for backwards compatibility but always returns True.

    Returns:
        True (Node.js not required)
    """
    return True


class SotweClient:
    """
    Async client for scraping tweets from Sotwe.com.

    Uses curl_cffi for Cloudflare bypass via browser impersonation,
    and BeautifulSoup for HTML parsing.
    """

    def __init__(self, timeout: float = 30.0):
        """
        Initialize Sotwe client.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self._timeout = timeout

    async def check_available(self) -> bool:
        """
        Check if Sotwe scraping is available.

        Verifies curl_cffi is installed and Sotwe is accessible.

        Returns:
            True if all requirements are met
        """
        # Try to import curl_cffi
        try:
            from curl_cffi.requests import AsyncSession  # noqa: F401
        except ImportError:
            logger.warning(
                "curl_cffi not installed. Install with: pip install curl_cffi"
            )
            return False

        # Quick connectivity check
        try:
            html = await self._fetch_page("nvidia")
            # Verify we got actual content (not a block page)
            return "tweet-card" in html or "tweet-text" in html
        except Exception as e:
            logger.warning(f"Sotwe connectivity check failed: {e}")
            return False

    async def fetch_user_tweets(
        self,
        username: str,
        max_tweets: int = 50,
    ) -> list[SotweTweet]:
        """
        Fetch recent tweets for a Twitter user via Sotwe.

        Args:
            username: Twitter username (without @ prefix)
            max_tweets: Maximum tweets to return

        Returns:
            List of SotweTweet objects

        Raises:
            SotweClientError: If scraping fails
        """
        try:
            html = await self._fetch_page(username)
            tweets = self._parse_tweets_from_html(html, username)
            return tweets[:max_tweets]

        except SotweClientError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch tweets for @{username}: {e}")
            raise SotweClientError(f"Failed to fetch @{username}: {e}") from e

    async def _fetch_page(self, username: str) -> str:
        """
        Fetch Sotwe page HTML with browser impersonation.

        Args:
            username: Twitter username

        Returns:
            Page HTML content
        """
        from curl_cffi.requests import AsyncSession

        url = SOTWE_USER_URL.format(username=username)

        async with AsyncSession() as session:
            response = await session.get(
                url,
                impersonate="chrome",
                timeout=self._timeout,
            )

            if response.status_code != 200:
                raise SotweClientError(
                    f"HTTP {response.status_code} for @{username}"
                )

            return response.text

    def _parse_tweets_from_html(
        self,
        html: str,
        username: str,
    ) -> list[SotweTweet]:
        """
        Parse tweets from Sotwe HTML page.

        Extracts tweet data from embedded NUXT JavaScript data.

        Args:
            html: Page HTML content
            username: Twitter username for attribution

        Returns:
            List of SotweTweet objects
        """
        tweets = []
        seen_ids: set[str] = set()

        # Try to extract tweets from NUXT data first (most reliable)
        tweets = self._parse_tweets_from_nuxt(html, username)

        if tweets:
            return tweets

        # Fallback to HTML parsing if NUXT extraction fails
        return self._parse_tweets_fallback(html, username)

    def _parse_tweets_from_nuxt(
        self,
        html: str,
        username: str,
    ) -> list[SotweTweet]:
        """
        Parse tweets from NUXT embedded data.

        The NUXT data contains tweet objects with id, text, engagement metrics, etc.

        Args:
            html: Page HTML content
            username: Twitter username

        Returns:
            List of SotweTweet objects
        """
        tweets = []
        seen_ids: set[str] = set()

        # Find all tweet objects by looking for id followed by retweetCount
        # This pattern finds the start of each tweet object
        tweet_starts = list(re.finditer(
            r'id:"(\d{15,20})",retweetCount:(\d+|[a-z]+),',
            html
        ))

        for i, match in enumerate(tweet_starts):
            try:
                tweet_id = match.group(1)

                if tweet_id in seen_ids:
                    continue
                seen_ids.add(tweet_id)

                # Get the chunk of text for this tweet (up to next tweet or 2000 chars)
                start_pos = match.start()
                if i + 1 < len(tweet_starts):
                    end_pos = tweet_starts[i + 1].start()
                else:
                    end_pos = start_pos + 2000
                chunk = html[start_pos:end_pos]

                # Extract fields from this chunk
                retweets = self._parse_nuxt_number(match.group(2))

                # Extract other metrics
                fav_match = re.search(r'favoriteCount:(\d+|[a-z]+)', chunk)
                likes = self._parse_nuxt_number(fav_match.group(1)) if fav_match else 0

                view_match = re.search(r'viewCount:(\d+|[a-z]+)', chunk)
                views = self._parse_nuxt_number(view_match.group(1)) if view_match else 0

                reply_match = re.search(r'replyCount:(\d+|[a-z]+)', chunk)
                replies = self._parse_nuxt_number(reply_match.group(1)) if reply_match else 0

                # Extract timestamp
                time_match = re.search(r'createdAt:(\d+)', chunk)
                if time_match:
                    timestamp_ms = int(time_match.group(1))
                    created_at = datetime.fromtimestamp(
                        timestamp_ms / 1000, tz=timezone.utc
                    )
                else:
                    created_at = datetime.now(timezone.utc)

                # Extract text - find text:" and capture until closing "
                text_match = re.search(r'text:"((?:[^"\\]|\\.)*)"', chunk)
                if not text_match:
                    continue

                text = text_match.group(1)
                # Unescape
                text = text.replace("\\n", " ")
                text = text.replace("\\u002F", "/")
                text = text.replace('\\"', '"')
                text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
                text = re.sub(r"\s+", " ", text).strip()

                if not text or len(text) < 10:
                    continue

                tweets.append(
                    SotweTweet(
                        id=tweet_id,
                        text=text,
                        created_at=created_at,
                        username=username,
                        author_name=username,
                        likes=likes,
                        retweets=retweets,
                        replies=replies,
                        views=views if views > 0 else None,
                    )
                )

            except Exception as e:
                logger.debug(f"Failed to parse NUXT tweet {tweet_id}: {e}")
                continue

        logger.debug(f"Extracted {len(tweets)} tweets from NUXT data")
        return tweets

    def _parse_nuxt_number(self, value: str) -> int:
        """
        Parse a number from NUXT data.

        NUXT uses variable references (like 'cc', 'ae') for some values.
        We only return the number if it's a literal numeric value.

        Args:
            value: String that may be a number or variable reference

        Returns:
            Integer value, or 0 if not parseable
        """
        try:
            return int(value)
        except ValueError:
            return 0

    def _parse_tweet_card(
        self,
        card: Any,
        default_username: str,
    ) -> SotweTweet | None:
        """
        Parse a single tweet card element.

        Args:
            card: BeautifulSoup element for tweet card
            default_username: Username to use if not found in card

        Returns:
            SotweTweet object or None
        """
        # Extract tweet text
        text_elem = card.find("div", class_="tweet-text")
        if not text_elem:
            return None

        # Get text content, preserving @mentions
        text = self._extract_text_with_mentions(text_elem)
        if not text or len(text) < 20:
            return None

        # Filter out profile/bio cards (they often contain "official handle" or similar)
        text_lower = text.lower()
        bio_indicators = [
            "official handle",
            "all our social",
            "follow us",
            "official account",
            "blog: support:",  # Common bio pattern
        ]
        if any(phrase in text_lower for phrase in bio_indicators):
            logger.debug(f"Filtered out bio card: {text[:50]}...")
            return None

        # Extract tweet ID from any link containing /status/
        tweet_id = None
        for link in card.find_all("a", href=True):
            href = link.get("href", "")
            match = re.search(r"/status/(\d+)", href)
            if match:
                tweet_id = match.group(1)
                break

        # Fallback: look for ID in data attributes or other patterns
        if not tweet_id:
            # Try to find any 18+ digit number that looks like a tweet ID
            card_html = str(card)
            id_match = re.search(r'["\'"](\d{15,20})["\']', card_html)
            if id_match:
                tweet_id = id_match.group(1)

        if not tweet_id:
            return None

        # Extract author info
        username = default_username
        author_name = default_username

        profile_elem = card.find("div", class_="tweet-profile")
        if profile_elem:
            # Find the link to the user profile
            user_link = profile_elem.find("a", href=True)
            if user_link:
                href = user_link.get("href", "")
                # Extract username from href like "/nvidia"
                username_match = re.match(r"^/([^/]+)$", href)
                if username_match:
                    username = username_match.group(1)

            # Find display name
            name_elem = profile_elem.find("span", class_="font-weight-medium")
            if name_elem:
                author_name = name_elem.get_text(strip=True)

        # Extract engagement metrics
        likes = self._extract_metric_from_card(card, ["like", "heart", "favorite"])
        retweets = self._extract_metric_from_card(card, ["retweet", "repeat"])
        replies = self._extract_metric_from_card(card, ["reply", "comment"])

        # Extract timestamp
        created_at = self._extract_timestamp_from_card(card)

        return SotweTweet(
            id=tweet_id,
            text=text,
            created_at=created_at,
            username=username,
            author_name=author_name,
            likes=likes,
            retweets=retweets,
            replies=replies,
            views=None,  # Not typically shown on Sotwe
        )

    def _extract_text_with_mentions(self, text_elem: Any) -> str:
        """
        Extract tweet text, preserving @mentions.

        Args:
            text_elem: BeautifulSoup element containing tweet text

        Returns:
            Tweet text with @mentions preserved
        """
        parts = []

        for child in text_elem.descendants:
            if isinstance(child, str):
                # Clean up the string
                text = child.strip()
                if text:
                    parts.append(text)
            elif child.name == "a" and child.get("href", "").startswith("/"):
                # This is likely a @mention link
                mention_text = child.get_text(strip=True)
                if mention_text and not mention_text.startswith("@"):
                    parts.append(f"@{mention_text}")
                else:
                    parts.append(mention_text)

        # Join and clean up
        text = " ".join(parts)
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_metric_from_card(
        self,
        card: Any,
        keywords: list[str],
    ) -> int:
        """
        Extract an engagement metric from a tweet card.

        Args:
            card: BeautifulSoup element for tweet card
            keywords: Keywords to search for (e.g., ["like", "heart"])

        Returns:
            Metric count (0 if not found)
        """
        # Look for buttons/icons with metric counts
        for elem in card.find_all(["button", "span", "div"]):
            elem_text = elem.get_text(strip=True).lower()
            elem_class = " ".join(elem.get("class", []))

            # Check if this element relates to our metric
            matches_keyword = any(kw in elem_class.lower() for kw in keywords)

            if matches_keyword:
                # Look for a number in this element or nearby
                numbers = re.findall(r"\d+", elem_text)
                if numbers:
                    return int(numbers[0])

                # Check sibling elements
                for sibling in elem.find_next_siblings(limit=2):
                    sib_text = sibling.get_text(strip=True)
                    numbers = re.findall(r"\d+", sib_text)
                    if numbers:
                        return int(numbers[0])

        return 0

    def _extract_timestamp_from_card(self, card: Any) -> datetime:
        """
        Extract timestamp from a tweet card.

        Args:
            card: BeautifulSoup element for tweet card

        Returns:
            Parsed datetime (defaults to now if not found)
        """
        # Look for time elements
        time_elem = card.find("time")
        if time_elem:
            datetime_str = time_elem.get("datetime")
            if datetime_str:
                try:
                    return datetime.fromisoformat(
                        datetime_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

        # Look for relative time text like "2h", "3d", "Jan 15"
        for elem in card.find_all(["span", "a"]):
            text = elem.get_text(strip=True)
            # Match patterns like "2h", "3d", "5m"
            if re.match(r"^\d+[hdmw]$", text, re.I):
                # Relative time - return current time as approximation
                return datetime.now(timezone.utc)

        return datetime.now(timezone.utc)

    def _parse_tweets_fallback(
        self,
        html: str,
        username: str,
    ) -> list[SotweTweet]:
        """
        Fallback parsing when tweet cards aren't found.

        Extracts tweet IDs from NUXT data and text from HTML, then combines them.

        Args:
            html: Page HTML content
            username: Twitter username

        Returns:
            List of SotweTweet objects
        """
        tweets = []
        seen_ids: set[str] = set()

        # Extract tweet metadata from NUXT data (timeline.data.username)
        # Pattern: {id:"...",retweetCount:N,favoriteCount:N,...}
        tweet_metadata: dict[str, dict[str, int]] = {}
        id_pattern = r'\{id:"(\d{15,20})",retweetCount:(\d+),favoriteCount:(\d+)'
        for match in re.finditer(id_pattern, html):
            tweet_id = match.group(1)
            tweet_metadata[tweet_id] = {
                "retweets": int(match.group(2)),
                "likes": int(match.group(3)),
            }

        # Get ordered list of tweet IDs
        tweet_ids = list(tweet_metadata.keys())
        logger.debug(f"Found {len(tweet_ids)} tweet IDs in NUXT data")

        # Find text blocks that look like tweets
        soup = BeautifulSoup(html, "html.parser")

        # Look for dynamic-link-content divs which contain tweet text
        text_divs = soup.find_all("div", class_="dynamic-link-content")

        # Bio indicators to filter out
        bio_indicators = [
            "official handle",
            "all our social",
            "follow us",
            "official account",
            "blog: support",
        ]

        tweet_texts = []
        for div in text_divs:
            text = self._extract_text_with_mentions(div)
            if not text or len(text) < 20:
                continue

            # Filter out bio/profile text
            text_lower = text.lower()
            if any(phrase in text_lower for phrase in bio_indicators):
                logger.debug(f"Filtered bio text: {text[:50]}...")
                continue

            tweet_texts.append(text)

        logger.debug(f"Found {len(tweet_texts)} tweet texts after filtering")

        # Pair IDs with texts
        for i, text in enumerate(tweet_texts):
            tweet_id = tweet_ids[i] if i < len(tweet_ids) else None

            if not tweet_id:
                # Generate a pseudo-ID from text hash
                tweet_id = str(abs(hash(text)))[:18]

            if tweet_id in seen_ids:
                continue
            seen_ids.add(tweet_id)

            # Get engagement metrics if available
            metadata = tweet_metadata.get(tweet_id, {})

            tweets.append(
                SotweTweet(
                    id=tweet_id,
                    text=text,
                    created_at=datetime.now(timezone.utc),
                    username=username,
                    author_name=username,
                    likes=metadata.get("likes", 0),
                    retweets=metadata.get("retweets", 0),
                    replies=0,
                    views=None,
                )
            )

        return tweets

    # Keep old methods for backwards compatibility with tests
    def _is_tweet_like(self, data: dict[str, Any]) -> bool:
        """Check if a dictionary looks like tweet data (for test compatibility)."""
        has_text = any(
            key in data
            for key in ["text", "full_text", "rawContent"]
        )
        has_id = any(
            key in data
            for key in ["id", "id_str", "tweetId", "tweet_id"]
        )
        is_user = "screen_name" in data and "followers_count" in data
        text_value = (
            data.get("full_text")
            or data.get("text")
            or data.get("rawContent")
            or ""
        )
        has_substantial_text = len(str(text_value)) > 10
        return has_text and has_id and not is_user and has_substantial_text

    def _find_tweets_in_data(self, data: Any) -> list[dict[str, Any]]:
        """Recursively find tweets in data (for test compatibility)."""
        tweets = []
        if isinstance(data, dict):
            if self._is_tweet_like(data):
                tweets.append(data)
            for value in data.values():
                tweets.extend(self._find_tweets_in_data(value))
        elif isinstance(data, list):
            for item in data:
                tweets.extend(self._find_tweets_in_data(item))
        return tweets

    def _parse_single_tweet(
        self,
        data: dict[str, Any],
        default_username: str,
    ) -> SotweTweet | None:
        """Parse a single tweet from dict data (for test compatibility)."""
        tweet_id = (
            data.get("id_str")
            or data.get("id")
            or data.get("tweetId")
            or data.get("tweet_id")
        )
        if not tweet_id:
            return None
        tweet_id = str(tweet_id)

        text = (
            data.get("full_text")
            or data.get("text")
            or data.get("rawContent")
            or data.get("content")
            or ""
        )
        if not text:
            return None

        created_at = self._parse_timestamp(data)

        user_data = data.get("user") or data.get("author") or {}
        username = (
            user_data.get("screen_name")
            or user_data.get("username")
            or data.get("username")
            or default_username
        )
        author_name = (
            user_data.get("name")
            or data.get("author_name")
            or username
        )

        likes = self._extract_metric(data, ["favorite_count", "likes", "likeCount"])
        retweets = self._extract_metric(data, ["retweet_count", "retweets", "retweetCount"])
        replies = self._extract_metric(data, ["reply_count", "replies", "replyCount"])
        views = self._extract_metric(data, ["view_count", "views", "viewCount"])

        return SotweTweet(
            id=tweet_id,
            text=text,
            created_at=created_at,
            username=username,
            author_name=author_name,
            likes=likes,
            retweets=retweets,
            replies=replies,
            views=views if views > 0 else None,
        )

    def _parse_timestamp(self, data: dict[str, Any]) -> datetime:
        """Parse timestamp from dict data (for test compatibility)."""
        timestamp_str = (
            data.get("created_at")
            or data.get("timestamp")
            or data.get("date")
        )
        if not timestamp_str:
            return datetime.now(timezone.utc)
        try:
            if isinstance(timestamp_str, str):
                if "T" in timestamp_str:
                    return datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                return datetime.strptime(
                    timestamp_str,
                    "%a %b %d %H:%M:%S %z %Y"
                )
            elif isinstance(timestamp_str, (int, float)):
                return datetime.fromtimestamp(timestamp_str, tz=timezone.utc)
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return datetime.now(timezone.utc)

    def _extract_metric(
        self,
        data: dict[str, Any],
        keys: list[str],
    ) -> int:
        """Extract a numeric metric from dict (for test compatibility)."""
        for key in keys:
            value = data.get(key)
            if value is not None:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
        return 0


class SotweClientError(Exception):
    """Exception raised for Sotwe client errors."""

    pass
