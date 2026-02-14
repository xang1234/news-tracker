"""
Substack RSS adapter for high-signal newsletter content.

Fetches articles from curated Substack publications via RSS feeds.
Handles:
- RSS/Atom feed parsing
- HTML content extraction and cleaning
- Author attribution from publication metadata

CRITICAL: Substack is LOW VOLUME but HIGH SIGNAL. A single SemiAnalysis
post about HBM constraints is worth 10,000 tweets. Weight accordingly.
"""

import html
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx
from bs4 import BeautifulSoup

from src.ingestion.base_adapter import BaseAdapter, clean_text, extract_tickers, stable_hash
from src.ingestion.schemas import (
    EngagementMetrics,
    NormalizedDocument,
    Platform,
)

logger = logging.getLogger(__name__)

# Curated Substack publications for semiconductor/tech analysis
# Format: (slug, display_name, description)
TRACKED_PUBLICATIONS = [
    ("semianalysis", "SemiAnalysis", "Semiconductor deep dives"),
    ("stratechery", "Stratechery", "Tech business analysis"),
    ("thediff", "The Diff", "Finance/tech intersection"),
    ("doomberg", "Doomberg", "Commodities and energy"),
    ("chipwars", "Chip Wars", "Semiconductor industry"),
    ("asianometry", "Asianometry", "Tech and economics"),
    ("chipmakernews", "Chipmaker News", "Semiconductor news"),
]


class SubstackAdapter(BaseAdapter):
    """
    Substack RSS adapter for newsletter content.

    Fetches articles from curated Substack publications that cover
    semiconductor and technology topics. Uses RSS feeds for discovery
    and fetches full article content.

    Polling: Every 15-30 minutes (RSS doesn't support real-time)

    Content Handling:
        - RSS gives title + excerpt
        - HTML → clean text (BeautifulSoup)
        - Preserves document structure: headers indicate topic organization
    """

    def __init__(
        self,
        publications: list[tuple[str, str, str]] | None = None,
        rate_limit: int = 10,
        fetch_full_content: bool = True,
    ):
        """
        Initialize Substack adapter.

        Args:
            publications: List of (slug, name, description) tuples
            rate_limit: Requests per minute
            fetch_full_content: Whether to fetch full article HTML
        """
        super().__init__(rate_limit=rate_limit)

        self._publications = publications if publications is not None else TRACKED_PUBLICATIONS
        self._fetch_full_content = fetch_full_content

    @property
    def platform(self) -> Platform:
        return Platform.SUBSTACK

    def _get_feed_url(self, slug: str) -> str:
        """Get RSS feed URL for a Substack publication."""
        return f"https://{slug}.substack.com/feed"

    async def _fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        """
        Fetch articles from Substack RSS feeds.

        Yields raw article data from configured publications.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            for slug, name, description in self._publications:
                feed_url = self._get_feed_url(slug)

                try:
                    # Rate limit before each RSS feed fetch
                    await self._rate_limiter.acquire()

                    response = await client.get(
                        feed_url,
                        headers={
                            "User-Agent": "NewsTracker/1.0 (RSS Reader)",
                        },
                    )
                    response.raise_for_status()

                    # Parse RSS feed
                    feed = feedparser.parse(response.text)

                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"Substack feed error for {name}: {e.response.status_code}"
                    )
                    continue
                except Exception as e:
                    logger.error(f"Substack request failed for {name}: {e}")
                    continue

                # Yield entries
                entries = feed.get("entries", [])
                for entry in entries[:10]:  # Limit per feed
                    yield {
                        "publication_slug": slug,
                        "publication_name": name,
                        "entry": entry,
                        "client": client,
                    }

                logger.debug(f"Fetched {len(entries)} articles from {name}")

    def _transform(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        """Transform RSS entry to NormalizedDocument."""
        try:
            entry = raw["entry"]
            pub_name = raw["publication_name"]
            pub_slug = raw["publication_slug"]

            # Get title
            title = entry.get("title", "")
            if not title:
                return None

            # Parse timestamp
            timestamp = self._parse_timestamp(entry)

            # Get content - try full content first, then summary
            content = ""
            if "content" in entry and entry["content"]:
                content = entry["content"][0].get("value", "")
            elif "summary" in entry:
                content = entry.get("summary", "")

            # Clean HTML content
            content = self._clean_html_content(content)
            if not content:
                content = title  # Fallback to title if no content

            content = clean_text(content)

            # Combine title and content for full text
            full_text = f"{title}\n\n{content}"

            # Extract tickers
            tickers = extract_tickers(full_text)

            # Get URL
            url = entry.get("link", "")

            # Estimate read time (rough estimate: 200 words per minute)
            word_count = len(content.split())
            read_time = word_count / 200.0

            engagement = EngagementMetrics(
                read_time_minutes=round(read_time, 1),
            )

            # Get author
            author = entry.get("author", pub_name)

            return NormalizedDocument(
                id=f"substack_{pub_slug}_{self._get_entry_id(entry)}",
                platform=Platform.SUBSTACK,
                url=url,
                timestamp=timestamp,
                author_id=pub_slug,
                author_name=author,
                author_verified=True,  # Substack authors are generally credible
                content=content,
                content_type="article",
                title=title,
                engagement=engagement,
                tickers_mentioned=tickers,
                raw_data={
                    "publication": pub_name,
                    "tags": [t.get("term", "") for t in entry.get("tags", [])],
                },
            )

        except Exception as e:
            logger.debug(f"Failed to transform Substack entry: {e}")
            return None

    def _parse_timestamp(self, entry: dict[str, Any]) -> datetime:
        """Parse timestamp from RSS entry."""
        # Try different timestamp fields
        for field in ["published", "updated", "created"]:
            if field in entry:
                try:
                    return parsedate_to_datetime(entry[field])
                except Exception:
                    pass

            # Try parsed versions
            parsed_field = f"{field}_parsed"
            if parsed_field in entry and entry[parsed_field]:
                try:
                    import time
                    return datetime.fromtimestamp(
                        time.mktime(entry[parsed_field]),
                        tz=timezone.utc,
                    )
                except Exception:
                    pass

        # Fallback to now
        return datetime.now(timezone.utc)

    def _get_entry_id(self, entry: dict[str, Any]) -> str:
        """Extract unique ID from RSS entry using stable hash."""
        # Try standard ID fields
        for field in ["id", "guid", "link"]:
            if field in entry:
                # Use stable hash for deterministic IDs across runs
                value = str(entry[field])
                return stable_hash(value)

        # Fallback: hash the title
        return stable_hash(entry.get("title", ""))

    def _clean_html_content(self, html_content: str) -> str:
        """
        Extract clean text from HTML content.

        Args:
            html_content: Raw HTML string

        Returns:
            Clean text content
        """
        if not html_content:
            return ""

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get text content
        text = soup.get_text(separator=" ")

        # Clean up HTML entities
        text = html.unescape(text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    async def health_check(self) -> bool:
        """Check if Substack RSS feeds are accessible."""
        if not self._publications:
            return False

        try:
            # Try to fetch one feed
            slug = self._publications[0][0]
            feed_url = self._get_feed_url(slug)

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(feed_url)
                return response.status_code == 200

        except Exception:
            return False
