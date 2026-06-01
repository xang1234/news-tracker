"""Generic RSS/Atom feed adapter."""

from __future__ import annotations

import html
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import feedparser
import httpx
from bs4 import BeautifulSoup

from src.config.feeds import FEEDS, Feed
from src.ingestion.base_adapter import BaseAdapter, clean_text, extract_tickers, stable_hash
from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform

logger = logging.getLogger(__name__)

_USER_AGENT = "NewsTracker/1.0 (RSS/Atom Reader)"


class FeedRateLimiter(Protocol):
    async def acquire(self) -> None:
        """Wait until one outbound request can be made."""
        ...


class FeedAdapter(BaseAdapter):
    """Fetch RSS/Atom feed entries and normalize them as article documents."""

    def __init__(
        self,
        feeds: list[Feed] | None = None,
        *,
        rate_limit: int = 20,
        max_items_per_feed: int = 50,
        recency_days: int = 7,
        fetch_timeout: float = 15.0,
        full_text_enabled: bool = True,
        rate_limiter: FeedRateLimiter | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(rate_limit=rate_limit)
        self._request_limiter = rate_limiter or self._rate_limiter
        self._feeds = list(feeds if feeds is not None else FEEDS)
        self._max_items_per_feed = max_items_per_feed
        self._recency_days = recency_days
        self._fetch_timeout = fetch_timeout
        self._full_text_enabled = full_text_enabled
        self._now = now or (lambda: datetime.now(UTC))
        self._conditional_headers: dict[str, dict[str, str]] = {}
        self._feed_health: dict[str, dict[str, object]] = {}
        self._seen_entry_ids_by_feed: dict[str, set[str]] = {}

    @property
    def platform(self) -> Platform:
        return Platform.RSS

    async def _fetch_raw(self) -> AsyncIterator[dict[str, Any]]:  # type: ignore[override,misc]
        seen_urls: set[str] = set()
        async with httpx.AsyncClient(timeout=self._fetch_timeout) as client:
            for feed in self._enabled_feeds():
                try:
                    async for raw_entry in self._fetch_feed_entries(client, feed, seen_urls):
                        yield raw_entry
                except Exception as exc:
                    self._mark_feed_health(feed, "error", str(exc))
                    logger.warning("RSS feed fetch failed for %s: %s", feed.slug, exc)
                    continue

    async def _fetch_feed_entries(
        self,
        client: httpx.AsyncClient,
        feed: Feed,
        seen_urls: set[str],
    ) -> AsyncIterator[dict[str, Any]]:
        await self._request_limiter.acquire()
        response = await client.get(feed.url, headers=self._request_headers(feed))
        if response.status_code == 304:
            self._mark_feed_health(feed, "not_modified")
            return
        response.raise_for_status()
        self._store_conditional_headers(feed, response)

        parsed = feedparser.parse(response.text)
        entries = parsed.get("entries", [])
        self._mark_feed_health(feed, "ok")
        yielded = 0

        for entry in entries:
            if yielded >= self._max_items_per_feed:
                break
            url = self._entry_url(entry)
            if not url or url in seen_urls:
                continue
            if not self._entry_is_recent(entry):
                continue
            entry_id = self._entry_identity(entry, url)
            seen_ids = self._seen_entry_ids_by_feed.setdefault(feed.slug, set())
            if entry_id in seen_ids:
                continue

            content, content_mode = await self._resolve_content(client, feed, entry, url)
            seen_urls.add(url)
            seen_ids.add(entry_id)
            yielded += 1
            yield {
                "feed": feed,
                "entry": entry,
                "url": url,
                "content": content,
                "content_mode": content_mode,
                "fetched_at": self._now(),
                "feed_health": self._feed_health.get(feed.slug, {"status": "ok"}),
            }

    def _transform(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        try:
            feed: Feed = raw["feed"]
            entry = raw["entry"]
            url = str(raw.get("url") or self._entry_url(entry))
            title = clean_text(str(entry.get("title", "")))
            content = clean_text(str(raw.get("content", "")))
            if not url or not content:
                return None

            timestamp = self._parse_timestamp(entry)
            tickers = extract_tickers(f"{title}\n\n{content}")
            word_count = len(content.split())
            raw_entry = dict(entry)

            return NormalizedDocument(
                id=f"rss_{stable_hash(url)}",
                platform=Platform.RSS,
                url=url,
                timestamp=timestamp,
                fetched_at=raw.get("fetched_at") or self._now(),
                author_id=feed.slug,
                author_name=feed.name,
                author_verified=feed.authority in {"high", "primary", "official"},
                content=content,
                content_type="article",
                title=title or None,
                engagement=EngagementMetrics(read_time_minutes=round(word_count / 200.0, 1)),
                tickers_mentioned=tickers,
                raw_data={
                    "feed": feed.to_metadata(),
                    "feed_health": dict(raw.get("feed_health") or {}),
                    "content_mode": raw.get("content_mode", "feed"),
                    "entry": raw_entry,
                },
            )
        except Exception as exc:
            logger.debug("Failed to transform RSS entry: %s", exc)
            return None

    async def _resolve_content(
        self,
        client: httpx.AsyncClient,
        feed: Feed,
        entry: Any,
        url: str,
    ) -> tuple[str, str]:
        fallback = self._entry_feed_content(entry)
        if not (feed.full_text and self._full_text_enabled and url):
            return fallback, "feed"

        try:
            await self._request_limiter.acquire()
            response = await client.get(url, headers={"User-Agent": _USER_AGENT})
            response.raise_for_status()
            article_text = self._extract_article_text(response.text)
            if article_text:
                return article_text, "article_full_text"
        except Exception as exc:
            logger.debug("RSS full-text extraction failed for %s %s: %s", feed.slug, url, exc)

        return fallback, "feed_fallback"

    def _entry_feed_content(self, entry: Any) -> str:
        for content_item in entry.get("content", []) or []:
            value = content_item.get("value") if isinstance(content_item, dict) else None
            if value:
                return self._clean_html_content(str(value))
        for field in ("summary", "description", "subtitle"):
            value = entry.get(field)
            if value:
                return self._clean_html_content(str(value))
        return clean_text(str(entry.get("title", "")))

    def _extract_article_text(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        container = soup.find("article") or soup.body or soup
        return clean_text(html.unescape(container.get_text(separator=" ")))

    def _clean_html_content(self, html_content: str) -> str:
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ")
        text = html.unescape(text)
        return clean_text(re.sub(r"\s+", " ", text))

    def _parse_timestamp(self, entry: Any) -> datetime:
        for field in ("published", "updated", "created"):
            value = entry.get(field)
            if value:
                parsed = self._parse_timestamp_string(str(value))
                if parsed is not None:
                    return parsed
            parsed_field = entry.get(f"{field}_parsed")
            if parsed_field:
                try:
                    return datetime.fromtimestamp(time.mktime(parsed_field), tz=UTC)
                except Exception:
                    continue
        return self._now()

    def _parse_timestamp_string(self, value: str) -> datetime | None:
        try:
            parsed = parsedate_to_datetime(value)
        except Exception:
            parsed = None
        if parsed is None:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _entry_is_recent(self, entry: Any) -> bool:
        if self._recency_days <= 0:
            return True
        threshold = self._now() - timedelta(days=self._recency_days)
        return self._parse_timestamp(entry) >= threshold

    def _entry_url(self, entry: Any) -> str:
        link = entry.get("link")
        if link:
            return str(link)
        for link_info in entry.get("links", []) or []:
            href = link_info.get("href") if isinstance(link_info, dict) else None
            if href:
                return str(href)
        return ""

    def _entry_identity(self, entry: Any, url: str) -> str:
        for field in ("id", "guid", "link"):
            value = entry.get(field)
            if value:
                return str(value)
        return url

    def _request_headers(self, feed: Feed) -> dict[str, str]:
        headers = {"User-Agent": _USER_AGENT}
        conditional = self._conditional_headers.get(feed.slug, {})
        if conditional.get("etag"):
            headers["If-None-Match"] = conditional["etag"]
        if conditional.get("last_modified"):
            headers["If-Modified-Since"] = conditional["last_modified"]
        return headers

    def _store_conditional_headers(self, feed: Feed, response: httpx.Response) -> None:
        cached = self._conditional_headers.setdefault(feed.slug, {})
        if response.headers.get("ETag"):
            cached["etag"] = response.headers["ETag"]
        if response.headers.get("Last-Modified"):
            cached["last_modified"] = response.headers["Last-Modified"]

    def _mark_feed_health(self, feed: Feed, status: str, error: str | None = None) -> None:
        payload: dict[str, object] = {"status": status, "checked_at": self._now().isoformat()}
        if error:
            payload["error"] = error
        self._feed_health[feed.slug] = payload

    def _enabled_feeds(self) -> list[Feed]:
        return [feed for feed in self._feeds if feed.enabled]

    async def health_check(self) -> bool:
        enabled_feeds = self._enabled_feeds()
        if not enabled_feeds:
            return False
        feed = enabled_feeds[0]
        try:
            async with httpx.AsyncClient(timeout=self._fetch_timeout) as client:
                await self._request_limiter.acquire()
                response = await client.get(feed.url, headers=self._request_headers(feed))
                if response.status_code == 304:
                    return True
                response.raise_for_status()
                parsed = feedparser.parse(response.text)
                return bool(parsed.get("entries"))
        except Exception:
            return False
