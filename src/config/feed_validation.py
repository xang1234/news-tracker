"""Live validation helpers for RSS/Atom feed URLs."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import feedparser
import httpx

from src.config.feeds import FEEDS, Feed

_USER_AGENT = "NewsTracker/1.0 feed validation"


@dataclass(frozen=True)
class LiveFeedCheck:
    """Result of fetching and parsing one feed URL."""

    feed_slug: str
    url: str
    ok: bool
    reason: str
    status_code: int | None = None
    entry_count: int = 0
    title: str = ""
    final_url: str = ""
    error: str = ""


async def validate_feed_url(
    feed: Feed,
    *,
    client: httpx.AsyncClient,
) -> LiveFeedCheck:
    """Fetch one feed and confirm it returns parseable RSS/Atom entries."""
    try:
        response = await client.get(feed.url)
    except httpx.HTTPError as exc:
        return LiveFeedCheck(
            feed_slug=feed.slug,
            url=feed.url,
            ok=False,
            reason="network_error",
            error=str(exc),
        )

    final_url = str(response.url)
    if response.status_code >= 400:
        return LiveFeedCheck(
            feed_slug=feed.slug,
            url=feed.url,
            ok=False,
            reason="http_status",
            status_code=response.status_code,
            final_url=final_url,
        )

    parsed = feedparser.parse(response.content)
    entry_count = len(parsed.entries)
    title = str(parsed.feed.get("title", "")) if parsed.feed else ""
    if parsed.bozo and entry_count == 0:
        return LiveFeedCheck(
            feed_slug=feed.slug,
            url=feed.url,
            ok=False,
            reason="parse_error",
            status_code=response.status_code,
            entry_count=entry_count,
            title=title,
            final_url=final_url,
            error=str(parsed.bozo_exception),
        )
    if entry_count == 0:
        return LiveFeedCheck(
            feed_slug=feed.slug,
            url=feed.url,
            ok=False,
            reason="empty_feed",
            status_code=response.status_code,
            entry_count=entry_count,
            title=title,
            final_url=final_url,
        )

    return LiveFeedCheck(
        feed_slug=feed.slug,
        url=feed.url,
        ok=True,
        reason="ok",
        status_code=response.status_code,
        entry_count=entry_count,
        title=title,
        final_url=final_url,
    )


async def validate_live_feed_urls(
    feeds: Sequence[Feed] = FEEDS,
    *,
    include_disabled: bool = False,
    timeout: float = 20.0,
) -> list[LiveFeedCheck]:
    """Validate every enabled feed URL over HTTP."""
    selected_feeds = [feed for feed in feeds if include_disabled or feed.enabled]
    headers = {"User-Agent": _USER_AGENT}
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers=headers,
        timeout=timeout,
    ) as client:
        return list(
            await asyncio.gather(
                *(validate_feed_url(feed, client=client) for feed in selected_feeds)
            )
        )
