"""Tests for RSS/Atom feed URL validation helpers."""

from __future__ import annotations

import httpx
import pytest

from src.config.feed_validation import validate_feed_url
from src.config.feeds import Feed

RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Chip Wire</title>
    <item>
      <guid>chip-wire-nvda</guid>
      <title>NVIDIA expands HBM supply</title>
      <link>https://example.com/rss/nvda-hbm</link>
      <pubDate>Fri, 31 May 2026 14:30:00 GMT</pubDate>
      <description>NVIDIA says HBM demand is expanding.</description>
    </item>
  </channel>
</rss>
"""


@pytest.mark.asyncio
async def test_live_feed_validator_accepts_parseable_feed() -> None:
    feed = Feed(
        slug="chip-wire",
        name="Chip Wire",
        url="https://feeds.example.com/rss.xml",
        category="trade_press",
    )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=RSS_FEED))
    ) as client:
        result = await validate_feed_url(feed, client=client)

    assert result.ok is True
    assert result.status_code == 200
    assert result.entry_count == 1
    assert result.reason == "ok"


@pytest.mark.asyncio
async def test_live_feed_validator_catches_dead_feed_url() -> None:
    feed = Feed(
        slug="dead-feed",
        name="Dead Feed",
        url="https://feeds.example.com/dead.xml",
        category="trade_press",
    )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(404, text="missing"))
    ) as client:
        result = await validate_feed_url(feed, client=client)

    assert result.ok is False
    assert result.status_code == 404
    assert result.entry_count == 0
    assert result.reason == "http_status"
