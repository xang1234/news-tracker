"""Tests for the generic RSS/Atom feed adapter."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from src.config.feeds import Feed
from src.ingestion.feed_adapter import FeedAdapter
from src.ingestion.schemas import NormalizedDocument, Platform

RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Chip Wire</title>
    <item>
      <guid>chip-wire-nvda</guid>
      <title>NVIDIA expands HBM supply</title>
      <link>https://example.com/rss/nvda-hbm</link>
      <pubDate>Fri, 31 May 2026 14:30:00 GMT</pubDate>
      <description><![CDATA[NVIDIA says $NVDA secured more HBM capacity.]]></description>
    </item>
  </channel>
</rss>
"""

ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Tech</title>
  <entry>
    <id>tag:example.com,2026:tsmc-cowos</id>
    <title>TSMC adds CoWoS capacity</title>
    <link href="https://example.com/atom/tsmc-cowos" />
    <updated>2026-05-30T10:00:00Z</updated>
    <summary>TSMC capacity update references NVDA demand.</summary>
  </entry>
</feed>
"""


def _now() -> datetime:
    return datetime(2026, 6, 1, tzinfo=UTC)


def _feed(
    slug: str = "chip-wire",
    *,
    url: str = "https://feeds.example.com/rss.xml",
    full_text: bool = False,
) -> Feed:
    return Feed(
        slug=slug,
        name="Chip Wire",
        url=url,
        category="trade_press",
        authority="high",
        full_text=full_text,
    )


async def _collect_docs(adapter: FeedAdapter) -> list[NormalizedDocument]:
    docs: list[NormalizedDocument] = []
    async for doc in adapter.fetch():
        docs.append(doc)
    return docs


class CountingLimiter:
    def __init__(self) -> None:
        self.calls = 0

    async def acquire(self) -> None:
        self.calls += 1


@pytest.mark.asyncio
@respx.mock
async def test_rss_feed_entries_transform_to_normalized_documents() -> None:
    respx.get("https://feeds.example.com/rss.xml").mock(
        return_value=httpx.Response(
            200,
            text=RSS_FEED,
            headers={"ETag": '"rss-v1"', "Last-Modified": "Fri, 31 May 2026 15:00:00 GMT"},
        )
    )
    limiter = CountingLimiter()
    adapter = FeedAdapter(
        feeds=[_feed()],
        now=_now,
        rate_limiter=limiter,
        recency_days=7,
    )

    docs = await _collect_docs(adapter)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.platform == Platform.RSS
    assert doc.content_type == "article"
    assert doc.id.startswith("rss_")
    assert doc.url == "https://example.com/rss/nvda-hbm"
    assert doc.title == "NVIDIA expands HBM supply"
    assert doc.timestamp == datetime(2026, 5, 31, 14, 30, tzinfo=UTC)
    assert doc.author_id == "chip-wire"
    assert doc.author_name == "Chip Wire"
    assert "NVIDIA says $NVDA secured more HBM capacity." in doc.content
    assert "NVDA" in doc.tickers_mentioned
    assert doc.raw_data["feed"]["category"] == "trade_press"
    assert doc.raw_data["feed"]["authority"] == "high"
    assert doc.raw_data["feed"]["full_text"] is False
    assert doc.raw_data["feed_health"]["status"] == "ok"
    assert limiter.calls == 1


@pytest.mark.asyncio
@respx.mock
async def test_atom_feed_entries_transform_to_normalized_documents() -> None:
    respx.get("https://feeds.example.com/rss.xml").mock(
        return_value=httpx.Response(200, text=ATOM_FEED)
    )
    adapter = FeedAdapter(feeds=[_feed()], now=_now, recency_days=7)

    docs = await _collect_docs(adapter)

    assert len(docs) == 1
    assert docs[0].platform == Platform.RSS
    assert docs[0].url == "https://example.com/atom/tsmc-cowos"
    assert docs[0].title == "TSMC adds CoWoS capacity"
    assert docs[0].timestamp == datetime(2026, 5, 30, 10, 0, tzinfo=UTC)


@pytest.mark.asyncio
@respx.mock
async def test_conditional_get_headers_skip_unchanged_feed() -> None:
    route = respx.get("https://feeds.example.com/rss.xml").mock(
        side_effect=[
            httpx.Response(
                200,
                text=RSS_FEED,
                headers={
                    "ETag": '"rss-v1"',
                    "Last-Modified": "Fri, 31 May 2026 15:00:00 GMT",
                },
            ),
            httpx.Response(304),
        ]
    )
    adapter = FeedAdapter(feeds=[_feed()], now=_now, recency_days=7)

    first_docs = await _collect_docs(adapter)
    second_docs = await _collect_docs(adapter)

    assert len(first_docs) == 1
    assert second_docs == []
    second_request = route.calls[1].request
    assert second_request.headers["if-none-match"] == '"rss-v1"'
    assert second_request.headers["if-modified-since"] == "Fri, 31 May 2026 15:00:00 GMT"


@pytest.mark.asyncio
@respx.mock
async def test_recency_filter_per_feed_cap_and_in_cycle_dedupe() -> None:
    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Chip Wire</title>
      <item>
        <guid>old</guid>
        <title>Old news</title>
        <link>https://example.com/rss/old</link>
        <pubDate>Fri, 01 Jan 2021 00:00:00 GMT</pubDate>
        <description>Too old.</description>
      </item>
      <item>
        <guid>recent-a</guid>
        <title>Recent A</title>
        <link>https://example.com/rss/recent-a</link>
        <pubDate>Fri, 31 May 2026 14:30:00 GMT</pubDate>
        <description>Recent A body.</description>
      </item>
      <item>
        <guid>recent-a-duplicate</guid>
        <title>Recent A duplicate</title>
        <link>https://example.com/rss/recent-a</link>
        <pubDate>Fri, 31 May 2026 14:31:00 GMT</pubDate>
        <description>Duplicate body.</description>
      </item>
      <item>
        <guid>recent-b</guid>
        <title>Recent B</title>
        <link>https://example.com/rss/recent-b</link>
        <pubDate>Fri, 31 May 2026 14:32:00 GMT</pubDate>
        <description>Recent B body.</description>
      </item>
    </channel></rss>
    """
    respx.get("https://feeds.example.com/rss.xml").mock(return_value=httpx.Response(200, text=rss))
    adapter = FeedAdapter(
        feeds=[_feed()],
        now=_now,
        recency_days=7,
        max_items_per_feed=1,
    )

    docs = await _collect_docs(adapter)

    assert [doc.url for doc in docs] == ["https://example.com/rss/recent-a"]


@pytest.mark.asyncio
@respx.mock
async def test_feed_and_article_failures_do_not_abort_adapter_run() -> None:
    broken = _feed("broken", url="https://feeds.example.com/broken.xml")
    full_text = _feed("full", url="https://feeds.example.com/full.xml", full_text=True)
    respx.get("https://feeds.example.com/broken.xml").mock(
        return_value=httpx.Response(500, text="server error")
    )
    respx.get("https://feeds.example.com/full.xml").mock(
        return_value=httpx.Response(200, text=RSS_FEED)
    )
    respx.get("https://example.com/rss/nvda-hbm").mock(
        return_value=httpx.Response(503, text="article unavailable")
    )
    adapter = FeedAdapter(
        feeds=[broken, full_text],
        now=_now,
        recency_days=7,
        full_text_enabled=True,
    )

    docs = await _collect_docs(adapter)

    assert len(docs) == 1
    assert docs[0].author_id == "full"
    assert "NVIDIA says $NVDA secured more HBM capacity." in docs[0].content
    assert docs[0].raw_data["content_mode"] == "feed_fallback"


@pytest.mark.asyncio
@respx.mock
async def test_full_text_fetch_uses_article_body_when_available() -> None:
    respx.get("https://feeds.example.com/rss.xml").mock(
        return_value=httpx.Response(200, text=RSS_FEED)
    )
    respx.get("https://example.com/rss/nvda-hbm").mock(
        return_value=httpx.Response(
            200,
            text=(
                "<html><body><article><p>Full article text names NVDA and CoWoS.</p>"
                "</article></body></html>"
            ),
        )
    )
    adapter = FeedAdapter(
        feeds=[_feed(full_text=True)],
        now=_now,
        recency_days=7,
        full_text_enabled=True,
    )

    docs = await _collect_docs(adapter)

    assert len(docs) == 1
    assert docs[0].content == "Full article text names NVDA and CoWoS."
    assert docs[0].raw_data["content_mode"] == "article_full_text"


@pytest.mark.asyncio
@respx.mock
async def test_health_check_requires_one_parseable_enabled_feed() -> None:
    respx.get("https://feeds.example.com/rss.xml").mock(
        return_value=httpx.Response(200, text=RSS_FEED)
    )
    adapter = FeedAdapter(feeds=[_feed()], now=_now)

    assert await adapter.health_check() is True
