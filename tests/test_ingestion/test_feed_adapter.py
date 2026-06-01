"""Tests for the generic RSS/Atom feed adapter."""

from __future__ import annotations

import os
import time
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
    enabled: bool = True,
) -> Feed:
    return Feed(
        slug=slug,
        name="Chip Wire",
        url=url,
        category="trade_press",
        authority="high",
        full_text=full_text,
        enabled=enabled,
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


class RecordingMetrics:
    def __init__(self) -> None:
        self.feed_fetches: list[tuple[str, str]] = []
        self.feed_documents: list[tuple[str, int]] = []

    def record_rss_feed_fetch(self, feed_slug: str, status: str) -> None:
        self.feed_fetches.append((feed_slug, status))

    def record_rss_feed_documents(self, feed_slug: str, count: int) -> None:
        self.feed_documents.append((feed_slug, count))


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
async def test_feed_health_snapshots_and_metrics_capture_success_and_parse_failure() -> None:
    good = _feed("good", url="https://feeds.example.com/good.xml")
    broken = _feed("broken", url="https://feeds.example.com/broken.xml")
    respx.get("https://feeds.example.com/good.xml").mock(
        return_value=httpx.Response(200, text=RSS_FEED)
    )
    respx.get("https://feeds.example.com/broken.xml").mock(
        return_value=httpx.Response(200, text="<html>not a feed</html>")
    )
    metrics = RecordingMetrics()
    adapter = FeedAdapter(
        feeds=[good, broken],
        now=_now,
        recency_days=7,
        metrics=metrics,
    )

    docs = await _collect_docs(adapter)
    health = {item["slug"]: item for item in adapter.feed_health_snapshots()}

    assert [doc.author_id for doc in docs] == ["good"]
    assert health["good"]["status"] == "ok"
    assert health["good"]["last_fetch_at"] == "2026-06-01T00:00:00+00:00"
    assert health["good"]["last_success_at"] == "2026-06-01T00:00:00+00:00"
    assert health["good"]["recent_document_count"] == 1
    assert health["broken"]["status"] == "parse_failure"
    assert health["broken"]["recent_document_count"] == 0
    assert "error" in health["broken"]
    assert metrics.feed_fetches == [("good", "ok"), ("broken", "parse_failure")]
    assert metrics.feed_documents == [("good", 1), ("broken", 0)]


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
        max_items_per_feed=3,
    )

    docs = await _collect_docs(adapter)

    assert [doc.url for doc in docs] == [
        "https://example.com/rss/recent-a",
        "https://example.com/rss/recent-b",
    ]


@pytest.mark.asyncio
@respx.mock
async def test_seen_entry_cache_is_bounded_per_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.ingestion.feed_adapter._MAX_SEEN_ENTRY_IDS_PER_FEED",
        2,
        raising=False,
    )
    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Chip Wire</title>
      <item>
        <guid>entry-1</guid>
        <title>Recent 1</title>
        <link>https://example.com/rss/recent-1</link>
        <pubDate>Fri, 31 May 2026 14:30:00 GMT</pubDate>
        <description>Recent 1 body.</description>
      </item>
      <item>
        <guid>entry-2</guid>
        <title>Recent 2</title>
        <link>https://example.com/rss/recent-2</link>
        <pubDate>Fri, 31 May 2026 14:31:00 GMT</pubDate>
        <description>Recent 2 body.</description>
      </item>
      <item>
        <guid>entry-3</guid>
        <title>Recent 3</title>
        <link>https://example.com/rss/recent-3</link>
        <pubDate>Fri, 31 May 2026 14:32:00 GMT</pubDate>
        <description>Recent 3 body.</description>
      </item>
    </channel></rss>
    """
    respx.get("https://feeds.example.com/rss.xml").mock(return_value=httpx.Response(200, text=rss))
    adapter = FeedAdapter(
        feeds=[_feed()],
        now=_now,
        recency_days=7,
        max_items_per_feed=3,
    )

    docs = await _collect_docs(adapter)

    assert len(docs) == 3
    seen_ids = adapter._seen_entry_ids_by_feed["chip-wire"]
    assert len(seen_ids) == 2
    assert "entry-1" not in seen_ids
    assert {"entry-2", "entry-3"} <= set(seen_ids)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="requires POSIX timezone support")
def test_parse_timestamp_treats_feedparser_struct_time_as_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        adapter = FeedAdapter(feeds=[_feed()], now=_now, recency_days=7)
        parsed = time.struct_time((2026, 5, 31, 14, 30, 0, 6, 151, 0))

        assert adapter._parse_timestamp({"published_parsed": parsed}) == datetime(
            2026,
            5,
            31,
            14,
            30,
            tzinfo=UTC,
        )
    finally:
        if original_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_tz)
        time.tzset()


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
async def test_health_check_requires_enabled_feeds() -> None:
    adapter = FeedAdapter(feeds=[_feed(enabled=True)], now=_now)
    disabled_adapter = FeedAdapter(feeds=[_feed(enabled=False)], now=_now)

    assert await adapter.health_check() is True
    assert await disabled_adapter.health_check() is False
