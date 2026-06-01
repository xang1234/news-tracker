"""Tests for OpenAlex and arXiv research metadata providers."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from src.innovation.research import (
    ArxivResearchProvider,
    OpenAlexResearchProvider,
    ResearchQuery,
)


class FakeHTTPClient:
    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self._responses.pop(0)


class FakeRateLimiter:
    def __init__(self) -> None:
        self.acquired = 0

    async def acquire(self) -> None:
        self.acquired += 1


def _openalex_work(work_id: str, *, doi: str | None = None) -> dict[str, Any]:
    return {
        "id": work_id,
        "doi": doi,
        "display_name": "Chiplet routing for AI accelerators",
        "publication_date": "2026-05-10",
        "primary_location": {"landing_page_url": "https://example.org/work"},
        "authorships": [
            {
                "author": {"display_name": "Ada Researcher"},
                "institutions": [
                    {
                        "display_name": "NVIDIA Research",
                        "id": "https://openalex.org/I123",
                    }
                ],
            }
        ],
        "concepts": [{"display_name": "Artificial intelligence", "score": 0.91}],
        "topics": [{"display_name": "AI accelerators", "score": 0.87}],
    }


@pytest.mark.asyncio
async def test_openalex_provider_uses_cursor_pagination_filters_and_lineage() -> None:
    client = FakeHTTPClient(
        httpx.Response(
            200,
            json={
                "meta": {"next_cursor": "next-page"},
                "results": [_openalex_work("https://openalex.org/W1", doi="10.1000/chiplet")],
            },
        ),
        httpx.Response(200, json={"meta": {"next_cursor": None}, "results": []}),
    )
    limiter = FakeRateLimiter()
    provider = OpenAlexResearchProvider(client, rate_limiter=limiter, page_size=1)

    records = await provider.fetch_records(
        ResearchQuery(
            topics=["AI accelerators"],
            institutions=["NVIDIA Research"],
            start=date(2026, 5, 1),
            end=date(2026, 6, 1),
        ),
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert [record.record_id for record in records] == ["https://openalex.org/W1"]
    assert records[0].source == "openalex"
    assert records[0].doi == "10.1000/chiplet"
    assert records[0].authors == ["Ada Researcher"]
    assert records[0].institutions == ["NVIDIA Research"]
    assert records[0].topics == ["AI accelerators", "Artificial intelligence"]
    assert records[0].source_lineage["openalex_institution_ids"] == ["https://openalex.org/I123"]
    assert limiter.acquired == 2
    assert client.calls[0]["url"] == "https://api.openalex.org/works"
    assert client.calls[0]["params"]["cursor"] == "*"
    assert client.calls[1]["params"]["cursor"] == "next-page"
    assert client.calls[0]["params"]["per-page"] == "1"
    assert client.calls[0]["params"]["search"] == "AI accelerators NVIDIA Research"
    assert client.calls[0]["params"]["filter"] == (
        "from_publication_date:2026-05-01,to_publication_date:2026-06-01"
    )


@pytest.mark.asyncio
async def test_openalex_provider_searches_all_text_criteria_to_avoid_broad_fetches() -> None:
    client = FakeHTTPClient(httpx.Response(200, json={"meta": {}, "results": []}))
    provider = OpenAlexResearchProvider(client)

    await provider.fetch_records(
        ResearchQuery(categories=["cs.AR"], institutions=["NVIDIA Research"])
    )

    assert client.calls[0]["params"]["search"] == "cs.AR NVIDIA Research"


def _arxiv_feed(*entries: str, total: int | None = 1) -> str:
    total_xml = (
        f"<opensearch:totalResults>{total}</opensearch:totalResults>"
        if total is not None
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  {total_xml}
  {''.join(entries)}
</feed>"""


def _arxiv_entry(arxiv_id: str) -> str:
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <title>EDA placement for AI accelerators</title>
    <summary>Routing and placement for accelerator chiplets.</summary>
    <published>2026-05-20T12:00:00Z</published>
    <updated>2026-05-21T12:00:00Z</updated>
    <author><name>Grace Author</name></author>
    <category term="cs.AR" />
    <category term="cs.LG" />
    <link href="http://arxiv.org/abs/{arxiv_id}" rel="alternate" />
    <arxiv:doi>10.2000/arxiv-chiplet</arxiv:doi>
  </entry>"""


@pytest.mark.asyncio
async def test_arxiv_provider_parses_atom_feed_and_respects_start_pagination() -> None:
    client = FakeHTTPClient(
        httpx.Response(200, text=_arxiv_feed(_arxiv_entry("2605.12345"), total=2)),
        httpx.Response(200, text=_arxiv_feed(total=2)),
    )
    limiter = FakeRateLimiter()
    provider = ArxivResearchProvider(client, rate_limiter=limiter, page_size=1)

    records = await provider.fetch_records(
        ResearchQuery(
            topics=["AI accelerators"],
            categories=["cs.AR"],
            start=date(2026, 5, 1),
            end=date(2026, 6, 1),
        ),
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert [record.record_id for record in records] == ["2605.12345"]
    assert records[0].source == "arxiv"
    assert records[0].arxiv_id == "2605.12345"
    assert records[0].doi == "10.2000/arxiv-chiplet"
    assert records[0].authors == ["Grace Author"]
    assert records[0].categories == ["cs.AR", "cs.LG"]
    assert records[0].published_date == date(2026, 5, 20)
    assert limiter.acquired == 2
    assert client.calls[0]["url"] == "https://export.arxiv.org/api/query"
    assert client.calls[0]["params"]["start"] == "0"
    assert client.calls[1]["params"]["start"] == "1"
    assert client.calls[0]["params"]["max_results"] == "1"
    assert "all:\"AI accelerators\"" in client.calls[0]["params"]["search_query"]
    assert "cat:cs.AR" in client.calls[0]["params"]["search_query"]
    assert client.calls[0]["params"]["sortBy"] == "submittedDate"


@pytest.mark.asyncio
async def test_arxiv_provider_continues_when_total_results_is_absent() -> None:
    client = FakeHTTPClient(
        httpx.Response(200, text=_arxiv_feed(_arxiv_entry("2605.12345"), total=None)),
        httpx.Response(200, text=_arxiv_feed(total=None)),
    )
    provider = ArxivResearchProvider(client, page_size=1)

    records = await provider.fetch_records(ResearchQuery(topics=["AI accelerators"]))

    assert [record.record_id for record in records] == ["2605.12345"]
    assert [call["params"]["start"] for call in client.calls] == ["0", "1"]


@pytest.mark.asyncio
async def test_arxiv_provider_searches_institutions_to_avoid_empty_queries() -> None:
    client = FakeHTTPClient(httpx.Response(200, text=_arxiv_feed(total=0)))
    provider = ArxivResearchProvider(client)

    await provider.fetch_records(ResearchQuery(institutions=["NVIDIA Research"]))

    assert client.calls[0]["params"]["search_query"] == 'all:"NVIDIA Research"'
