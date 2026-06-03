"""arXiv Atom API provider for research innovation evidence."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime

from src.innovation.openalex import ResearchHTTPClient, ResearchRateLimiter
from src.innovation.research_schemas import (
    ARXIV_QUERY_URL,
    ResearchProviderResponseError,
    ResearchQuery,
    ResearchRecord,
)

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"


class ArxivResearchProvider:
    """arXiv Atom query provider using start/max_results paging."""

    def __init__(
        self,
        http_client: ResearchHTTPClient,
        *,
        rate_limiter: ResearchRateLimiter | None = None,
        base_url: str = ARXIV_QUERY_URL,
        page_size: int = 100,
        max_pages: int = 100,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        self.http_client = http_client
        self.rate_limiter = rate_limiter
        self.base_url = base_url
        self.page_size = page_size
        self.max_pages = max_pages

    async def fetch_records(
        self,
        query: ResearchQuery,
        *,
        fetched_at: datetime | None = None,
    ) -> list[ResearchRecord]:
        observed_at = fetched_at or datetime.now(UTC)
        records: list[ResearchRecord] = []
        search_query = _arxiv_search_query(query)

        for page in range(self.max_pages):
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire()
            start = page * self.page_size
            response = await self.http_client.get(
                self.base_url,
                params={
                    "search_query": search_query,
                    "start": str(start),
                    "max_results": str(self.page_size),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
            )
            entries, total = _parse_feed(getattr(response, "text", ""), fetched_at=observed_at)
            records.extend(entries)
            if _is_last_page(
                entries,
                records_seen=len(records),
                page_size=self.page_size,
                total=total,
            ):
                break
        return records


def _arxiv_search_query(query: ResearchQuery) -> str:
    parts = [
        f'all:"{term.strip()}"' for term in [*query.topics, *query.institutions] if term.strip()
    ]
    parts.extend(f"cat:{category}" for category in query.categories if category.strip())
    if query.start is not None or query.end is not None:
        start = query.start.strftime("%Y%m%d0000") if query.start else "*"
        end = query.end.strftime("%Y%m%d2359") if query.end else "*"
        parts.append(f"submittedDate:[{start} TO {end}]")
    return " AND ".join(parts)


def _parse_feed(feed_text: str, *, fetched_at: datetime) -> tuple[list[ResearchRecord], int | None]:
    try:
        root = ET.fromstring(feed_text)
    except ET.ParseError as exc:
        raise ResearchProviderResponseError("arXiv returned invalid Atom XML") from exc
    total = _int_text(root.findtext(f"{OPENSEARCH_NS}totalResults"))
    records = [
        _record_from_entry(entry, fetched_at=fetched_at)
        for entry in root.findall(f"{ATOM_NS}entry")
    ]
    return records, total


def _is_last_page(
    entries: list[ResearchRecord],
    *,
    records_seen: int,
    page_size: int,
    total: int | None,
) -> bool:
    if not entries or len(entries) < page_size:
        return True
    return total is not None and records_seen >= total


def _record_from_entry(entry: ET.Element, *, fetched_at: datetime) -> ResearchRecord:
    arxiv_id = _arxiv_id(entry.findtext(f"{ATOM_NS}id") or "")
    categories = [
        category.attrib["term"]
        for category in entry.findall(f"{ATOM_NS}category")
        if category.attrib.get("term")
    ]
    return ResearchRecord(
        source="arxiv",
        record_id=arxiv_id,
        title=_clean_text(entry.findtext(f"{ATOM_NS}title") or ""),
        abstract=_clean_text(entry.findtext(f"{ATOM_NS}summary") or ""),
        authors=[
            _clean_text(author.findtext(f"{ATOM_NS}name") or "")
            for author in entry.findall(f"{ATOM_NS}author")
        ],
        institutions=[],
        topics=[],
        categories=categories,
        published_date=_parse_datetime_date(entry.findtext(f"{ATOM_NS}published") or ""),
        url=_entry_url(entry),
        doi=_clean_text(entry.findtext(f"{ARXIV_NS}doi") or "") or None,
        arxiv_id=arxiv_id,
        source_lineage={"source": "arxiv", "arxiv_id": arxiv_id},
        fetched_at=fetched_at,
        metadata={"raw_provider": "arxiv"},
    )


def _entry_url(entry: ET.Element) -> str:
    for link in entry.findall(f"{ATOM_NS}link"):
        if link.attrib.get("rel") == "alternate" and link.attrib.get("href"):
            return link.attrib["href"]
    return entry.findtext(f"{ATOM_NS}id") or ""


def _arxiv_id(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1]


def _parse_datetime_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return date.min


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _int_text(value: str | None) -> int | None:
    try:
        return int(value or "")
    except ValueError:
        return None
