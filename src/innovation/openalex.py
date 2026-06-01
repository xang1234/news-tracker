"""OpenAlex Works provider for research innovation evidence."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Protocol

from src.innovation.research_schemas import (
    OPENALEX_WORKS_URL,
    ResearchProviderResponseError,
    ResearchQuery,
    ResearchRecord,
)


class ResearchHTTPClient(Protocol):
    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Fetch provider records."""
        ...


class ResearchRateLimiter(Protocol):
    async def acquire(self) -> None:
        """Wait until another provider request is allowed."""
        ...


class OpenAlexResearchProvider:
    """OpenAlex Works API provider using cursor pagination."""

    def __init__(
        self,
        http_client: ResearchHTTPClient,
        *,
        rate_limiter: ResearchRateLimiter | None = None,
        base_url: str = OPENALEX_WORKS_URL,
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
        cursor: str | None = "*"
        records: list[ResearchRecord] = []

        for _ in range(self.max_pages):
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire()
            response = await self.http_client.get(
                self.base_url,
                params=_openalex_params(query, cursor=cursor, page_size=self.page_size),
            )
            payload = _response_json_dict(response)
            records.extend(
                _record_from_work(work, fetched_at=observed_at)
                for work in _payload_results(payload)
            )
            next_cursor = _next_cursor(payload)
            if not next_cursor:
                break
            cursor = next_cursor
        return records


def _openalex_params(
    query: ResearchQuery,
    *,
    cursor: str | None,
    page_size: int,
) -> dict[str, str]:
    params = {
        "per-page": str(page_size),
        "cursor": cursor or "*",
    }
    text_terms = _query_text_terms(query)
    if text_terms:
        params["search"] = " ".join(text_terms)
    filters = []
    if query.start is not None:
        filters.append(f"from_publication_date:{query.start.isoformat()}")
    if query.end is not None:
        filters.append(f"to_publication_date:{query.end.isoformat()}")
    if filters:
        params["filter"] = ",".join(filters)
    return params


def _query_text_terms(query: ResearchQuery) -> list[str]:
    return [
        value.strip()
        for value in [*query.topics, *query.categories, *query.institutions]
        if value.strip()
    ]


def _response_json_dict(response: Any) -> dict[str, Any]:
    status_code = getattr(response, "status_code", 200)
    if status_code >= 400:
        raise ResearchProviderResponseError(f"OpenAlex returned HTTP {status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ResearchProviderResponseError("OpenAlex returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ResearchProviderResponseError("OpenAlex returned non-object JSON")
    return payload


def _payload_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ResearchProviderResponseError("OpenAlex results were not a list")
    return [row for row in rows if isinstance(row, dict)]


def _next_cursor(payload: dict[str, Any]) -> str | None:
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        return None
    cursor = meta.get("next_cursor")
    return str(cursor) if cursor else None


def _record_from_work(work: dict[str, Any], *, fetched_at: datetime) -> ResearchRecord:
    institutions, institution_ids = _authorship_institutions(work.get("authorships"))
    return ResearchRecord(
        source="openalex",
        record_id=str(work.get("id") or ""),
        title=str(work.get("display_name") or ""),
        abstract=str(work.get("abstract") or ""),
        authors=_authorship_authors(work.get("authorships")),
        institutions=institutions,
        topics=_work_topics(work),
        categories=[],
        published_date=_parse_date(str(work.get("publication_date") or "")),
        url=_work_url(work),
        doi=_normalize_doi(work.get("doi")),
        arxiv_id=None,
        source_lineage={
            "source": "openalex",
            "openalex_id": work.get("id"),
            "openalex_institution_ids": institution_ids,
        },
        fetched_at=fetched_at,
        metadata={"raw_provider": "openalex"},
    )


def _authorship_authors(value: Any) -> list[str]:
    authors = []
    for authorship in value if isinstance(value, list) else []:
        author = authorship.get("author", {}) if isinstance(authorship, dict) else {}
        name = author.get("display_name") if isinstance(author, dict) else None
        if name:
            authors.append(str(name))
    return _unique(authors)


def _authorship_institutions(value: Any) -> tuple[list[str], list[str]]:
    names: list[str] = []
    ids: list[str] = []
    for authorship in value if isinstance(value, list) else []:
        institutions = authorship.get("institutions", {}) if isinstance(authorship, dict) else []
        for institution in institutions if isinstance(institutions, list) else []:
            if not isinstance(institution, dict):
                continue
            if institution.get("display_name"):
                names.append(str(institution["display_name"]))
            if institution.get("id"):
                ids.append(str(institution["id"]))
    return _unique(names), _unique(ids)


def _work_topics(work: dict[str, Any]) -> list[str]:
    topics: list[str] = []
    for key in ("topics", "concepts"):
        values = work.get(key, [])
        for item in values if isinstance(values, list) else []:
            if isinstance(item, dict) and item.get("display_name"):
                topics.append(str(item["display_name"]))
    return _unique(topics)


def _work_url(work: dict[str, Any]) -> str:
    location = work.get("primary_location", {})
    if isinstance(location, dict) and location.get("landing_page_url"):
        return str(location["landing_page_url"])
    return str(work.get("id") or "")


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return date.min


def _normalize_doi(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    return text.removeprefix("https://doi.org/").lower()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value.strip()))
