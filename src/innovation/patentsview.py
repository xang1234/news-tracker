"""USPTO PatentsView/ODP provider and bulk snapshot loader."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any, Protocol

from src.innovation.patent_schemas import (
    ODP_PATENT_SEARCH_URL,
    PATENT_SOURCE_BULK,
    PATENT_SOURCE_ODP,
    MissingPatentProviderCredentialError,
    PatentProviderResponseError,
    PatentQuery,
    PatentRecord,
    StalePatentSnapshotError,
)


class PatentHTTPClient(Protocol):
    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Fetch patent search results."""
        ...


class PatentsViewProvider:
    """USPTO ODP search provider for PatentsView-transition patent data."""

    def __init__(
        self,
        http_client: PatentHTTPClient,
        *,
        api_key: str | None = None,
        base_url: str = ODP_PATENT_SEARCH_URL,
        page_size: int = 100,
        max_pages: int = 100,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        self.http_client = http_client
        self.api_key = api_key
        self.base_url = base_url
        self.page_size = page_size
        self.max_pages = max_pages

    async def fetch_patents(
        self,
        query: PatentQuery,
        *,
        fetched_at: datetime | None = None,
    ) -> list[PatentRecord]:
        """Fetch patent/application records with offset pagination."""
        if not self.api_key:
            raise MissingPatentProviderCredentialError("USPTO ODP API key is required")

        observed_at = fetched_at or datetime.now(UTC)
        records: list[PatentRecord] = []
        query_text = _build_odp_query(query)

        for page in range(self.max_pages):
            response = await self.http_client.get(
                self.base_url,
                params=_page_params(query_text, page=page, page_size=self.page_size),
                headers={"X-API-KEY": self.api_key},
            )
            payload = _response_json_dict(response)
            batch = [
                _record_from_odp_row(row, fetched_at=observed_at, source_url=self.base_url)
                for row in _payload_rows(payload)
            ]
            records.extend(batch)

            total = _payload_total(payload)
            if _is_last_page(
                batch,
                records_seen=len(records),
                page_size=self.page_size,
                total=total,
            ):
                break
        return records


def load_patentsview_bulk_snapshot(
    rows: list[dict[str, Any]],
    *,
    snapshot_date: date,
    fetched_at: datetime,
    max_age_days: int = 120,
    allow_stale: bool = False,
) -> list[PatentRecord]:
    """Normalize rows from PatentsView/ODP bulk downloads with staleness checks."""
    age_days = (fetched_at.date() - snapshot_date).days
    is_stale = age_days > max_age_days
    if is_stale and not allow_stale:
        raise StalePatentSnapshotError(
            f"PatentsView bulk snapshot is stale: {age_days} days old"
        )

    metadata = {
        "snapshot_date": snapshot_date.isoformat(),
        "snapshot_age_days": age_days,
        "snapshot_stale": is_stale,
        "max_age_days": max_age_days,
    }
    return [
        _record_from_bulk_row(row, fetched_at=fetched_at, snapshot_metadata=metadata)
        for row in rows
    ]


def deduplicate_patent_families(records: list[PatentRecord]) -> list[PatentRecord]:
    """Suppress duplicate family-level evidence, preferring grants over applications."""
    chosen: dict[str, PatentRecord] = {}
    for record in records:
        key = record.patent_family_id or record.application_id or record.patent_id
        current = chosen.get(key)
        if current is None or _dedupe_rank(record) > _dedupe_rank(current):
            chosen[key] = record
    return list(chosen.values())


def _page_params(query_text: str, *, page: int, page_size: int) -> dict[str, str]:
    return {
        "q": query_text,
        "offset": str(page * page_size),
        "limit": str(page_size),
    }


def _is_last_page(
    batch: list[PatentRecord],
    *,
    records_seen: int,
    page_size: int,
    total: int | None,
) -> bool:
    if not batch or len(batch) < page_size:
        return True
    return total is not None and records_seen >= total


def _dedupe_rank(record: PatentRecord) -> tuple[int, date, str]:
    event_date = record.event_date or date.min
    return (1 if record.grant_date else 0, event_date, record.patent_id or record.application_id)


def _build_odp_query(query: PatentQuery) -> str:
    parts: list[str] = []
    parts.extend(
        _field_terms("applicationMetaData.applicantNameBag.applicantNameText", query.assignees)
    )
    parts.extend(
        _field_terms(
            "applicationMetaData.cpcClassificationBag.cpcClassificationText",
            query.cpc_classes,
        )
    )
    parts.extend(
        _field_terms(
            "applicationMetaData.ipcClassificationBag.ipcClassificationText",
            query.ipc_classes,
        )
    )
    parts.extend(_keyword_terms(query.keywords))
    if query.start is not None or query.end is not None:
        start = query.start.isoformat() if query.start else "*"
        end = query.end.isoformat() if query.end else "*"
        parts.append(f"applicationMetaData.filingDate:[{start} TO {end}]")
    return " AND ".join(parts)


def _field_terms(field: str, values: list[str]) -> list[str]:
    terms = [f"{field}:{_prefix_token(value)}" for value in values if value.strip()]
    if len(terms) <= 1:
        return terms
    return [f"({' OR '.join(terms)})"]


def _keyword_terms(values: list[str]) -> list[str]:
    terms = [value.strip() for value in values if value.strip()]
    if len(terms) <= 1:
        return terms
    return [f"({' OR '.join(terms)})"]


def _prefix_token(value: str) -> str:
    match = re.search(r"[A-Za-z0-9]+", value)
    token = match.group(0) if match else value.strip()
    return f"{token}*"


def _response_json_dict(response: Any) -> dict[str, Any]:
    status_code = getattr(response, "status_code", 200)
    if status_code >= 400:
        raise PatentProviderResponseError(f"patent provider returned HTTP {status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise PatentProviderResponseError("patent provider returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise PatentProviderResponseError("patent provider returned non-object JSON")
    return payload


def _payload_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (
        payload.get("patentFileWrapperDataBag")
        or payload.get("results")
        or payload.get("patents")
        or []
    )
    if not isinstance(rows, list):
        raise PatentProviderResponseError("patent provider rows were not a list")
    return [row for row in rows if isinstance(row, dict)]


def _payload_total(payload: dict[str, Any]) -> int | None:
    raw_total = payload.get("count") or payload.get("total") or payload.get("numFound")
    try:
        return int(raw_total)
    except (TypeError, ValueError):
        return None


def _record_from_odp_row(
    row: dict[str, Any],
    *,
    fetched_at: datetime,
    source_url: str,
) -> PatentRecord:
    metadata = row.get("applicationMetaData")
    meta = metadata if isinstance(metadata, dict) else {}
    patent_id = _first_text(row, meta, "patentNumber", "patent_number", "patent_id")
    application_id = _first_text(
        row,
        meta,
        "applicationNumberText",
        "application_number",
        "application_id",
    )
    return PatentRecord(
        patent_id=patent_id,
        application_id=application_id,
        patent_family_id=(
            _first_text(row, meta, "familyId", "family_id") or application_id or patent_id
        ),
        title=_first_text(row, meta, "inventionTitle", "patent_title", "title"),
        abstract=_first_text(row, meta, "abstractText", "abstract"),
        assignees=_extract_name_bag(meta.get("applicantNameBag"), "applicantNameText"),
        cpc_classes=_extract_name_bag(meta.get("cpcClassificationBag"), "cpcClassificationText"),
        ipc_classes=_extract_name_bag(meta.get("ipcClassificationBag"), "ipcClassificationText"),
        application_date=_parse_date(_first_text(row, meta, "filingDate", "application_date")),
        grant_date=_parse_date(_first_text(row, meta, "grantDate", "patent_date", "grant_date")),
        source_url=source_url,
        source_attribution=PATENT_SOURCE_ODP,
        fetched_at=fetched_at,
        metadata={"raw_provider": "uspto_odp"},
    )


def _record_from_bulk_row(
    row: dict[str, Any],
    *,
    fetched_at: datetime,
    snapshot_metadata: dict[str, Any],
) -> PatentRecord:
    patent_id = _first_text(row, {}, "patent_number", "patent_id", "patentNumber")
    application_id = _first_text(row, {}, "application_number", "application_id")
    return PatentRecord(
        patent_id=patent_id,
        application_id=application_id,
        patent_family_id=(
            _first_text(row, {}, "family_id", "familyId") or application_id or patent_id
        ),
        title=_first_text(row, {}, "invention_title", "patent_title", "title"),
        abstract=_first_text(row, {}, "abstract", "abstract_text"),
        assignees=_list_from_field(row.get("assignee_organization") or row.get("assignees")),
        cpc_classes=_list_from_field(row.get("cpc_group_id") or row.get("cpc_classes")),
        ipc_classes=_list_from_field(row.get("ipc_class") or row.get("ipc_classes")),
        application_date=_parse_date(_first_text(row, {}, "filing_date", "application_date")),
        grant_date=_parse_date(_first_text(row, {}, "grant_date", "patent_date")),
        source_url=_first_text(row, {}, "source_url"),
        source_attribution=PATENT_SOURCE_BULK,
        fetched_at=fetched_at,
        metadata={"raw_provider": "patentsview_bulk", **snapshot_metadata},
    )


def _first_text(primary: dict[str, Any], secondary: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = primary.get(key)
        if value is None:
            value = secondary.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_name_bag(value: Any, key: str) -> list[str]:
    if isinstance(value, list):
        extracted = []
        for item in value:
            if isinstance(item, dict) and item.get(key):
                extracted.append(str(item[key]).strip())
            elif isinstance(item, str) and item.strip():
                extracted.append(item.strip())
        return extracted
    return _list_from_field(value)


def _list_from_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[;|]", text) if part.strip()]


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None
