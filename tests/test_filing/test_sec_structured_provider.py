"""Tests for SEC structured submissions and Company Facts provider."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.filing.sec_policy import SECPolicy
from src.filing.sec_structured import (
    SECStructuredDataError,
    SECStructuredDataProvider,
    SECStructuredPayloadRecord,
)

SUBMISSIONS_PAYLOAD: dict[str, Any] = {
    "cik": "0000320193",
    "name": "Apple Inc.",
    "tickers": ["AAPL"],
    "exchanges": ["Nasdaq"],
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-24-000123",
                "0000320193-24-000111",
            ],
            "form": ["10-K", "10-Q"],
            "filingDate": ["2024-11-01", "2024-08-02"],
        }
    },
}

COMPANY_FACTS_PAYLOAD: dict[str, Any] = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        {
                            "accn": "0000320193-24-000123",
                            "fy": 2024,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2024-11-01",
                        },
                        {
                            "accn": "0000320193-24-000111",
                            "fy": 2024,
                            "fp": "Q3",
                            "form": "10-Q",
                            "filed": "2024-08-02",
                        },
                    ]
                },
            }
        }
    },
}


class _Response:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _HTTPClient:
    def __init__(self, *responses: _Response | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.is_closed = False

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> _Response:
        self.requests.append({"url": url, "headers": headers})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self) -> None:
        self.is_closed = True


class _RateLimiter:
    def __init__(self) -> None:
        self.acquire_count = 0

    async def acquire(self) -> None:
        self.acquire_count += 1


class _Repository:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], SECStructuredPayloadRecord] = {}
        self.upserts: list[SECStructuredPayloadRecord] = []

    async def get_latest_payload(
        self,
        cik: str,
        payload_type: str,
    ) -> SECStructuredPayloadRecord | None:
        return self.records.get((cik, payload_type))

    async def upsert_payload(
        self,
        record: SECStructuredPayloadRecord,
    ) -> SECStructuredPayloadRecord:
        self.upserts.append(record)
        self.records[(record.cik, record.payload_type)] = record
        return record


@pytest.mark.asyncio
async def test_submissions_fetch_uses_sec_policy_and_caches_accession_lineage() -> None:
    policy = SECPolicy(
        user_agent="News Tracker tests contact@example.com",
        data_api_url="https://data.sec.gov",
    )
    client = _HTTPClient(_Response(200, SUBMISSIONS_PAYLOAD))
    limiter = _RateLimiter()
    repository = _Repository()
    provider = SECStructuredDataProvider(
        repository=repository,
        policy=policy,
        http_client=client,
        rate_limiter=limiter,
    )

    record = await provider.fetch_submissions("320193")

    assert client.requests == [
        {
            "url": "https://data.sec.gov/submissions/CIK0000320193.json",
            "headers": policy.headers,
        }
    ]
    assert limiter.acquire_count == 1
    assert record.cik == "0000320193"
    assert record.payload_type == "submissions"
    assert record.accession_numbers == [
        "0000320193-24-000111",
        "0000320193-24-000123",
    ]
    assert record.payload_hash.startswith("sha256:")
    assert repository.upserts == [record]


@pytest.mark.asyncio
async def test_companyfacts_fetch_extracts_accession_lineage() -> None:
    client = _HTTPClient(_Response(200, COMPANY_FACTS_PAYLOAD))
    repository = _Repository()
    provider = SECStructuredDataProvider(
        repository=repository,
        policy=SECPolicy(data_api_url="https://data.sec.gov"),
        http_client=client,
        rate_limiter=_RateLimiter(),
    )

    record = await provider.fetch_company_facts("0000320193")

    assert client.requests[0]["url"] == (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    )
    assert record.payload_type == "companyfacts"
    assert record.accession_numbers == [
        "0000320193-24-000111",
        "0000320193-24-000123",
    ]


@pytest.mark.asyncio
async def test_cache_hit_skips_sec_request() -> None:
    cached = SECStructuredPayloadRecord(
        cik="0000320193",
        payload_type="submissions",
        source_url="https://data.sec.gov/submissions/CIK0000320193.json",
        payload_hash="sha256:cached",
        payload=SUBMISSIONS_PAYLOAD,
        accession_numbers=["0000320193-24-000123"],
        fetched_at=datetime(2026, 5, 31, tzinfo=UTC),
    )
    repository = _Repository()
    repository.records[(cached.cik, cached.payload_type)] = cached
    client = _HTTPClient()
    limiter = _RateLimiter()
    provider = SECStructuredDataProvider(
        repository=repository,
        http_client=client,
        rate_limiter=limiter,
    )

    record = await provider.fetch_submissions("0000320193")

    assert record is cached
    assert client.requests == []
    assert limiter.acquire_count == 0


@pytest.mark.asyncio
async def test_transient_sec_error_retries_before_cache_write() -> None:
    client = _HTTPClient(
        _Response(503, {"message": "temporarily unavailable"}),
        _Response(200, SUBMISSIONS_PAYLOAD),
    )
    limiter = _RateLimiter()
    repository = _Repository()
    provider = SECStructuredDataProvider(
        repository=repository,
        policy=SECPolicy(max_retries=1, retry_base_delay=0.0),
        http_client=client,
        rate_limiter=limiter,
    )

    record = await provider.fetch_submissions("0000320193")

    assert record.payload_type == "submissions"
    assert len(client.requests) == 2
    assert limiter.acquire_count == 2
    assert len(repository.upserts) == 1


@pytest.mark.asyncio
async def test_malformed_sec_payload_raises_without_cache_write() -> None:
    client = _HTTPClient(_Response(200, {"cik": "0000320193", "entityName": "Apple Inc."}))
    repository = _Repository()
    provider = SECStructuredDataProvider(
        repository=repository,
        http_client=client,
        rate_limiter=_RateLimiter(),
    )

    with pytest.raises(SECStructuredDataError, match="company facts"):
        await provider.fetch_company_facts("0000320193")

    assert repository.upserts == []


@pytest.mark.asyncio
async def test_close_does_not_close_injected_http_client() -> None:
    client = _HTTPClient()
    provider = SECStructuredDataProvider(repository=_Repository(), http_client=client)

    await provider.close()

    assert client.is_closed is False


def test_provider_exposes_official_bulk_archive_urls_for_backfills() -> None:
    policy = SECPolicy(
        companyfacts_bulk_url="https://www.sec.gov/companyfacts.zip",
        submissions_bulk_url="https://www.sec.gov/submissions.zip",
    )
    provider = SECStructuredDataProvider(repository=_Repository(), policy=policy)

    assert provider.bulk_archive_urls() == {
        "companyfacts": "https://www.sec.gov/companyfacts.zip",
        "submissions": "https://www.sec.gov/submissions.zip",
    }
