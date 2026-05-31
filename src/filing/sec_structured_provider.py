"""Cache-first SEC structured submissions and Company Facts provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

from src.filing.provider import SECRateLimiter
from src.filing.sec_policy import SECPolicy
from src.filing.sec_structured_models import (
    SECStructuredDataError,
    SECStructuredPayloadRecord,
)
from src.security_master.schemas import normalize_sec_cik


class SECStructuredHTTPClient(Protocol):
    @property
    def is_closed(self) -> bool:
        """Whether the client has already been closed."""
        ...

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Fetch a SEC JSON URL."""
        ...

    async def aclose(self) -> None:
        """Close the client."""
        ...


class SECStructuredRepository(Protocol):
    async def get_latest_payload(
        self,
        cik: str,
        payload_type: str,
    ) -> SECStructuredPayloadRecord | None:
        """Return the latest cached payload for a CIK and type."""
        ...

    async def upsert_payload(
        self,
        record: SECStructuredPayloadRecord,
    ) -> SECStructuredPayloadRecord:
        """Persist a structured SEC payload idempotently."""
        ...


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _normalize_payload_cik(payload: dict[str, Any]) -> str | None:
    raw_cik = payload.get("cik")
    return normalize_sec_cik(raw_cik) if raw_cik is not None else None


def _validate_cik_matches(payload: dict[str, Any], expected_cik: str, label: str) -> None:
    payload_cik = _normalize_payload_cik(payload)
    if payload_cik != expected_cik:
        raise SECStructuredDataError(f"SEC {label} payload CIK does not match requested issuer")


def _extract_submissions_accessions(payload: dict[str, Any]) -> list[str]:
    recent = payload.get("filings", {}).get("recent", {})
    values = recent.get("accessionNumber", [])
    if not isinstance(values, list):
        return []
    return sorted({str(value) for value in values if value})


def _extract_companyfacts_accessions(payload: dict[str, Any]) -> list[str]:
    accessions: set[str] = set()
    facts = payload.get("facts", {})
    if not isinstance(facts, dict):
        return []

    for taxonomy in facts.values():
        if not isinstance(taxonomy, dict):
            continue
        for concept in taxonomy.values():
            if not isinstance(concept, dict):
                continue
            units = concept.get("units", {})
            if not isinstance(units, dict):
                continue
            for observations in units.values():
                if not isinstance(observations, list):
                    continue
                for observation in observations:
                    if isinstance(observation, dict) and observation.get("accn"):
                        accessions.add(str(observation["accn"]))
    return sorted(accessions)


class SECStructuredDataProvider:
    """Cache-first provider for SEC submissions and Company Facts JSON."""

    def __init__(
        self,
        *,
        repository: SECStructuredRepository,
        policy: SECPolicy | None = None,
        http_client: SECStructuredHTTPClient | None = None,
        rate_limiter: SECRateLimiter | None = None,
    ) -> None:
        self._repository = repository
        self._policy = policy or SECPolicy()
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._rate_limiter = rate_limiter or SECRateLimiter(self._policy.rate_limit_per_second)

    async def fetch_submissions(
        self,
        cik: str,
        *,
        use_cache: bool = True,
    ) -> SECStructuredPayloadRecord:
        """Fetch or return cached SEC submissions metadata for an issuer."""
        return await self._fetch_payload(
            cik,
            payload_type="submissions",
            source_url=self._submissions_url(cik),
            use_cache=use_cache,
        )

    async def fetch_company_facts(
        self,
        cik: str,
        *,
        use_cache: bool = True,
    ) -> SECStructuredPayloadRecord:
        """Fetch or return cached SEC Company Facts for an issuer."""
        return await self._fetch_payload(
            cik,
            payload_type="companyfacts",
            source_url=self._companyfacts_url(cik),
            use_cache=use_cache,
        )

    def bulk_archive_urls(self) -> dict[str, str]:
        """Return official nightly SEC bulk archives for backfill jobs."""
        return {
            "companyfacts": self._policy.companyfacts_bulk_url,
            "submissions": self._policy.submissions_bulk_url,
        }

    async def close(self) -> None:
        """Close the owned HTTP client if one was opened lazily."""
        if (
            self._owns_http_client
            and self._http_client is not None
            and not self._http_client.is_closed
        ):
            await self._http_client.aclose()

    async def _fetch_payload(
        self,
        cik: str,
        *,
        payload_type: str,
        source_url: str,
        use_cache: bool,
    ) -> SECStructuredPayloadRecord:
        normalized_cik = self._normalize_required_cik(cik)
        if use_cache:
            cached = await self._repository.get_latest_payload(normalized_cik, payload_type)
            if cached is not None:
                return cached

        payload = await self._request_json(source_url)
        self._validate_payload(payload, normalized_cik, payload_type)
        record = SECStructuredPayloadRecord(
            cik=normalized_cik,
            payload_type=payload_type,
            source_url=source_url,
            payload_hash=_payload_hash(payload),
            payload=payload,
            accession_numbers=self._extract_accessions(payload, payload_type),
            fetched_at=datetime.now(UTC),
            metadata={"source": "sec", "cache_policy": "cache_first"},
        )
        return await self._repository.upsert_payload(record)

    async def _request_json(self, url: str) -> dict[str, Any]:
        client = await self._get_client()
        last_error: Exception | None = None
        for attempt in range(self._policy.max_retries + 1):
            await self._rate_limiter.acquire()
            try:
                response = await client.get(url, headers=self._policy.headers)
            except Exception as exc:
                last_error = exc
                if attempt < self._policy.max_retries:
                    await self._sleep_before_retry(attempt)
                    continue
                raise SECStructuredDataError(f"SEC request failed for {url}: {exc}") from exc

            if response.status_code == 200:
                return self._decode_json_response(response, url)
            if not self._should_retry_status(response.status_code):
                raise SECStructuredDataError(
                    f"SEC request failed for {url}: HTTP {response.status_code}"
                )
            if attempt < self._policy.max_retries:
                await self._sleep_before_retry(attempt)
                continue
            last_error = SECStructuredDataError(
                f"SEC request failed for {url}: HTTP {response.status_code}"
            )
        raise SECStructuredDataError(f"SEC request failed for {url}: {last_error}")

    async def _get_client(self) -> SECStructuredHTTPClient:
        client = self._http_client
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                headers=self._policy.headers,
                timeout=self._policy.request_timeout,
                follow_redirects=True,
            )
            self._http_client = client
            self._owns_http_client = True
        return client

    async def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(
            self._policy.retry_max_delay,
            self._policy.retry_base_delay * (2**attempt),
        )
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _decode_json_response(response: httpx.Response, url: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise SECStructuredDataError(f"SEC response was not valid JSON for {url}") from exc
        if not isinstance(payload, dict):
            raise SECStructuredDataError(f"SEC response was not a JSON object for {url}")
        return payload

    @staticmethod
    def _should_retry_status(status_code: int) -> bool:
        return status_code == 429 or status_code >= 500

    @staticmethod
    def _normalize_required_cik(cik: str) -> str:
        normalized = normalize_sec_cik(cik)
        if normalized is None:
            raise ValueError("cik must be a non-empty SEC CIK")
        return normalized

    def _submissions_url(self, cik: str) -> str:
        normalized = self._normalize_required_cik(cik)
        return f"{self._policy.data_api_url}/submissions/CIK{normalized}.json"

    def _companyfacts_url(self, cik: str) -> str:
        normalized = self._normalize_required_cik(cik)
        return f"{self._policy.data_api_url}/api/xbrl/companyfacts/CIK{normalized}.json"

    @staticmethod
    def _validate_payload(payload: dict[str, Any], cik: str, payload_type: str) -> None:
        if payload_type == "submissions":
            _validate_cik_matches(payload, cik, "submissions")
            filings = payload.get("filings")
            if not isinstance(filings, dict) or not isinstance(filings.get("recent"), dict):
                raise SECStructuredDataError("SEC submissions payload missing filings.recent")
            return

        _validate_cik_matches(payload, cik, "company facts")
        if not isinstance(payload.get("facts"), dict):
            raise SECStructuredDataError("SEC company facts payload missing facts")

    @staticmethod
    def _extract_accessions(payload: dict[str, Any], payload_type: str) -> list[str]:
        if payload_type == "submissions":
            return _extract_submissions_accessions(payload)
        return _extract_companyfacts_accessions(payload)
