"""SecApiProvider — fallback SEC filing provider using direct EDGAR APIs.

Uses raw HTTP requests to SEC EDGAR EFTS (full-text search) and filing
index/archives endpoints. Designed as a narrow fallback for when the
primary edgartools path fails or needs low-level validation.

Responsibilities:
    - Filing inventory retrieval (search by CIK/ticker/form type)
    - Filing text recovery (direct archive download)
    - Reconciliation support (same lineage output as EdgarToolsProvider)

Non-responsibilities:
    - Rich section parsing (that's edgartools' strength)
    - XBRL extraction (handled by the primary provider)
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any

import httpx

from src.filing.provider import FilingProvider
from src.filing.schemas import (
    FilingIdentity,
    FilingResult,
    FilingSection,
)
from src.filing.sec_policy import SECPolicy
from src.filing.utils import make_section_id, normalize_filing_type, parse_filing_date

logger = logging.getLogger(__name__)


class SecApiProvider(FilingProvider):
    """Fallback filing provider using direct SEC EDGAR HTTP APIs.

    Provides filing inventory and text recovery without edgartools.
    Output shapes are identical to EdgarToolsProvider for downstream
    compatibility.
    """

    def __init__(self, policy: SECPolicy | None = None) -> None:
        super().__init__(policy=policy)
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "sec_api"

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self._policy.headers,
                timeout=self._policy.request_timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def fetch_filing(self, accession_number: str) -> FilingResult:
        """Fetch a filing by accession number from SEC archives.

        Downloads the filing's index page to extract metadata, then
        fetches the primary document text.
        """
        try:
            # Normalize accession number format
            acc_clean = accession_number.replace("-", "")

            # Fetch filing index to get metadata
            client = await self._get_client()
            index_url = (
                f"{self._policy.filing_base_url}"
                f"?action=getcompany&accession={accession_number}"
                f"&type=&dateb=&owner=include&count=1&output=atom"
            )
            await self._acquire_rate_limit()
            index_resp = await client.get(index_url)

            if index_resp.status_code != 200:
                return self._failed_result(
                    accession_number,
                    f"Index fetch failed: HTTP {index_resp.status_code}",
                )

            # EDGAR archive paths use integer CIK (no leading zeros).
            # The first 10 digits of the accession number are the CIK.
            cik_raw = acc_clean[:10]
            cik_dir = cik_raw.lstrip("0") or "0"
            text_url = (
                f"{self._policy.archives_url}/"
                f"{cik_dir}/{acc_clean}/{accession_number}.txt"
            )
            await self._acquire_rate_limit()
            text_resp = await client.get(text_url)

            if text_resp.status_code == 200:
                content = text_resp.text
                truncated = content[:500_000]
                word_count = len(truncated.split())
                sections = [
                    FilingSection(
                        section_id=make_section_id(accession_number, 0, "full_text"),
                        section_name="Full Text",
                        section_type="narrative",
                        content=truncated,
                        word_count=word_count,
                    )
                ]
                status = "parsed"
            else:
                sections = []
                status = "fetched"

            return FilingResult(
                identity=FilingIdentity(
                    cik=cik_dir,
                    accession_number=accession_number,
                    filing_type="8-K",  # Unknown from this path
                    filed_date=date.today(),
                ),
                sections=sections,
                raw_url=text_url,
                status=status,
                provider=self.name,
                fetched_at=datetime.now(UTC),
            )

        except Exception as e:
            logger.error(
                "SecApiProvider.fetch_filing failed: %s: %s",
                accession_number,
                e,
            )
            return self._failed_result(accession_number, str(e))

    async def search_filings(
        self,
        *,
        cik: str | None = None,
        ticker: str | None = None,
        filing_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 20,
    ) -> list[FilingIdentity]:
        """Search for filings using SEC EDGAR EFTS full-text search API."""
        if not cik and not ticker:
            raise ValueError("At least one of cik or ticker must be provided")

        try:
            client = await self._get_client()

            # Build EFTS search query
            params: dict[str, Any] = {
                "q": "",
                "dateRange": "custom",
                "startdt": date_from or "2020-01-01",
                "enddt": date_to or date.today().isoformat(),
            }
            if cik:
                params["q"] = f'cik:"{cik}"'
            elif ticker:
                params["q"] = f'ticker:"{ticker}"'
            if filing_type:
                params["forms"] = filing_type

            url = f"{self._policy.base_url}/search-index"
            await self._acquire_rate_limit()
            resp = await client.get(url, params=params)

            if resp.status_code != 200:
                logger.warning(
                    "EFTS search failed: HTTP %d", resp.status_code
                )
                return []

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            results: list[FilingIdentity] = []
            for hit in hits[:limit]:
                source = hit.get("_source", {})
                form = source.get("form_type", source.get("file_type", ""))
                normalized = normalize_filing_type(form)
                filed = parse_filing_date(
                    source.get("file_date", source.get("date_filed"))
                )
                period = parse_filing_date(source.get("period_of_report"))

                results.append(
                    FilingIdentity(
                        cik=str(source.get("entity_id", cik or "")),
                        accession_number=str(
                            source.get("accession_no", source.get("accession_number", ""))
                        ),
                        filing_type=normalized,
                        filed_date=filed,
                        period_of_report=period if source.get("period_of_report") else None,
                        company_name=str(source.get("entity_name", "")),
                        ticker=ticker,
                    )
                )

            return results

        except Exception as e:
            logger.error("SecApiProvider.search_filings failed: %s", e)
            return []

    def _failed_result(
        self, accession_number: str, error_message: str
    ) -> FilingResult:
        """Build a failed FilingResult preserving lineage fields."""
        return FilingResult(
            identity=FilingIdentity(
                cik="",
                accession_number=accession_number,
                filing_type="8-K",
                filed_date=date.today(),
            ),
            status="failed",
            error_message=error_message,
            provider=self.name,
            fetched_at=datetime.now(UTC),
        )
