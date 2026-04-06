"""EdgarToolsProvider — primary SEC filing ingestion via edgartools.

Wraps the edgartools library behind the FilingProvider interface,
translating its rich typed objects into our normalized FilingResult
output shape.

edgartools is imported lazily so the filing module stays importable
without the dependency installed. Synchronous edgartools calls are
run in a thread executor to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime
from functools import partial
from typing import Any

from src.filing.provider import FilingProvider
from src.filing.schemas import (
    FilingIdentity,
    FilingResult,
    FilingSection,
)
from src.filing.utils import make_section_id, normalize_filing_type, parse_filing_date

logger = logging.getLogger(__name__)

# Well-known 10-K/10-Q section names for structured extraction
_10K_SECTIONS = [
    "Business",
    "Risk Factors",
    "Properties",
    "Legal Proceedings",
    "Management's Discussion and Analysis",
    "Financial Statements",
    "Controls and Procedures",
]


def _ensure_edgartools() -> Any:
    """Import edgartools lazily, raising a clear error if missing."""
    try:
        import edgar

        return edgar
    except ImportError as e:
        raise ImportError(
            "edgartools is required for EdgarToolsProvider. Install it with: pip install edgartools"
        ) from e


def _extract_sections(filing: Any, accession: str) -> list[FilingSection]:
    """Extract sections from an edgartools Filing object.

    Tries filing.sections first (available on parsed 10-K/10-Q),
    then falls back to full text as a single section.
    """
    sections: list[FilingSection] = []

    try:
        # edgartools Filing.sections returns parsed document sections
        filing_sections = filing.sections
        if filing_sections:
            for i, section in enumerate(filing_sections):
                name = getattr(section, "title", None) or getattr(
                    section, "name", f"Section {i + 1}"
                )
                content = str(section) if section else ""
                word_count = len(content.split()) if content else 0
                sections.append(
                    FilingSection(
                        section_id=make_section_id(accession, i, str(name)),
                        section_name=str(name),
                        section_type="narrative",
                        content=content,
                        word_count=word_count,
                    )
                )
            return sections
    except Exception:
        pass

    # Fallback: get full text as a single section
    try:
        text = filing.text() if callable(getattr(filing, "text", None)) else ""
        if text:
            word_count = len(text.split())
            sections.append(
                FilingSection(
                    section_id=make_section_id(accession, 0, "full_text"),
                    section_name="Full Text",
                    section_type="narrative",
                    content=text,
                    word_count=word_count,
                )
            )
    except Exception as e:
        logger.debug("Failed to extract text from filing %s: %s", accession, e)

    return sections


def _filing_to_result(filing: Any, provider_name: str) -> FilingResult:
    """Convert an edgartools Filing to our FilingResult schema."""
    header = filing.header if hasattr(filing, "header") else None

    # Extract identity fields
    cik = str(getattr(header, "cik", "") or getattr(filing, "cik", "") or "")
    accession = str(filing.accession_number) if filing.accession_number else ""
    form_type = str(getattr(header, "form_type", "") or "")
    company_name = str(getattr(header, "company_name", "") or getattr(header, "company", "") or "")

    # Parse dates
    filed_date = parse_filing_date(
        getattr(header, "filed", None) or getattr(header, "filing_date", None)
    )
    period = parse_filing_date(getattr(filing, "period_of_report", None))

    # Normalize filing type to match our VALID_FILING_TYPES
    normalized_type = normalize_filing_type(form_type)

    # Build identity
    identity = FilingIdentity(
        cik=cik,
        accession_number=accession,
        filing_type=normalized_type,
        filed_date=filed_date,
        period_of_report=period,
        company_name=company_name,
    )

    # Extract sections
    sections = _extract_sections(filing, accession)

    # Build URL
    raw_url = str(getattr(filing, "filing_url", "") or getattr(filing, "url", "") or "")

    return FilingResult(
        identity=identity,
        sections=sections,
        raw_url=raw_url,
        status="parsed" if sections else "fetched",
        provider=provider_name,
        fetched_at=datetime.now(UTC),
    )


class EdgarToolsProvider(FilingProvider):
    """Primary filing provider using edgartools.

    Provides the richest access to SEC filings with typed objects,
    section parsing, and XBRL support.
    """

    @property
    def name(self) -> str:
        return "edgartools"

    def _setup_identity(self) -> None:
        """Set the SEC identity for edgartools requests."""
        edgar = _ensure_edgartools()
        edgar.set_identity(self._policy.user_agent)

    async def fetch_filing(self, accession_number: str) -> FilingResult:
        """Fetch and parse a filing by accession number."""
        await self._acquire_rate_limit()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(self._fetch_filing_sync, accession_number),
            )
            return result
        except Exception as e:
            logger.error(
                "EdgarToolsProvider.fetch_filing failed: %s: %s",
                accession_number,
                e,
            )
            return FilingResult(
                identity=FilingIdentity(
                    cik="",
                    accession_number=accession_number,
                    filing_type="8-K",
                    filed_date=date.today(),
                ),
                status="failed",
                error_message=str(e),
                provider=self.name,
                fetched_at=datetime.now(UTC),
            )

    def _fetch_filing_sync(self, accession_number: str) -> FilingResult:
        """Synchronous fetch — runs in thread executor."""
        edgar = _ensure_edgartools()
        self._setup_identity()

        # edgartools can find filings by accession number
        filing = edgar.find(accession_number)
        if filing is None:
            return FilingResult(
                identity=FilingIdentity(
                    cik="",
                    accession_number=accession_number,
                    filing_type="8-K",
                    filed_date=date.today(),
                ),
                status="failed",
                error_message=f"Filing not found: {accession_number}",
                provider=self.name,
                fetched_at=datetime.now(UTC),
            )

        return _filing_to_result(filing, self.name)

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
        """Search for filings using edgartools Company API."""
        if not cik and not ticker:
            raise ValueError("At least one of cik or ticker must be provided")

        await self._acquire_rate_limit()
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(
                    self._search_filings_sync,
                    cik=cik,
                    ticker=ticker,
                    filing_type=filing_type,
                    date_from=date_from,
                    date_to=date_to,
                    limit=limit,
                ),
            )
            return results
        except Exception as e:
            logger.error("EdgarToolsProvider.search_filings failed: %s", e)
            return []

    def _search_filings_sync(
        self,
        *,
        cik: str | None,
        ticker: str | None,
        filing_type: str | None,
        date_from: str | None,
        date_to: str | None,
        limit: int,
    ) -> list[FilingIdentity]:
        """Synchronous search — runs in thread executor."""
        edgar = _ensure_edgartools()
        self._setup_identity()

        # Resolve to Company
        lookup = cik or ticker
        company = edgar.Company(lookup)

        # Get filings with optional form filter
        filings = company.get_filings(form=filing_type) if filing_type else company.get_filings()

        # Apply date filtering and limit
        results: list[FilingIdentity] = []
        for filing in filings:
            if len(results) >= limit:
                break

            header = filing.header if hasattr(filing, "header") else None
            filed = parse_filing_date(
                getattr(header, "filed", None) or getattr(header, "filing_date", None)
            )

            # Date range filtering
            if date_from and filed < date.fromisoformat(date_from):
                continue
            if date_to and filed > date.fromisoformat(date_to):
                continue

            form_type = str(getattr(header, "form_type", "") or "")
            normalized = normalize_filing_type(form_type)
            accession = str(filing.accession_number) if filing.accession_number else ""
            company_name = str(
                getattr(header, "company_name", "") or getattr(header, "company", "") or ""
            )

            results.append(
                FilingIdentity(
                    cik=str(getattr(company, "cik", cik or "")),
                    accession_number=accession,
                    filing_type=normalized,
                    filed_date=filed,
                    company_name=company_name,
                    ticker=ticker,
                )
            )

        return results
