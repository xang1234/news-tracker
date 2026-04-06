"""Persistence schemas and repository for filing artifacts.

Maps 1:1 to the tables in migration 022. Provides CRUD operations
for filings, sections, attachments, and XBRL facts with full lineage.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from src.filing.schemas import VALID_FILING_STATUSES, FilingResult
from src.storage.database import Database

logger = logging.getLogger(__name__)


# -- Dataclasses mapping to DB tables --------------------------------------


@dataclass
class FilingRecord:
    """Persisted filing metadata (filings table).

    Attributes:
        accession_number: SEC accession number (primary key).
        cik: SEC Central Index Key.
        filing_type: SEC form type.
        filed_date: When filed with the SEC.
        period_of_report: Reporting period end date.
        company_name: Filer name.
        ticker: Primary ticker symbol.
        concept_id: Canonical concept ID (if resolved).
        raw_url: URL to the filing on EDGAR.
        content_hash: SHA-256 hash of the full filing text.
        total_word_count: Total words across all sections.
        section_count: Number of parsed sections.
        provider: Which provider produced this record.
        run_id: Lane run that ingested this filing.
        status: Processing status.
        error_message: Error details if failed.
        source_published_at: When the SEC published the filing.
        ingested_at: When we fetched the filing.
        metadata: Extensible metadata.
    """

    accession_number: str
    cik: str
    filing_type: str
    filed_date: date
    period_of_report: date | None = None
    company_name: str = ""
    ticker: str | None = None
    concept_id: str | None = None
    raw_url: str = ""
    content_hash: str | None = None
    total_word_count: int = 0
    section_count: int = 0
    provider: str = ""
    run_id: str | None = None
    status: str = "fetched"
    error_message: str | None = None
    source_published_at: datetime | None = None
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_FILING_STATUSES:
            raise ValueError(
                f"Invalid filing status {self.status!r}. "
                f"Must be one of {sorted(VALID_FILING_STATUSES)}"
            )


@dataclass
class FilingSectionRecord:
    """Persisted filing section (filing_sections table)."""

    section_id: str
    accession_number: str
    section_index: int
    section_name: str
    section_type: str = "narrative"
    content: str = ""
    word_count: int = 0
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class FilingAttachmentRecord:
    """Persisted filing attachment (filing_attachments table)."""

    attachment_id: str
    accession_number: str
    filename: str
    content_type: str = ""
    description: str = ""
    url: str = ""
    size_bytes: int | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class XBRLFactRecord:
    """Persisted XBRL data point (filing_xbrl_facts table)."""

    accession_number: str
    concept_name: str
    value: str
    taxonomy: str = "us-gaap"
    unit: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    instant_date: date | None = None
    decimals: int | None = None
    segment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    created_at: datetime | None = None


# -- Conversion from FilingResult to persistence records -------------------


def filing_result_to_records(
    result: FilingResult,
    *,
    run_id: str | None = None,
) -> tuple[FilingRecord, list[FilingSectionRecord]]:
    """Convert a FilingResult from a provider into persistence records.

    Args:
        result: Provider output.
        run_id: Lane run that produced this filing.

    Returns:
        Tuple of (FilingRecord, list of FilingSectionRecords).
    """
    # Compute content hash over all section content
    all_content = "\n".join(s.content for s in result.sections)
    content_hash = (
        f"sha256:{hashlib.sha256(all_content.encode()).hexdigest()}" if all_content else None
    )

    filing = FilingRecord(
        accession_number=result.identity.accession_number,
        cik=result.identity.cik,
        filing_type=result.identity.filing_type,
        filed_date=result.identity.filed_date,
        period_of_report=result.identity.period_of_report,
        company_name=result.identity.company_name,
        ticker=result.identity.ticker,
        raw_url=result.raw_url,
        content_hash=content_hash,
        total_word_count=result.total_word_count,
        section_count=len(result.sections),
        provider=result.provider,
        run_id=run_id,
        status=result.status,
        error_message=result.error_message,
        ingested_at=result.fetched_at or datetime.now(UTC),
        metadata=result.metadata,
    )

    sections = []
    for i, s in enumerate(result.sections):
        section_hash = (
            f"sha256:{hashlib.sha256(s.content.encode()).hexdigest()}" if s.content else None
        )
        sections.append(
            FilingSectionRecord(
                section_id=s.section_id,
                accession_number=result.identity.accession_number,
                section_index=i,
                section_name=s.section_name,
                section_type=s.section_type,
                content=s.content,
                word_count=s.word_count,
                content_hash=section_hash,
                metadata=s.metadata,
            )
        )

    return filing, sections


# -- Row converters --------------------------------------------------------


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value)


def _row_to_filing(row: Any) -> FilingRecord:
    return FilingRecord(
        accession_number=row["accession_number"],
        cik=row["cik"],
        filing_type=row["filing_type"],
        filed_date=row["filed_date"],
        period_of_report=row["period_of_report"],
        company_name=row["company_name"],
        ticker=row["ticker"],
        concept_id=row["concept_id"],
        raw_url=row["raw_url"],
        content_hash=row["content_hash"],
        total_word_count=row["total_word_count"],
        section_count=row["section_count"],
        provider=row["provider"],
        run_id=row["run_id"],
        status=row["status"],
        error_message=row["error_message"],
        source_published_at=row["source_published_at"],
        ingested_at=row["ingested_at"],
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_section(row: Any) -> FilingSectionRecord:
    return FilingSectionRecord(
        section_id=row["section_id"],
        accession_number=row["accession_number"],
        section_index=row["section_index"],
        section_name=row["section_name"],
        section_type=row["section_type"],
        content=row["content"],
        word_count=row["word_count"],
        content_hash=row["content_hash"],
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
    )


def _row_to_xbrl_fact(row: Any) -> XBRLFactRecord:
    return XBRLFactRecord(
        accession_number=row["accession_number"],
        concept_name=row["concept_name"],
        value=row["value"],
        taxonomy=row["taxonomy"],
        unit=row["unit"],
        period_start=row["period_start"],
        period_end=row["period_end"],
        instant_date=row["instant_date"],
        decimals=row["decimals"],
        segment=row["segment"],
        metadata=_parse_json(row["metadata"]),
        id=row["id"],
        created_at=row["created_at"],
    )


def _row_to_attachment(row: Any) -> FilingAttachmentRecord:
    return FilingAttachmentRecord(
        attachment_id=row["attachment_id"],
        accession_number=row["accession_number"],
        filename=row["filename"],
        content_type=row["content_type"],
        description=row["description"],
        url=row["url"],
        size_bytes=row["size_bytes"],
        content_hash=row["content_hash"],
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
    )


# -- Repository ------------------------------------------------------------


class FilingRepository:
    """CRUD operations for filing artifacts."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # -- Filings -----------------------------------------------------------

    async def upsert_filing(self, filing: FilingRecord) -> FilingRecord:
        """Insert or update a filing record."""
        row = await self._db.fetchrow(
            """
            INSERT INTO filings (
                accession_number, cik, filing_type, filed_date,
                period_of_report, company_name, ticker, concept_id,
                raw_url, content_hash, total_word_count, section_count,
                provider, run_id, status, error_message,
                source_published_at, ingested_at, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19
            )
            ON CONFLICT (accession_number) DO UPDATE SET
                status = $15,
                error_message = $16,
                content_hash = $10,
                total_word_count = $11,
                section_count = $12,
                metadata = $19
            RETURNING *
            """,
            filing.accession_number,
            filing.cik,
            filing.filing_type,
            filing.filed_date,
            filing.period_of_report,
            filing.company_name,
            filing.ticker,
            filing.concept_id,
            filing.raw_url,
            filing.content_hash,
            filing.total_word_count,
            filing.section_count,
            filing.provider,
            filing.run_id,
            filing.status,
            filing.error_message,
            filing.source_published_at,
            filing.ingested_at,
            json.dumps(filing.metadata),
        )
        return _row_to_filing(row)

    async def get_filing(self, accession_number: str) -> FilingRecord | None:
        """Fetch a filing by accession number."""
        row = await self._db.fetchrow(
            "SELECT * FROM filings WHERE accession_number = $1",
            accession_number,
        )
        return _row_to_filing(row) if row else None

    async def list_filings(
        self,
        *,
        cik: str | None = None,
        ticker: str | None = None,
        filing_type: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[FilingRecord]:
        """List filings with optional filters."""
        conditions = []
        params: list[Any] = []
        if cik is not None:
            params.append(cik)
            conditions.append(f"cik = ${len(params)}")
        if ticker is not None:
            params.append(ticker)
            conditions.append(f"ticker = ${len(params)}")
        if filing_type is not None:
            params.append(filing_type)
            conditions.append(f"filing_type = ${len(params)}")
        if status is not None:
            params.append(status)
            conditions.append(f"status = ${len(params)}")
        params.append(limit)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT * FROM filings
            {where}
            ORDER BY filed_date DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_filing(row) for row in rows]

    # -- Sections ----------------------------------------------------------

    async def upsert_section(self, section: FilingSectionRecord) -> FilingSectionRecord:
        """Insert or update a filing section."""
        row = await self._db.fetchrow(
            """
            INSERT INTO filing_sections (
                section_id, accession_number, section_index,
                section_name, section_type, content, word_count,
                content_hash, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (section_id) DO UPDATE SET
                content = $6,
                word_count = $7,
                content_hash = $8,
                metadata = $9
            RETURNING *
            """,
            section.section_id,
            section.accession_number,
            section.section_index,
            section.section_name,
            section.section_type,
            section.content,
            section.word_count,
            section.content_hash,
            json.dumps(section.metadata),
        )
        return _row_to_section(row)

    async def get_sections(self, accession_number: str) -> list[FilingSectionRecord]:
        """Get all sections for a filing, ordered by index."""
        rows = await self._db.fetch(
            """
            SELECT * FROM filing_sections
            WHERE accession_number = $1
            ORDER BY section_index
            """,
            accession_number,
        )
        return [_row_to_section(row) for row in rows]

    # -- Attachments -------------------------------------------------------

    async def upsert_attachment(self, attachment: FilingAttachmentRecord) -> FilingAttachmentRecord:
        """Insert or update a filing attachment."""
        row = await self._db.fetchrow(
            """
            INSERT INTO filing_attachments (
                attachment_id, accession_number, filename,
                content_type, description, url,
                size_bytes, content_hash, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (attachment_id) DO UPDATE SET
                content_type = $4,
                description = $5,
                url = $6,
                size_bytes = $7,
                content_hash = $8,
                metadata = $9
            RETURNING *
            """,
            attachment.attachment_id,
            attachment.accession_number,
            attachment.filename,
            attachment.content_type,
            attachment.description,
            attachment.url,
            attachment.size_bytes,
            attachment.content_hash,
            json.dumps(attachment.metadata),
        )
        return _row_to_attachment(row)

    async def get_attachments(self, accession_number: str) -> list[FilingAttachmentRecord]:
        """Get all attachments for a filing."""
        rows = await self._db.fetch(
            """
            SELECT * FROM filing_attachments
            WHERE accession_number = $1
            ORDER BY filename
            """,
            accession_number,
        )
        return [_row_to_attachment(row) for row in rows]

    # -- XBRL facts --------------------------------------------------------

    async def insert_xbrl_fact(self, fact: XBRLFactRecord) -> XBRLFactRecord:
        """Insert an XBRL fact."""
        row = await self._db.fetchrow(
            """
            INSERT INTO filing_xbrl_facts (
                accession_number, taxonomy, concept_name, value,
                unit, period_start, period_end, instant_date,
                decimals, segment, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING *
            """,
            fact.accession_number,
            fact.taxonomy,
            fact.concept_name,
            fact.value,
            fact.unit,
            fact.period_start,
            fact.period_end,
            fact.instant_date,
            fact.decimals,
            fact.segment,
            json.dumps(fact.metadata),
        )
        return _row_to_xbrl_fact(row)

    async def get_xbrl_facts(
        self,
        accession_number: str,
        *,
        concept_name: str | None = None,
    ) -> list[XBRLFactRecord]:
        """Get XBRL facts for a filing, optionally filtered by concept."""
        if concept_name is not None:
            rows = await self._db.fetch(
                """
                SELECT * FROM filing_xbrl_facts
                WHERE accession_number = $1 AND concept_name = $2
                ORDER BY period_end DESC NULLS LAST
                """,
                accession_number,
                concept_name,
            )
        else:
            rows = await self._db.fetch(
                """
                SELECT * FROM filing_xbrl_facts
                WHERE accession_number = $1
                ORDER BY concept_name, period_end DESC NULLS LAST
                """,
                accession_number,
            )
        return [_row_to_xbrl_fact(row) for row in rows]
