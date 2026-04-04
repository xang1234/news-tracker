"""Schema definitions for the filing lane.

Defines the output shape that all filing providers must emit and the
SEC-specific identifiers needed for lineage tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

VALID_FILING_TYPES = frozenset(
    {
        "10-K",
        "10-Q",
        "8-K",
        "DEF 14A",
        "S-1",
        "S-3",
        "20-F",
        "6-K",
        "SC 13D",
        "SC 13G",
        "4",
        "13F-HR",
    }
)

VALID_FILING_STATUSES = frozenset(
    {"pending", "fetched", "parsed", "failed", "skipped"}
)


@dataclass
class FilingIdentity:
    """SEC-specific identifiers for a filing.

    These fields are required for lineage and are the canonical
    way to reference a specific SEC filing across the system.

    Attributes:
        cik: SEC Central Index Key (company identifier).
        accession_number: Unique filing identifier (e.g., "0001234567-24-000123").
        filing_type: SEC form type (10-K, 10-Q, 8-K, etc.).
        filed_date: When the filing was submitted to the SEC.
        period_of_report: The reporting period end date.
        company_name: Filer name as reported to the SEC.
        ticker: Primary ticker symbol (if resolvable).
    """

    cik: str
    accession_number: str
    filing_type: str
    filed_date: date
    period_of_report: date | None = None
    company_name: str = ""
    ticker: str | None = None

    def __post_init__(self) -> None:
        if self.filing_type not in VALID_FILING_TYPES:
            raise ValueError(
                f"Invalid filing_type {self.filing_type!r}. "
                f"Must be one of {sorted(VALID_FILING_TYPES)}"
            )


@dataclass
class FilingSection:
    """A parsed section from a filing.

    Attributes:
        section_id: Unique identifier within the filing.
        section_name: Human-readable section name (e.g., "Risk Factors").
        section_type: Classification (narrative, financial, exhibit, etc.).
        content: Raw text content of the section.
        word_count: Number of words in the content.
        metadata: Extensible metadata (XBRL tags, table indicators, etc.).
    """

    section_id: str
    section_name: str
    section_type: str = "narrative"
    content: str = ""
    word_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilingResult:
    """The complete output of a filing provider for one filing.

    This is the contract shape that all providers must produce.
    Downstream tasks (claim extraction, NER, etc.) consume this.

    Attributes:
        identity: SEC-specific identifiers for lineage.
        sections: Parsed sections from the filing.
        raw_url: URL to the filing on SEC EDGAR.
        status: Processing status (fetched, parsed, failed, skipped).
        error_message: Error details if status is 'failed'.
        provider: Which provider produced this result.
        fetched_at: When the filing was retrieved.
        metadata: Extensible metadata (file size, content hash, etc.).
    """

    identity: FilingIdentity
    sections: list[FilingSection] = field(default_factory=list)
    raw_url: str = ""
    status: str = "fetched"
    error_message: str | None = None
    provider: str = ""
    fetched_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_FILING_STATUSES:
            raise ValueError(
                f"Invalid filing status {self.status!r}. "
                f"Must be one of {sorted(VALID_FILING_STATUSES)}"
            )

    @property
    def total_word_count(self) -> int:
        """Total words across all sections."""
        return sum(s.word_count for s in self.sections)

    @property
    def section_names(self) -> list[str]:
        """List of section names in order."""
        return [s.section_name for s in self.sections]
