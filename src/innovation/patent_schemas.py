"""Typed models shared by innovation patent ingestion components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

ODP_PATENT_SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"
PATENT_SOURCE_ODP = "uspto_odp_patentsview_transition"
PATENT_SOURCE_BULK = "uspto_patentsview_bulk"
VALID_PATENT_EVENT_TYPES = frozenset({"application", "grant"})


class PatentProviderError(Exception):
    """Base exception for patent provider failures."""


class MissingPatentProviderCredentialError(PatentProviderError):
    """Raised when an API-backed patent provider lacks a required free key."""


class PatentProviderResponseError(PatentProviderError):
    """Raised when an upstream patent provider returns unusable data."""


class StalePatentSnapshotError(PatentProviderError):
    """Raised when a bulk snapshot is too old to ingest by default."""


@dataclass(frozen=True)
class PatentQuery:
    """Search criteria for patent/application metadata."""

    assignees: list[str] = field(default_factory=list)
    cpc_classes: list[str] = field(default_factory=list)
    ipc_classes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    start: date | None = None
    end: date | None = None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must be on or before end")
        criteria = self.assignees + self.cpc_classes + self.ipc_classes + self.keywords
        if not any(value.strip() for value in criteria):
            raise ValueError("at least one assignee, class, or keyword is required")


@dataclass(frozen=True)
class PatentRecord:
    """Normalized patent/application metadata from API or bulk files."""

    patent_id: str
    application_id: str
    patent_family_id: str
    title: str
    abstract: str
    assignees: list[str]
    cpc_classes: list[str]
    ipc_classes: list[str]
    application_date: date | None
    grant_date: date | None
    source_url: str
    source_attribution: str
    fetched_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.patent_id and not self.application_id:
            raise ValueError("patent_id or application_id is required")
        if self.fetched_at.tzinfo is None or self.fetched_at.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")

    @property
    def event_type(self) -> str:
        """Primary event represented by this record."""
        return "grant" if self.grant_date is not None else "application"

    @property
    def event_date(self) -> date | None:
        """Best dated innovation event for this record."""
        return self.grant_date or self.application_date


@dataclass(frozen=True)
class PatentSignal:
    """Patent evidence linked to issuer, security, and theme concepts."""

    patent_id: str
    patent_family_id: str
    event_type: str
    event_date: date
    title: str
    issuer_concept_id: str
    security_concept_id: str
    theme_id: str
    confidence: float
    confidence_reasons: list[str]
    source_lineage: dict[str, Any]
    metadata: dict[str, Any]
    source_url: str
    fetched_at: datetime

    def __post_init__(self) -> None:
        if self.event_type not in VALID_PATENT_EVENT_TYPES:
            raise ValueError(f"event_type must be one of {sorted(VALID_PATENT_EVENT_TYPES)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.fetched_at.tzinfo is None or self.fetched_at.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")
