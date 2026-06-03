"""Typed models for research and preprint innovation evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

VALID_RESEARCH_SOURCES = frozenset({"openalex", "arxiv"})
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"


class ResearchProviderError(Exception):
    """Base exception for research provider failures."""


class ResearchProviderResponseError(ResearchProviderError):
    """Raised when a provider returns invalid response data."""


@dataclass(frozen=True)
class ResearchQuery:
    """Search criteria for research metadata providers."""

    topics: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    start: date | None = None
    end: date | None = None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must be on or before end")
        criteria = self.topics + self.categories + self.institutions
        if not any(value.strip() for value in criteria):
            raise ValueError("at least one topic, category, or institution is required")


@dataclass(frozen=True)
class ResearchRecord:
    """Normalized work/preprint metadata from OpenAlex or arXiv."""

    source: str
    record_id: str
    title: str
    abstract: str
    authors: list[str]
    institutions: list[str]
    topics: list[str]
    categories: list[str]
    published_date: date
    url: str
    doi: str | None
    arxiv_id: str | None
    source_lineage: dict[str, Any]
    fetched_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in VALID_RESEARCH_SOURCES:
            raise ValueError(f"source must be one of {sorted(VALID_RESEARCH_SOURCES)}")
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if self.fetched_at.tzinfo is None or self.fetched_at.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")


@dataclass(frozen=True)
class ResearchSignal:
    """Research evidence linked to issuer, security, and theme concepts."""

    source: str
    record_id: str
    published_date: date
    title: str
    issuer_concept_id: str
    security_concept_id: str
    theme_id: str
    confidence: float
    confidence_reasons: list[str]
    source_lineage: dict[str, Any]
    metadata: dict[str, Any]
    url: str
    fetched_at: datetime

    def __post_init__(self) -> None:
        if self.source not in VALID_RESEARCH_SOURCES:
            raise ValueError(f"source must be one of {sorted(VALID_RESEARCH_SOURCES)}")
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.fetched_at.tzinfo is None or self.fetched_at.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")
