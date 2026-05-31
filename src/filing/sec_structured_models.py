"""Shared models for SEC structured payload ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.security_master.schemas import normalize_sec_cik

STRUCTURED_PAYLOAD_TYPES = frozenset({"companyfacts", "submissions"})


class SECStructuredDataError(Exception):
    """Raised when SEC structured data cannot be fetched or validated."""


@dataclass
class SECStructuredPayloadRecord:
    """Cached SEC JSON payload with issuer and accession lineage."""

    cik: str
    payload_type: str
    source_url: str
    payload_hash: str
    payload: dict[str, Any]
    accession_numbers: list[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        normalized_cik = normalize_sec_cik(self.cik)
        if normalized_cik is None:
            raise ValueError("cik must be a non-empty SEC CIK")
        self.cik = normalized_cik
        if self.payload_type not in STRUCTURED_PAYLOAD_TYPES:
            raise ValueError(f"payload_type must be one of {sorted(STRUCTURED_PAYLOAD_TYPES)}")
        self.accession_numbers = sorted(set(self.accession_numbers))
