"""Models for point-in-time SEC filing-delta events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.security_master.schemas import normalize_sec_cik

SEC_FILING_DELTA_EVENT_TYPES = frozenset(
    {
        "revenue_growth",
        "inventory_change",
        "capex_change",
        "rnd_change",
        "margin_compression",
        "restatement",
    }
)


def _decimal_to_payload(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _datetime_to_payload(value: datetime) -> str:
    return value.isoformat()


def _date_to_payload(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def make_sec_delta_event_id(parts: list[Any]) -> str:
    """Build a deterministic event id from stable SEC delta fields."""
    encoded = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode()
    return f"sec_delta:{hashlib.sha256(encoded).hexdigest()}"


@dataclass
class SECFactObservation:
    """One parsed Company Facts observation from a specific SEC filing."""

    cik: str
    taxonomy: str
    fact_name: str
    unit: str
    accession_number: str
    form: str
    filed_date: date
    period_end: date
    value: Decimal
    fetched_at: datetime
    source_payload_hash: str
    source_url: str
    period_start: date | None = None
    fy: int | None = None
    fp: str | None = None
    frame: str | None = None

    @property
    def period_key(self) -> tuple[date | None, date, int | None, str | None]:
        return (self.period_start, self.period_end, self.fy, self.fp)

    @property
    def amended_filing(self) -> bool:
        return self.form.upper().endswith("/A")


@dataclass
class SECFilingDeltaEvent:
    """Narrow SEC filing-derived fact delta with point-in-time lineage."""

    event_id: str
    cik: str
    event_type: str
    accession_number: str
    taxonomy: str
    fact_name: str
    unit: str
    period_end: date
    filed_date: date
    form: str
    available_at: datetime
    fetched_at: datetime
    source_payload_hash: str
    source_url: str
    previous_accession_number: str | None = None
    period_start: date | None = None
    previous_period_start: date | None = None
    previous_period_end: date | None = None
    previous_filed_date: date | None = None
    previous_form: str | None = None
    current_value: Decimal | None = None
    previous_value: Decimal | None = None
    absolute_delta: Decimal | None = None
    relative_delta: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        normalized_cik = normalize_sec_cik(self.cik)
        if normalized_cik is None:
            raise ValueError("cik must be a non-empty SEC CIK")
        self.cik = normalized_cik
        if self.event_type not in SEC_FILING_DELTA_EVENT_TYPES:
            raise ValueError(f"event_type must be one of {sorted(SEC_FILING_DELTA_EVENT_TYPES)}")

    def to_payload(self) -> dict[str, Any]:
        """Serialize this event for publication manifests and read models."""
        return {
            "event_id": self.event_id,
            "cik": self.cik,
            "event_type": self.event_type,
            "accession_number": self.accession_number,
            "previous_accession_number": self.previous_accession_number,
            "taxonomy": self.taxonomy,
            "fact_name": self.fact_name,
            "unit": self.unit,
            "period_start": _date_to_payload(self.period_start),
            "period_end": self.period_end.isoformat(),
            "previous_period_start": _date_to_payload(self.previous_period_start),
            "previous_period_end": _date_to_payload(self.previous_period_end),
            "filed_date": self.filed_date.isoformat(),
            "previous_filed_date": _date_to_payload(self.previous_filed_date),
            "form": self.form,
            "previous_form": self.previous_form,
            "available_at": _datetime_to_payload(self.available_at),
            "fetched_at": _datetime_to_payload(self.fetched_at),
            "current_value": _decimal_to_payload(self.current_value),
            "previous_value": _decimal_to_payload(self.previous_value),
            "absolute_delta": _decimal_to_payload(self.absolute_delta),
            "relative_delta": self.relative_delta,
            "source_payload_hash": self.source_payload_hash,
            "source_url": self.source_url,
            "metadata": self.metadata,
        }
