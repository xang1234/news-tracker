"""Models for structured SEC ownership filing events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.security_master.schemas import normalize_sec_cik

SEC_OWNERSHIP_EVENT_TYPES = frozenset(
    {
        "form4_non_derivative_transaction",
        "form4_derivative_transaction",
        "schedule_13d_ownership",
        "schedule_13g_ownership",
        "13f_position",
    }
)


def make_sec_ownership_event_id(parts: list[Any]) -> str:
    """Build a deterministic ownership event ID from stable filing fields."""
    encoded = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode()
    return f"sec_ownership:{hashlib.sha256(encoded).hexdigest()}"


def _normalize_optional_cik(value: str) -> str:
    if not value:
        return ""
    normalized = normalize_sec_cik(value)
    return normalized or ""


def _decimal_to_payload(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


@dataclass
class SECOwnershipEvent:
    """One normalized event from a Form 4, Schedule 13D/G, or 13F filing."""

    event_id: str
    event_type: str
    accession_number: str
    filing_type: str
    filed_date: date
    available_at: datetime
    issuer_cik: str = ""
    issuer_name: str = ""
    issuer_ticker: str | None = None
    filer_cik: str = ""
    filer_name: str = ""
    security_title: str = ""
    transaction_code: str | None = None
    transaction_date: date | None = None
    transaction_shares: Decimal | None = None
    transaction_price_per_share: Decimal | None = None
    transaction_acquired_disposed_code: str | None = None
    shares_owned_following: Decimal | None = None
    derivative_underlying_shares: Decimal | None = None
    ownership_percent: Decimal | None = None
    position_cusip: str | None = None
    position_shares: Decimal | None = None
    position_value_usd: Decimal | None = None
    previous_position_shares: Decimal | None = None
    position_delta_shares: Decimal | None = None
    is_amendment: bool = False
    fetched_at: datetime | None = None
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.event_type not in SEC_OWNERSHIP_EVENT_TYPES:
            raise ValueError(f"event_type must be one of {sorted(SEC_OWNERSHIP_EVENT_TYPES)}")
        self.issuer_cik = _normalize_optional_cik(self.issuer_cik)
        self.filer_cik = _normalize_optional_cik(self.filer_cik)

    def to_payload(self) -> dict[str, Any]:
        """Serialize this ownership event for downstream publication."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "accession_number": self.accession_number,
            "filing_type": self.filing_type,
            "filed_date": self.filed_date.isoformat(),
            "issuer_cik": self.issuer_cik,
            "issuer_name": self.issuer_name,
            "issuer_ticker": self.issuer_ticker,
            "filer_cik": self.filer_cik,
            "filer_name": self.filer_name,
            "security_title": self.security_title,
            "transaction_code": self.transaction_code,
            "transaction_date": self.transaction_date.isoformat()
            if self.transaction_date
            else None,
            "transaction_shares": _decimal_to_payload(self.transaction_shares),
            "transaction_price_per_share": _decimal_to_payload(self.transaction_price_per_share),
            "transaction_acquired_disposed_code": self.transaction_acquired_disposed_code,
            "shares_owned_following": _decimal_to_payload(self.shares_owned_following),
            "derivative_underlying_shares": _decimal_to_payload(self.derivative_underlying_shares),
            "ownership_percent": _decimal_to_payload(self.ownership_percent),
            "position_cusip": self.position_cusip,
            "position_shares": _decimal_to_payload(self.position_shares),
            "position_value_usd": _decimal_to_payload(self.position_value_usd),
            "previous_position_shares": _decimal_to_payload(self.previous_position_shares),
            "position_delta_shares": _decimal_to_payload(self.position_delta_shares),
            "is_amendment": self.is_amendment,
            "available_at": self.available_at.isoformat(),
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "source_url": self.source_url,
            "metadata": self.metadata,
        }
