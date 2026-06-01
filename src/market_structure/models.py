"""Models for point-in-time market-structure datasource events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.security_master.schemas import normalize_sec_cik

MARKET_STRUCTURE_EVENT_TYPES = frozenset(
    {
        "finra_short_volume",
        "sec_fail_to_deliver",
    }
)

MARKET_STRUCTURE_SIGNAL_TYPES = frozenset(
    {
        "short_volume_ratio",
        "fails_to_deliver_notional",
    }
)

MARKET_STRUCTURE_ANOMALY_LEVELS = frozenset({"none", "watch", "elevated", "extreme"})


def make_market_structure_event_id(parts: list[Any]) -> str:
    """Build a deterministic event ID from stable source fields."""
    encoded = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode()
    return f"market_structure:{hashlib.sha256(encoded).hexdigest()}"


def _normalize_optional_cik(value: str | None) -> str:
    if not value:
        return ""
    normalized = normalize_sec_cik(value)
    return normalized or ""


def _decimal_to_payload(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


@dataclass(frozen=True)
class MarketStructureSourceFile:
    """One source file payload with lineage needed by parsers."""

    source_name: str
    source_url: str
    content: str
    fetched_at: datetime | None = None


@dataclass
class MarketStructureEvent:
    """One normalized FINRA short-volume or SEC fails-to-deliver row."""

    event_id: str
    event_type: str
    source_name: str
    source_url: str
    source_date: date
    available_at: datetime
    trade_date: date | None = None
    settlement_date: date | None = None
    symbol: str = ""
    security_ticker: str = ""
    security_exchange: str = "US"
    issuer_cik: str = ""
    issuer_name: str = ""
    cusip: str = ""
    market_code: str = ""
    market_name: str = ""
    short_volume: int | None = None
    short_exempt_volume: int | None = None
    total_volume: int | None = None
    short_volume_ratio: Decimal | None = None
    short_exempt_ratio: Decimal | None = None
    fail_quantity: int | None = None
    fail_price: Decimal | None = None
    fail_notional: Decimal | None = None
    signal_type: str | None = None
    anomaly_level: str = "none"
    persistence_count: int = 0
    fetched_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.event_type not in MARKET_STRUCTURE_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {sorted(MARKET_STRUCTURE_EVENT_TYPES)}"
            )
        if (
            self.signal_type is not None
            and self.signal_type not in MARKET_STRUCTURE_SIGNAL_TYPES
        ):
            raise ValueError(
                f"signal_type must be one of {sorted(MARKET_STRUCTURE_SIGNAL_TYPES)}"
            )
        if self.anomaly_level not in MARKET_STRUCTURE_ANOMALY_LEVELS:
            raise ValueError(
                f"anomaly_level must be one of {sorted(MARKET_STRUCTURE_ANOMALY_LEVELS)}"
            )
        self.symbol = self.symbol.upper()
        self.security_ticker = (self.security_ticker or self.symbol).upper()
        self.security_exchange = self.security_exchange or "US"
        self.issuer_cik = _normalize_optional_cik(self.issuer_cik)
        self.cusip = self.cusip.upper()

    def to_payload(self) -> dict[str, Any]:
        """Serialize this event for publication and backtest read models."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "source_date": self.source_date.isoformat(),
            "trade_date": self.trade_date.isoformat() if self.trade_date else None,
            "settlement_date": (
                self.settlement_date.isoformat() if self.settlement_date else None
            ),
            "symbol": self.symbol,
            "security_ticker": self.security_ticker,
            "security_exchange": self.security_exchange,
            "issuer_cik": self.issuer_cik,
            "issuer_name": self.issuer_name,
            "cusip": self.cusip,
            "market_code": self.market_code,
            "market_name": self.market_name,
            "short_volume": self.short_volume,
            "short_exempt_volume": self.short_exempt_volume,
            "total_volume": self.total_volume,
            "short_volume_ratio": _decimal_to_payload(self.short_volume_ratio),
            "short_exempt_ratio": _decimal_to_payload(self.short_exempt_ratio),
            "fail_quantity": self.fail_quantity,
            "fail_price": _decimal_to_payload(self.fail_price),
            "fail_notional": _decimal_to_payload(self.fail_notional),
            "signal_type": self.signal_type,
            "anomaly_level": self.anomaly_level,
            "persistence_count": self.persistence_count,
            "available_at": self.available_at.isoformat(),
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "metadata": self.metadata,
        }
