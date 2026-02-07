"""Data models for the security master."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Security:
    """A tracked security (stock/ETF) in the security master.

    Uses composite key (ticker, exchange) to uniquely identify securities
    across different exchanges (e.g., Samsung as 005930.KS on KRX).
    """

    ticker: str
    exchange: str = "US"
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    sector: str = ""
    country: str = "US"
    currency: str = "USD"
    figi: str | None = None
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None
