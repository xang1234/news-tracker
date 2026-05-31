"""Data models for the security master."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def normalize_sec_cik(value: str | int | None) -> str | None:
    """Normalize SEC CIK values to the 10-digit Company Facts form."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text.startswith("CIK"):
        text = text[3:]
    if not text.isdigit() or len(text) > 10:
        raise ValueError("sec_cik must be numeric and at most 10 digits")
    return text.zfill(10)


@dataclass
class SecurityIdentifierLineage:
    """Auditable provenance for a security identifier.

    Examples include CIKs from SEC company_tickers.json, issuer names from
    SEC submissions, or externally curated LEIs. Records are stored as JSONB
    on the security row so downstream SEC facts can explain how an identifier
    was attached.
    """

    identifier_type: str
    value: str
    source: str
    observed_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier_type.strip():
            raise ValueError("identifier_lineage identifier_type must be non-empty")
        if not self.value.strip():
            raise ValueError("identifier_lineage value must be non-empty")
        if not self.source.strip():
            raise ValueError("identifier_lineage source must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("identifier_lineage confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONB storage."""
        return {
            "identifier_type": self.identifier_type,
            "value": self.value,
            "source": self.source,
            "observed_at": self.observed_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_raw(
        cls,
        value: "SecurityIdentifierLineage | dict[str, Any]",
    ) -> "SecurityIdentifierLineage":
        """Build a lineage record from an existing dataclass or JSON dict."""
        if isinstance(value, SecurityIdentifierLineage):
            return value
        raw = value
        return cls(
            identifier_type=str(raw.get("identifier_type", "")),
            value=str(raw.get("value", "")),
            source=str(raw.get("source", "")),
            observed_at=raw.get("observed_at"),
            valid_from=raw.get("valid_from"),
            valid_to=raw.get("valid_to"),
            confidence=float(raw.get("confidence", 1.0)),
            metadata=dict(raw.get("metadata") or {}),
        )


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
    sec_cik: str | None = None
    issuer_name: str = ""
    former_names: list[str] = field(default_factory=list)
    external_identifiers: dict[str, Any] = field(default_factory=dict)
    identifier_lineage: list[SecurityIdentifierLineage] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        self.sec_cik = normalize_sec_cik(self.sec_cik)
        self.identifier_lineage = [
            SecurityIdentifierLineage.from_raw(record) for record in self.identifier_lineage
        ]
