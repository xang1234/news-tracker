"""Publishable SEC filing-delta payloads and evidence references."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.filing.sec_delta_models import SECFilingDeltaEvent

SEC_DELTA_REASON_CODES = {
    "revenue_growth": "sec_fact_revenue_growth",
    "inventory_change": "sec_fact_inventory_change",
    "capex_change": "sec_fact_capex_change",
    "rnd_change": "sec_fact_rnd_change",
    "margin_compression": "sec_fact_margin_compression",
    "restatement": "sec_fact_restatement",
}


def _decimal_to_payload(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def sec_delta_reason_code(event: SECFilingDeltaEvent) -> str:
    """Return the stable publication reason code for a SEC delta event."""
    return SEC_DELTA_REASON_CODES[event.event_type]


def classify_sec_fact_evidence(event: SECFilingDeltaEvent) -> str:
    """Classify whether an official fact delta confirms or contradicts growth momentum."""
    if event.event_type in {"margin_compression", "restatement"}:
        return "contradictory"
    if event.absolute_delta is None:
        return "neutral"
    if event.absolute_delta > 0:
        return "corroborating"
    if event.absolute_delta < 0:
        return "contradictory"
    return "neutral"


def _lineage_payload(event: SECFilingDeltaEvent) -> dict[str, Any]:
    return {
        "accession_number": event.accession_number,
        "previous_accession_number": event.previous_accession_number,
        "taxonomy": event.taxonomy,
        "fact_name": event.fact_name,
        "unit": event.unit,
        "period_start": event.period_start.isoformat() if event.period_start else None,
        "period_end": event.period_end.isoformat(),
        "previous_period_start": (
            event.previous_period_start.isoformat() if event.previous_period_start else None
        ),
        "previous_period_end": (
            event.previous_period_end.isoformat() if event.previous_period_end else None
        ),
        "filed_date": event.filed_date.isoformat(),
        "previous_filed_date": (
            event.previous_filed_date.isoformat() if event.previous_filed_date else None
        ),
        "form": event.form,
        "previous_form": event.previous_form,
        "available_at": event.available_at.isoformat(),
        "fetched_at": event.fetched_at.isoformat(),
        "source_payload_hash": event.source_payload_hash,
        "source_url": event.source_url,
    }


@dataclass(frozen=True)
class SECDeltaPublicationPayload:
    """Filing-lane payload for an official SEC fact delta."""

    event: SECFilingDeltaEvent
    reason_code: str
    evidence_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_type": "filing_fact",
            "reason_code": self.reason_code,
            "evidence_role": self.evidence_role,
            "event": self.event.to_payload(),
            "lineage": _lineage_payload(self.event),
        }


@dataclass(frozen=True)
class SECFactEvidenceReference:
    """Compact SEC fact lineage reference for other lane outputs."""

    event: SECFilingDeltaEvent
    reason_code: str
    evidence_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event.event_id,
            "cik": self.event.cik,
            "event_type": self.event.event_type,
            "reason_code": self.reason_code,
            "evidence_role": self.evidence_role,
            "current_value": _decimal_to_payload(self.event.current_value),
            "previous_value": _decimal_to_payload(self.event.previous_value),
            "absolute_delta": _decimal_to_payload(self.event.absolute_delta),
            "relative_delta": self.event.relative_delta,
            "lineage": _lineage_payload(self.event),
            "metadata": dict(self.event.metadata),
        }


def build_sec_delta_payload(event: SECFilingDeltaEvent) -> SECDeltaPublicationPayload:
    """Build the filing-lane publishable payload for a SEC delta event."""
    return SECDeltaPublicationPayload(
        event=event,
        reason_code=sec_delta_reason_code(event),
        evidence_role=classify_sec_fact_evidence(event),
    )


def build_sec_fact_evidence(event: SECFilingDeltaEvent) -> SECFactEvidenceReference:
    """Build a compact SEC fact evidence reference for cross-lane payloads."""
    return SECFactEvidenceReference(
        event=event,
        reason_code=sec_delta_reason_code(event),
        evidence_role=classify_sec_fact_evidence(event),
    )
