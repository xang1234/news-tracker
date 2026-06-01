"""Publishable alert and read-model payloads for market-plumbing events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from src.alerts.schemas import Alert
from src.filing.sec_ownership_events import SECOwnershipEvent
from src.market_structure import MarketStructureEvent


@dataclass(frozen=True)
class MarketPlumbingSignalSpec:
    """Stable publication labels for one normalized datasource event type."""

    signal_class: str
    signal_label: str
    reason_code: str
    source_family: str
    default_severity: str = "info"
    guardrails: tuple[str, ...] = ()


SHORT_VOLUME_GUARDRAILS = (
    "FINRA short-volume is daily trade-flow volume, not short interest or an "
    "open short position.",
    "High ratios can reflect market making, hedging, or intraday liquidity and "
    "need corroboration.",
)

FTD_GUARDRAILS = (
    "SEC fails-to-deliver rows are settlement failures, not proof of an open "
    "short position or abusive short selling.",
    "Use alongside liquidity, corporate actions, recall dates, and persistence "
    "before escalating.",
)

MARKET_STRUCTURE_GUARDRAILS: dict[str, tuple[str, ...]] = {
    "finra_short_volume": SHORT_VOLUME_GUARDRAILS,
    "sec_fail_to_deliver": FTD_GUARDRAILS,
}

_OWNERSHIP_SPECS: dict[str, MarketPlumbingSignalSpec] = {
    "form4_non_derivative_transaction": MarketPlumbingSignalSpec(
        signal_class="insider_ownership",
        signal_label="Insider ownership transaction",
        reason_code="sec_form4_insider_transaction",
        source_family="sec_ownership",
    ),
    "form4_derivative_transaction": MarketPlumbingSignalSpec(
        signal_class="insider_ownership",
        signal_label="Insider ownership transaction",
        reason_code="sec_form4_derivative_transaction",
        source_family="sec_ownership",
    ),
    "schedule_13d_ownership": MarketPlumbingSignalSpec(
        signal_class="activist_ownership",
        signal_label="Activist ownership disclosure",
        reason_code="sec_schedule_13d_ownership",
        source_family="sec_ownership",
        default_severity="warning",
    ),
    "schedule_13g_ownership": MarketPlumbingSignalSpec(
        signal_class="institutional_holdings",
        signal_label="Institutional holdings update",
        reason_code="sec_schedule_13g_ownership",
        source_family="sec_ownership",
    ),
    "13f_position": MarketPlumbingSignalSpec(
        signal_class="institutional_holdings",
        signal_label="Institutional holdings update",
        reason_code="sec_13f_position_change",
        source_family="sec_ownership",
    ),
}

_MARKET_STRUCTURE_SPECS: dict[str, MarketPlumbingSignalSpec] = {
    "finra_short_volume": MarketPlumbingSignalSpec(
        signal_class="short_volume_anomaly",
        signal_label="Short-volume anomaly",
        reason_code="finra_short_volume_ratio",
        source_family="market_structure",
        default_severity="warning",
        guardrails=SHORT_VOLUME_GUARDRAILS,
    ),
    "sec_fail_to_deliver": MarketPlumbingSignalSpec(
        signal_class="fails_to_deliver_anomaly",
        signal_label="Fails-to-deliver anomaly",
        reason_code="sec_fails_to_deliver_notional",
        source_family="market_structure",
        default_severity="warning",
        guardrails=FTD_GUARDRAILS,
    ),
}

MARKET_PLUMBING_ALERT_TYPES = frozenset(
    spec.signal_class for spec in (*_OWNERSHIP_SPECS.values(), *_MARKET_STRUCTURE_SPECS.values())
)


def _iso_date(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _iso_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _ownership_lineage(event: SECOwnershipEvent) -> dict[str, Any]:
    return {
        "accession_number": event.accession_number,
        "filing_type": event.filing_type,
        "filed_date": event.filed_date.isoformat(),
        "available_at": event.available_at.isoformat(),
        "fetched_at": _iso_datetime(event.fetched_at),
        "source_url": event.source_url,
    }


def _market_structure_lineage(event: MarketStructureEvent) -> dict[str, Any]:
    return {
        "source_name": event.source_name,
        "source_url": event.source_url,
        "source_date": event.source_date.isoformat(),
        "trade_date": _iso_date(event.trade_date),
        "settlement_date": _iso_date(event.settlement_date),
        "available_at": event.available_at.isoformat(),
        "fetched_at": _iso_datetime(event.fetched_at),
    }


def _ownership_subject(event: SECOwnershipEvent) -> dict[str, Any]:
    return {
        "subject_type": "security",
        "subject_id": event.issuer_ticker or event.issuer_cik or event.event_id,
        "issuer_cik": event.issuer_cik,
        "issuer_name": event.issuer_name,
        "ticker": event.issuer_ticker,
        "filer_cik": event.filer_cik,
        "filer_name": event.filer_name,
    }


def _market_structure_subject(event: MarketStructureEvent) -> dict[str, Any]:
    return {
        "subject_type": "security",
        "subject_id": event.symbol or event.security_ticker or event.cusip or event.event_id,
        "issuer_cik": event.issuer_cik,
        "issuer_name": event.issuer_name,
        "ticker": event.security_ticker or event.symbol,
        "cusip": event.cusip,
    }


def _ownership_alert_data(
    event: SECOwnershipEvent,
    spec: MarketPlumbingSignalSpec,
) -> dict[str, Any]:
    return {
        "signal_class": spec.signal_class,
        "signal_label": spec.signal_label,
        "reason_code": spec.reason_code,
        "source_family": spec.source_family,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "issuer_cik": event.issuer_cik,
        "issuer_ticker": event.issuer_ticker,
        "filer_cik": event.filer_cik,
        "available_at": event.available_at.isoformat(),
        "lineage": _ownership_lineage(event),
        "event": event.to_payload(),
    }


def _market_structure_alert_data(
    event: MarketStructureEvent,
    spec: MarketPlumbingSignalSpec,
) -> dict[str, Any]:
    return {
        "signal_class": spec.signal_class,
        "signal_label": spec.signal_label,
        "reason_code": spec.reason_code,
        "source_family": spec.source_family,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "symbol": event.symbol,
        "cusip": event.cusip,
        "available_at": event.available_at.isoformat(),
        "caveats": list(spec.guardrails),
        "lineage": _market_structure_lineage(event),
        "event": event.to_payload(),
    }


def _market_structure_severity(
    event: MarketStructureEvent,
    spec: MarketPlumbingSignalSpec,
) -> str:
    return {
        "extreme": "critical",
        "elevated": "warning",
        "watch": "warning",
    }.get(event.anomaly_level, spec.default_severity)


def _subject_label(subject: dict[str, Any]) -> str:
    return str(subject.get("ticker") or subject.get("issuer_cik") or subject["subject_id"])


def _build_alert(
    *,
    spec: MarketPlumbingSignalSpec,
    subject: dict[str, Any],
    trigger_data: dict[str, Any],
    theme_id: str,
    severity: str,
) -> Alert:
    label = _subject_label(subject)
    return Alert(
        theme_id=theme_id,
        subject_type=str(subject["subject_type"]),
        subject_id=str(subject["subject_id"]),
        trigger_type=spec.signal_class,
        severity=severity,
        title=f"{spec.signal_label}: {label}",
        message=f"{spec.signal_label} detected for {label}.",
        trigger_data=trigger_data,
    )


def build_market_plumbing_alerts(
    *,
    ownership_events: Iterable[SECOwnershipEvent] | None = None,
    market_structure_events: Iterable[MarketStructureEvent] | None = None,
    theme_id: str,
) -> list[Alert]:
    """Build alert records for ownership and market-plumbing datasource events.

    ``theme_id`` must reference an existing theme until the alerts table is fully
    generic; the subject fields carry the security identity.
    """
    if not theme_id.strip():
        raise ValueError("theme_id is required for market-plumbing alerts")

    alerts: list[Alert] = []

    for event in ownership_events or ():
        spec = _OWNERSHIP_SPECS[event.event_type]
        subject = _ownership_subject(event)
        alerts.append(
            _build_alert(
                spec=spec,
                subject=subject,
                trigger_data=_ownership_alert_data(event, spec),
                theme_id=theme_id,
                severity=spec.default_severity,
            )
        )

    for event in market_structure_events or ():
        if not _is_market_structure_anomaly(event):
            continue
        spec = _MARKET_STRUCTURE_SPECS[event.event_type]
        subject = _market_structure_subject(event)
        alerts.append(
            _build_alert(
                spec=spec,
                subject=subject,
                trigger_data=_market_structure_alert_data(event, spec),
                theme_id=theme_id,
                severity=_market_structure_severity(event, spec),
            )
        )

    return alerts


def _is_market_structure_anomaly(event: MarketStructureEvent) -> bool:
    return event.anomaly_level != "none"


def _read_model_payload(
    *,
    event_payload: dict[str, Any],
    lineage: dict[str, Any],
    subject: dict[str, Any],
    spec: MarketPlumbingSignalSpec,
) -> dict[str, Any]:
    return {
        "object_type": "market_plumbing_signal",
        "object_id": event_payload["event_id"],
        "signal_class": spec.signal_class,
        "signal_label": spec.signal_label,
        "reason_code": spec.reason_code,
        "source_family": spec.source_family,
        "subject": subject,
        "event": event_payload,
        "lineage": lineage,
        "guardrails": list(spec.guardrails),
    }


def build_market_plumbing_read_model(
    *,
    ownership_events: Iterable[SECOwnershipEvent] | None = None,
    market_structure_events: Iterable[MarketStructureEvent] | None = None,
) -> list[dict[str, Any]]:
    """Build publish/read-model payloads with source lineage and guardrails."""
    payloads: list[dict[str, Any]] = []

    for event in ownership_events or ():
        spec = _OWNERSHIP_SPECS[event.event_type]
        payloads.append(
            _read_model_payload(
                event_payload=event.to_payload(),
                lineage=_ownership_lineage(event),
                subject=_ownership_subject(event),
                spec=spec,
            )
        )

    for event in market_structure_events or ():
        if not _is_market_structure_anomaly(event):
            continue
        spec = _MARKET_STRUCTURE_SPECS[event.event_type]
        payloads.append(
            _read_model_payload(
                event_payload=event.to_payload(),
                lineage=_market_structure_lineage(event),
                subject=_market_structure_subject(event),
                spec=spec,
            )
        )

    return payloads
