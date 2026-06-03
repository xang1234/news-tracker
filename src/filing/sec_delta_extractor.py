"""Compute SEC filing-delta events from Company Facts payloads."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime
from typing import Any

from src.filing.sec_companyfacts_parser import extract_companyfacts_observations
from src.filing.sec_delta_models import (
    SECFactObservation,
    SECFilingDeltaEvent,
    make_sec_delta_event_id,
)
from src.filing.sec_structured import SECStructuredPayloadRecord

DIRECT_FACT_EVENT_TYPES = {
    "Revenues": "revenue_growth",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue_growth",
    "SalesRevenueNet": "revenue_growth",
    "InventoryNet": "inventory_change",
    "InventoryFinishedGoodsNetOfReserves": "inventory_change",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex_change",
    "ResearchAndDevelopmentExpense": "rnd_change",
}

REVENUE_FACTS = frozenset(
    {
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    }
)
GROSS_PROFIT_FACT = "GrossProfit"
PeriodKey = tuple[date | None, date, int | None, str | None]
MarginKey = tuple[str, str, date | None, date, int | None, str | None]


def compute_sec_filing_delta_events(
    companyfacts: SECStructuredPayloadRecord,
) -> list[SECFilingDeltaEvent]:
    """Compute narrow, point-in-time SEC filing deltas from Company Facts."""
    if companyfacts.payload_type != "companyfacts":
        raise ValueError("SEC filing deltas require a companyfacts payload")

    observations = extract_companyfacts_observations(companyfacts)
    events: list[SECFilingDeltaEvent] = []
    events.extend(_compute_restatement_events(observations))
    events.extend(_compute_direct_fact_delta_events(observations))
    events.extend(_compute_margin_events(observations))
    return sorted(events, key=lambda event: (event.available_at, event.event_type, event.event_id))


def _compute_restatement_events(
    observations: list[SECFactObservation],
) -> list[SECFilingDeltaEvent]:
    events: list[SECFilingDeltaEvent] = []
    grouped: dict[tuple[str, str, str, str, PeriodKey], list[SECFactObservation]] = defaultdict(
        list
    )
    for observation in observations:
        grouped[
            (
                observation.taxonomy,
                observation.fact_name,
                observation.unit,
                observation.cik,
                observation.period_key,
            )
        ].append(observation)

    for period_observations in grouped.values():
        ordered = _sort_observations(period_observations)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.value == current.value and not current.amended_filing:
                continue
            events.append(
                _build_event(
                    event_type="restatement",
                    current=current,
                    previous=previous,
                    metadata={
                        "amended_filing": current.amended_filing,
                        "same_period_restatement": True,
                        "current_frame": current.frame,
                        "previous_frame": previous.frame,
                    },
                )
            )
    return events


def _compute_direct_fact_delta_events(
    observations: list[SECFactObservation],
) -> list[SECFilingDeltaEvent]:
    events: list[SECFilingDeltaEvent] = []
    grouped: dict[tuple[str, str, str, str], list[SECFactObservation]] = defaultdict(list)
    for observation in observations:
        if observation.fact_name not in DIRECT_FACT_EVENT_TYPES:
            continue
        grouped[
            (observation.cik, observation.taxonomy, observation.fact_name, observation.unit)
        ].append(observation)

    for fact_observations in grouped.values():
        latest_by_period = _latest_observation_by_period(fact_observations)
        for previous, current in zip(latest_by_period, latest_by_period[1:], strict=False):
            if current.value == previous.value:
                continue
            events.append(
                _build_event(
                    event_type=DIRECT_FACT_EVENT_TYPES[current.fact_name],
                    current=current,
                    previous=previous,
                    metadata=_delta_metadata(current, previous),
                )
            )
    return events


def _compute_margin_events(
    observations: list[SECFactObservation],
) -> list[SECFilingDeltaEvent]:
    revenue = [
        observation for observation in observations if observation.fact_name in REVENUE_FACTS
    ]
    gross_profit = [
        observation for observation in observations if observation.fact_name == GROSS_PROFIT_FACT
    ]
    revenue_by_key = {
        _margin_key(observation): observation
        for observation in _latest_observation_by_period(revenue)
        if observation.value != 0
    }
    profit_by_key = {
        _margin_key(observation): observation
        for observation in _latest_observation_by_period(gross_profit)
    }

    margin_observations: list[SECFactObservation] = []
    margin_sources: dict[PeriodKey, tuple[SECFactObservation, SECFactObservation]] = {}
    for key, revenue_observation in revenue_by_key.items():
        profit_observation = profit_by_key.get(key)
        if profit_observation is None:
            continue
        margin = SECFactObservation(
            cik=revenue_observation.cik,
            taxonomy=revenue_observation.taxonomy,
            fact_name="GrossMargin",
            unit="ratio",
            accession_number=revenue_observation.accession_number,
            form=revenue_observation.form,
            filed_date=max(revenue_observation.filed_date, profit_observation.filed_date),
            period_start=revenue_observation.period_start,
            period_end=revenue_observation.period_end,
            value=profit_observation.value / revenue_observation.value,
            fy=revenue_observation.fy,
            fp=revenue_observation.fp,
            frame=revenue_observation.frame,
            fetched_at=max(revenue_observation.fetched_at, profit_observation.fetched_at),
            source_payload_hash=revenue_observation.source_payload_hash,
            source_url=revenue_observation.source_url,
        )
        margin_observations.append(margin)
        margin_sources[margin.period_key] = (revenue_observation, profit_observation)

    events: list[SECFilingDeltaEvent] = []
    ordered_margins = _sort_observations(margin_observations)
    for previous, current in zip(ordered_margins, ordered_margins[1:], strict=False):
        if current.value >= previous.value:
            continue
        current_revenue, current_profit = margin_sources[current.period_key]
        previous_revenue, previous_profit = margin_sources[previous.period_key]
        events.append(
            _build_event(
                event_type="margin_compression",
                current=current,
                previous=previous,
                metadata={
                    **_delta_metadata(current, previous),
                    "current_gross_profit_accession": current_profit.accession_number,
                    "current_revenue_accession": current_revenue.accession_number,
                    "previous_gross_profit_accession": previous_profit.accession_number,
                    "previous_revenue_accession": previous_revenue.accession_number,
                },
            )
        )
    return events


def _margin_key(
    observation: SECFactObservation,
) -> MarginKey:
    return (
        observation.cik,
        observation.unit,
        observation.period_start,
        observation.period_end,
        observation.fy,
        observation.fp,
    )


def _latest_observation_by_period(
    observations: list[SECFactObservation],
) -> list[SECFactObservation]:
    grouped: dict[PeriodKey, list[SECFactObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.period_key].append(observation)
    latest = [_sort_observations(period_values)[-1] for period_values in grouped.values()]
    return _sort_observations(latest)


def _sort_observations(observations: list[SECFactObservation]) -> list[SECFactObservation]:
    return sorted(
        observations,
        key=lambda observation: (
            observation.period_end,
            observation.filed_date,
            observation.accession_number,
        ),
    )


def _delta_metadata(
    current: SECFactObservation,
    previous: SECFactObservation,
) -> dict[str, Any]:
    period_gap_years = (
        current.fy - previous.fy if current.fy is not None and previous.fy is not None else None
    )
    metadata: dict[str, Any] = {
        "current_frame": current.frame,
        "previous_frame": previous.frame,
        "current_fy": current.fy,
        "previous_fy": previous.fy,
        "current_fp": current.fp,
        "previous_fp": previous.fp,
        "period_gap": bool(period_gap_years is not None and period_gap_years > 1),
    }
    if period_gap_years is not None:
        metadata["period_gap_years"] = period_gap_years
    return metadata


def _build_event(
    *,
    event_type: str,
    current: SECFactObservation,
    previous: SECFactObservation,
    metadata: dict[str, Any],
) -> SECFilingDeltaEvent:
    absolute_delta = current.value - previous.value
    relative_delta = float(absolute_delta / previous.value) if previous.value != 0 else None
    event_id = make_sec_delta_event_id(
        [
            current.cik,
            event_type,
            current.taxonomy,
            current.fact_name,
            current.unit,
            current.accession_number,
            previous.accession_number,
            current.period_start,
            current.period_end,
            current.filed_date,
        ]
    )
    return SECFilingDeltaEvent(
        event_id=event_id,
        cik=current.cik,
        event_type=event_type,
        accession_number=current.accession_number,
        previous_accession_number=previous.accession_number,
        taxonomy=current.taxonomy,
        fact_name=current.fact_name,
        unit=current.unit,
        period_start=current.period_start,
        period_end=current.period_end,
        previous_period_start=previous.period_start,
        previous_period_end=previous.period_end,
        filed_date=current.filed_date,
        previous_filed_date=previous.filed_date,
        form=current.form,
        previous_form=previous.form,
        available_at=datetime.combine(current.filed_date, datetime.min.time(), tzinfo=UTC),
        fetched_at=current.fetched_at,
        current_value=current.value,
        previous_value=previous.value,
        absolute_delta=absolute_delta,
        relative_delta=relative_delta,
        source_payload_hash=current.source_payload_hash,
        source_url=current.source_url,
        metadata=metadata,
    )
