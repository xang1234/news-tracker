"""Tests for point-in-time SEC filing-delta event computation."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from src.filing.sec_delta_events import compute_sec_filing_delta_events
from src.filing.sec_structured import SECStructuredPayloadRecord

FETCHED_AT = datetime(2026, 5, 31, 8, 30, tzinfo=UTC)


def _companyfacts_record(facts: dict[str, Any]) -> SECStructuredPayloadRecord:
    return SECStructuredPayloadRecord(
        cik="0000320193",
        payload_type="companyfacts",
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        payload_hash="sha256:companyfacts",
        payload={
            "cik": 320193,
            "entityName": "Apple Inc.",
            "facts": facts,
        },
        accession_numbers=[
            "0000320193-22-000100",
            "0000320193-23-000100",
            "0000320193-24-000100",
        ],
        fetched_at=FETCHED_AT,
    )


def _fact(
    *,
    accn: str,
    value: int | float,
    fy: int,
    fp: str = "FY",
    form: str = "10-K",
    filed: str,
    start: str,
    end: str,
    frame: str | None = None,
) -> dict[str, Any]:
    payload = {
        "accn": accn,
        "val": value,
        "fy": fy,
        "fp": fp,
        "form": form,
        "filed": filed,
        "start": start,
        "end": end,
    }
    if frame is not None:
        payload["frame"] = frame
    return payload


def _facts_for(concept: str, unit: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "us-gaap": {
            concept: {
                "label": concept,
                "units": {
                    unit: observations,
                },
            }
        }
    }


def test_revenue_growth_event_preserves_fact_and_filing_lineage() -> None:
    record = _companyfacts_record(
        _facts_for(
            "Revenues",
            "USD",
            [
                _fact(
                    accn="0000320193-23-000100",
                    value=100,
                    fy=2023,
                    filed="2023-11-03",
                    start="2022-10-01",
                    end="2023-09-30",
                    frame="CY2023",
                ),
                _fact(
                    accn="0000320193-24-000100",
                    value=125,
                    fy=2024,
                    filed="2024-11-01",
                    start="2023-10-01",
                    end="2024-09-28",
                    frame="CY2024",
                ),
            ],
        )
    )

    events = compute_sec_filing_delta_events(record)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "revenue_growth"
    assert event.cik == "0000320193"
    assert event.accession_number == "0000320193-24-000100"
    assert event.previous_accession_number == "0000320193-23-000100"
    assert event.fact_name == "Revenues"
    assert event.taxonomy == "us-gaap"
    assert event.unit == "USD"
    assert event.period_start == date(2023, 10, 1)
    assert event.period_end == date(2024, 9, 28)
    assert event.previous_period_end == date(2023, 9, 30)
    assert event.filed_date == date(2024, 11, 1)
    assert event.available_at == datetime(2024, 11, 1, tzinfo=UTC)
    assert event.fetched_at == FETCHED_AT
    assert event.current_value == Decimal("125")
    assert event.previous_value == Decimal("100")
    assert event.absolute_delta == Decimal("25")
    assert event.relative_delta == 0.25
    assert event.source_payload_hash == "sha256:companyfacts"
    assert event.metadata["current_frame"] == "CY2024"


def test_event_identity_is_stable_across_payload_hash_changes() -> None:
    facts = _facts_for(
        "Revenues",
        "USD",
        [
            _fact(
                accn="0000320193-23-000100",
                value=100,
                fy=2023,
                filed="2023-11-03",
                start="2022-10-01",
                end="2023-09-30",
            ),
            _fact(
                accn="0000320193-24-000100",
                value=125,
                fy=2024,
                filed="2024-11-01",
                start="2023-10-01",
                end="2024-09-28",
            ),
        ],
    )
    first = _companyfacts_record(facts)
    second = _companyfacts_record(facts)
    second.payload_hash = "sha256:refetched-payload"

    first_event = compute_sec_filing_delta_events(first)[0]
    second_event = compute_sec_filing_delta_events(second)[0]

    assert first_event.event_id == second_event.event_id
    assert second_event.source_payload_hash == "sha256:refetched-payload"


def test_amended_filing_for_same_period_emits_restatement_event() -> None:
    record = _companyfacts_record(
        _facts_for(
            "Revenues",
            "USD",
            [
                _fact(
                    accn="0000320193-24-000100",
                    value=125,
                    fy=2024,
                    form="10-K",
                    filed="2024-11-01",
                    start="2023-10-01",
                    end="2024-09-28",
                ),
                _fact(
                    accn="0000320193-24-000101",
                    value=123,
                    fy=2024,
                    form="10-K/A",
                    filed="2025-01-15",
                    start="2023-10-01",
                    end="2024-09-28",
                ),
            ],
        )
    )

    events = compute_sec_filing_delta_events(record)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "restatement"
    assert event.accession_number == "0000320193-24-000101"
    assert event.previous_accession_number == "0000320193-24-000100"
    assert event.form == "10-K/A"
    assert event.filed_date == date(2025, 1, 15)
    assert event.period_end == date(2024, 9, 28)
    assert event.current_value == Decimal("123")
    assert event.previous_value == Decimal("125")
    assert event.absolute_delta == Decimal("-2")
    assert event.metadata["amended_filing"] is True


def test_missing_intermediate_period_is_marked_as_gap_without_lookahead() -> None:
    record = _companyfacts_record(
        _facts_for(
            "InventoryNet",
            "USD",
            [
                _fact(
                    accn="0000320193-22-000100",
                    value=50,
                    fy=2022,
                    filed="2022-11-01",
                    start="2021-10-01",
                    end="2022-09-24",
                ),
                _fact(
                    accn="0000320193-24-000100",
                    value=70,
                    fy=2024,
                    filed="2024-11-01",
                    start="2023-10-01",
                    end="2024-09-28",
                ),
            ],
        )
    )

    events = compute_sec_filing_delta_events(record)

    assert len(events) == 1
    assert events[0].event_type == "inventory_change"
    assert events[0].metadata["period_gap"] is True
    assert events[0].metadata["period_gap_years"] == 2
    assert events[0].available_at == datetime(2024, 11, 1, tzinfo=UTC)


def test_unit_mismatches_are_not_compared_across_units() -> None:
    facts = {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        _fact(
                            accn="0000320193-23-000100",
                            value=100,
                            fy=2023,
                            filed="2023-11-03",
                            start="2022-10-01",
                            end="2023-09-30",
                        )
                    ],
                    "EUR": [
                        _fact(
                            accn="0000320193-24-000100",
                            value=125,
                            fy=2024,
                            filed="2024-11-01",
                            start="2023-10-01",
                            end="2024-09-28",
                        )
                    ],
                },
            }
        }
    }
    record = _companyfacts_record(facts)

    events = compute_sec_filing_delta_events(record)

    assert events == []


def test_gross_margin_compression_event_uses_same_period_revenue_and_profit() -> None:
    record = _companyfacts_record(
        {
            "us-gaap": {
                "Revenues": {
                    "label": "Revenues",
                    "units": {
                        "USD": [
                            _fact(
                                accn="0000320193-23-000100",
                                value=100,
                                fy=2023,
                                filed="2023-11-03",
                                start="2022-10-01",
                                end="2023-09-30",
                            ),
                            _fact(
                                accn="0000320193-24-000100",
                                value=100,
                                fy=2024,
                                filed="2024-11-01",
                                start="2023-10-01",
                                end="2024-09-28",
                            ),
                        ]
                    },
                },
                "GrossProfit": {
                    "label": "Gross Profit",
                    "units": {
                        "USD": [
                            _fact(
                                accn="0000320193-23-000100",
                                value=45,
                                fy=2023,
                                filed="2023-11-03",
                                start="2022-10-01",
                                end="2023-09-30",
                            ),
                            _fact(
                                accn="0000320193-24-000100",
                                value=38,
                                fy=2024,
                                filed="2024-11-01",
                                start="2023-10-01",
                                end="2024-09-28",
                            ),
                        ]
                    },
                },
            }
        }
    )

    events = compute_sec_filing_delta_events(record)
    margin_events = [event for event in events if event.event_type == "margin_compression"]

    assert len(margin_events) == 1
    event = margin_events[0]
    assert event.fact_name == "GrossMargin"
    assert event.current_value == Decimal("0.38")
    assert event.previous_value == Decimal("0.45")
    assert event.absolute_delta == Decimal("-0.07")
    assert event.metadata["current_gross_profit_accession"] == "0000320193-24-000100"
    assert event.metadata["current_revenue_accession"] == "0000320193-24-000100"
