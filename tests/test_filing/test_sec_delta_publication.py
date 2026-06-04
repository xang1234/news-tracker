"""Tests for publishing SEC filing-delta events as official fact evidence."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from src.filing.sec_delta_events import (
    SECFilingDeltaEvent,
    build_sec_delta_payload,
    build_sec_fact_evidence,
)

NOW = datetime(2026, 5, 31, tzinfo=UTC)


def _event(
    *,
    event_type: str = "revenue_growth",
    current_value: Decimal = Decimal("120"),
    previous_value: Decimal = Decimal("100"),
    absolute_delta: Decimal = Decimal("20"),
    relative_delta: float = 0.2,
) -> SECFilingDeltaEvent:
    return SECFilingDeltaEvent(
        event_id="sec_delta:test",
        cik="320193",
        event_type=event_type,
        accession_number="0000320193-26-000001",
        previous_accession_number="0000320193-25-000001",
        taxonomy="us-gaap",
        fact_name="Revenues",
        unit="USD",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        previous_period_start=date(2024, 1, 1),
        previous_period_end=date(2024, 12, 31),
        filed_date=date(2026, 1, 31),
        previous_filed_date=date(2025, 1, 31),
        form="10-K",
        previous_form="10-K",
        available_at=NOW,
        fetched_at=NOW,
        current_value=current_value,
        previous_value=previous_value,
        absolute_delta=absolute_delta,
        relative_delta=relative_delta,
        source_payload_hash="sha256:payload",
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        metadata={"period_gap": False},
    )


def test_delta_payload_has_reason_code_and_lineage() -> None:
    payload = build_sec_delta_payload(_event())
    data = payload.to_dict()

    assert data["reason_code"] == "sec_fact_revenue_growth"
    assert data["object_type"] == "filing_fact"
    assert data["event"]["event_id"] == "sec_delta:test"
    assert data["lineage"]["accession_number"] == "0000320193-26-000001"
    assert data["lineage"]["previous_accession_number"] == "0000320193-25-000001"
    assert data["lineage"]["source_payload_hash"] == "sha256:payload"
    assert data["lineage"]["available_at"] == NOW.isoformat()


def test_sec_fact_evidence_marks_positive_growth_as_corroborating() -> None:
    evidence = build_sec_fact_evidence(_event())

    assert evidence.evidence_role == "corroborating"
    assert evidence.reason_code == "sec_fact_revenue_growth"
    assert evidence.to_dict()["lineage"]["fact_name"] == "Revenues"


def test_sec_fact_evidence_marks_negative_growth_as_contradictory() -> None:
    evidence = build_sec_fact_evidence(
        _event(
            current_value=Decimal("80"),
            previous_value=Decimal("100"),
            absolute_delta=Decimal("-20"),
            relative_delta=-0.2,
        )
    )

    assert evidence.evidence_role == "contradictory"


def test_sec_fact_evidence_marks_restatements_as_contradictory() -> None:
    evidence = build_sec_fact_evidence(
        _event(
            event_type="restatement",
            current_value=Decimal("95"),
            previous_value=Decimal("100"),
            absolute_delta=Decimal("-5"),
            relative_delta=-0.05,
        )
    )

    assert evidence.reason_code == "sec_fact_restatement"
    assert evidence.evidence_role == "contradictory"
