"""Tests for SEC filing-delta persistence."""

from __future__ import annotations

import pathlib
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.filing.sec_delta_events import (
    SECFilingDeltaEvent,
    SECFilingDeltaRepository,
)

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "migrations" / "036_sec_filing_delta_events.sql"
)


def _event() -> SECFilingDeltaEvent:
    return SECFilingDeltaEvent(
        event_id="sec-delta-1",
        cik="320193",
        event_type="revenue_growth",
        accession_number="0000320193-24-000100",
        previous_accession_number="0000320193-23-000100",
        taxonomy="us-gaap",
        fact_name="Revenues",
        unit="USD",
        period_start=date(2023, 10, 1),
        period_end=date(2024, 9, 28),
        previous_period_start=date(2022, 10, 1),
        previous_period_end=date(2023, 9, 30),
        filed_date=date(2024, 11, 1),
        previous_filed_date=date(2023, 11, 3),
        form="10-K",
        previous_form="10-K",
        available_at=datetime(2024, 11, 1, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 31, tzinfo=UTC),
        current_value=Decimal("125"),
        previous_value=Decimal("100"),
        absolute_delta=Decimal("25"),
        relative_delta=0.25,
        source_payload_hash="sha256:companyfacts",
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        metadata={"current_frame": "CY2024"},
    )


def _event_row(**overrides: Any) -> dict[str, Any]:
    event = _event()
    row = {
        "event_id": event.event_id,
        "cik": event.cik,
        "event_type": event.event_type,
        "accession_number": event.accession_number,
        "previous_accession_number": event.previous_accession_number,
        "taxonomy": event.taxonomy,
        "fact_name": event.fact_name,
        "unit": event.unit,
        "period_start": event.period_start,
        "period_end": event.period_end,
        "previous_period_start": event.previous_period_start,
        "previous_period_end": event.previous_period_end,
        "filed_date": event.filed_date,
        "previous_filed_date": event.previous_filed_date,
        "form": event.form,
        "previous_form": event.previous_form,
        "available_at": event.available_at,
        "fetched_at": event.fetched_at,
        "current_value": event.current_value,
        "previous_value": event.previous_value,
        "absolute_delta": event.absolute_delta,
        "relative_delta": event.relative_delta,
        "source_payload_hash": event.source_payload_hash,
        "source_url": event.source_url,
        "metadata": event.metadata,
        "created_at": datetime(2026, 5, 31, tzinfo=UTC),
        "updated_at": datetime(2026, 5, 31, tzinfo=UTC),
    }
    row.update(overrides)
    return row


class TestSECFilingDeltaEvent:
    def test_normalizes_cik(self) -> None:
        assert _event().cik == "0000320193"

    def test_to_payload_preserves_point_in_time_fields(self) -> None:
        payload = _event().to_payload()

        assert payload["accession_number"] == "0000320193-24-000100"
        assert payload["available_at"] == "2024-11-01T00:00:00+00:00"
        assert payload["fetched_at"] == "2026-05-31T00:00:00+00:00"
        assert payload["source_payload_hash"] == "sha256:companyfacts"


class TestSECFilingDeltaRepository:
    @pytest.mark.asyncio
    async def test_upsert_event_is_idempotent_by_event_id(self) -> None:
        database = AsyncMock()
        database.fetchrow.return_value = _event_row()
        repository = SECFilingDeltaRepository(database)

        persisted = await repository.upsert_event(_event())

        args = database.fetchrow.call_args[0]
        sql = args[0]
        assert "ON CONFLICT (event_id) DO UPDATE SET" in sql
        assert args[1] == "sec-delta-1"
        assert args[2] == "0000320193"
        assert args[18] == datetime(2026, 5, 31, tzinfo=UTC)
        assert persisted.cik == "0000320193"

    @pytest.mark.asyncio
    async def test_list_events_as_of_filters_by_available_at(self) -> None:
        database = AsyncMock()
        database.fetch.return_value = [_event_row()]
        repository = SECFilingDeltaRepository(database)

        events = await repository.list_events_as_of(
            "320193",
            as_of=datetime(2024, 12, 1, tzinfo=UTC),
            event_type="revenue_growth",
        )

        args = database.fetch.call_args[0]
        sql = args[0]
        assert "available_at <= $" in sql
        assert "event_type = $" in sql
        assert "ORDER BY available_at DESC, filed_date DESC" in sql
        assert args[1] == "0000320193"
        assert len(events) == 1
        assert events[0].event_id == "sec-delta-1"


class TestMigration036:
    @pytest.fixture()
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_delta_event_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS sec_filing_delta_events" in sql

    def test_preserves_required_lineage_columns(self, sql: str) -> None:
        for column in (
            "accession_number",
            "fact_name",
            "unit",
            "period_start",
            "period_end",
            "filed_date",
            "fetched_at",
            "source_payload_hash",
        ):
            assert column in sql

    def test_as_of_index_exists(self, sql: str) -> None:
        assert "idx_sec_filing_delta_events_as_of" in sql
        assert "available_at DESC" in sql

    def test_event_type_constraint_includes_supported_deltas(self, sql: str) -> None:
        for event_type in (
            "revenue_growth",
            "inventory_change",
            "capex_change",
            "rnd_change",
            "margin_compression",
            "restatement",
        ):
            assert event_type in sql
