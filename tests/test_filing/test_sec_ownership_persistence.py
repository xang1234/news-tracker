"""Tests for SEC ownership event persistence."""

from __future__ import annotations

import pathlib
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.filing.sec_ownership_events import SECOwnershipEvent, SECOwnershipEventRepository

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "migrations" / "037_sec_ownership_events.sql"
)


def _event() -> SECOwnershipEvent:
    return SECOwnershipEvent(
        event_id="ownership:event",
        event_type="form4_non_derivative_transaction",
        accession_number="0001045810-26-000004",
        filing_type="4",
        filed_date=date(2026, 6, 1),
        issuer_cik="1045810",
        issuer_name="NVIDIA Corporation",
        issuer_ticker="NVDA",
        filer_cik="1999999",
        filer_name="Example Insider",
        security_title="Common Stock",
        transaction_code="S",
        transaction_date=date(2026, 5, 28),
        transaction_shares=Decimal("125"),
        transaction_price_per_share=Decimal("123.45"),
        transaction_acquired_disposed_code="D",
        shares_owned_following=Decimal("875"),
        available_at=datetime(2026, 6, 1, tzinfo=UTC),
        fetched_at=datetime(2026, 6, 1, 17, tzinfo=UTC),
        source_url="https://www.sec.gov/Archives/example.txt",
        metadata={"mapping_status": "resolved"},
    )


def _row(**overrides: Any) -> dict[str, Any]:
    event = _event()
    row = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "accession_number": event.accession_number,
        "filing_type": event.filing_type,
        "filed_date": event.filed_date,
        "issuer_cik": event.issuer_cik,
        "issuer_name": event.issuer_name,
        "issuer_ticker": event.issuer_ticker,
        "filer_cik": event.filer_cik,
        "filer_name": event.filer_name,
        "security_title": event.security_title,
        "transaction_code": event.transaction_code,
        "transaction_date": event.transaction_date,
        "transaction_shares": event.transaction_shares,
        "transaction_price_per_share": event.transaction_price_per_share,
        "transaction_acquired_disposed_code": event.transaction_acquired_disposed_code,
        "shares_owned_following": event.shares_owned_following,
        "derivative_underlying_shares": event.derivative_underlying_shares,
        "ownership_percent": event.ownership_percent,
        "position_cusip": event.position_cusip,
        "position_shares": event.position_shares,
        "position_value_usd": event.position_value_usd,
        "previous_position_shares": event.previous_position_shares,
        "position_delta_shares": event.position_delta_shares,
        "is_amendment": event.is_amendment,
        "available_at": event.available_at,
        "fetched_at": event.fetched_at,
        "source_url": event.source_url,
        "metadata": event.metadata,
        "created_at": datetime(2026, 6, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 1, tzinfo=UTC),
    }
    row.update(overrides)
    return row


class TestSECOwnershipEventRepository:
    @pytest.mark.asyncio
    async def test_upsert_event_is_idempotent_by_event_id(self) -> None:
        database = AsyncMock()
        database.fetchrow.return_value = _row()
        repository = SECOwnershipEventRepository(database)

        persisted = await repository.upsert_event(_event())

        args = database.fetchrow.call_args[0]
        sql = args[0]
        assert "ON CONFLICT (event_id) DO UPDATE SET" in sql
        assert "security_title = $11" in sql
        assert "transaction_code = $12" in sql
        assert "position_cusip = $20" in sql
        assert "available_at = $26" in sql
        assert args[1] == "ownership:event"
        assert args[6] == "0001045810"
        assert persisted.issuer_cik == "0001045810"

    @pytest.mark.asyncio
    async def test_list_events_filters_by_issuer_and_available_at(self) -> None:
        database = AsyncMock()
        database.fetch.return_value = [_row()]
        repository = SECOwnershipEventRepository(database)

        events = await repository.list_events(
            issuer_cik="1045810",
            as_of=datetime(2026, 6, 2, tzinfo=UTC),
            event_type="form4_non_derivative_transaction",
        )

        args = database.fetch.call_args[0]
        sql = args[0]
        assert "issuer_cik = $1" in sql
        assert "available_at <= $" in sql
        assert "event_type = $" in sql
        assert args[1] == "0001045810"
        assert events[0].event_id == "ownership:event"

    @pytest.mark.asyncio
    async def test_unexpected_metadata_column_type_decodes_to_empty_object(self) -> None:
        database = AsyncMock()
        database.fetch.return_value = [_row(metadata=["not", "object"])]
        repository = SECOwnershipEventRepository(database)

        events = await repository.list_events()

        assert events[0].metadata == {}

    @pytest.mark.asyncio
    async def test_malformed_metadata_column_string_decodes_to_empty_object(self) -> None:
        database = AsyncMock()
        database.fetch.return_value = [_row(metadata="{bad-json")]
        repository = SECOwnershipEventRepository(database)

        events = await repository.list_events()

        assert events[0].metadata == {}


class TestMigration037:
    @pytest.fixture()
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_ownership_event_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS sec_ownership_events" in sql

    def test_event_type_check_keeps_form4_13d_13g_13f_separate(self, sql: str) -> None:
        for event_type in (
            "form4_non_derivative_transaction",
            "form4_derivative_transaction",
            "schedule_13d_ownership",
            "schedule_13g_ownership",
            "13f_position",
        ):
            assert event_type in sql

    def test_indexes_issuer_filer_accession_and_position_paths(self, sql: str) -> None:
        assert "idx_sec_ownership_events_issuer_as_of" in sql
        assert "idx_sec_ownership_events_filer" in sql
        assert "idx_sec_ownership_events_accession" in sql
        assert "idx_sec_ownership_events_position_cusip" in sql
