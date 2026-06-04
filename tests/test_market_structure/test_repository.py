"""Tests for market-structure event persistence."""

from __future__ import annotations

import pathlib
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.market_structure import MarketStructureEvent, MarketStructureEventRepository

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "migrations" / "038_market_structure_events.sql"
)


def _event() -> MarketStructureEvent:
    return MarketStructureEvent(
        event_id="market-structure:event",
        event_type="finra_short_volume",
        source_name="FINRA CNMS short volume",
        source_url="https://example.test/CNMSshvol20260601.txt",
        source_date=date(2026, 6, 1),
        trade_date=date(2026, 6, 1),
        symbol="NVDA",
        security_ticker="NVDA",
        security_exchange="US",
        issuer_cik="1045810",
        issuer_name="NVIDIA Corporation",
        market_code="Q",
        market_name="NASDAQ TRF Carteret",
        short_volume=600,
        short_exempt_volume=5,
        total_volume=1000,
        short_volume_ratio=Decimal("0.6"),
        signal_type="short_volume_ratio",
        anomaly_level="elevated",
        persistence_count=2,
        available_at=datetime(2026, 6, 1, tzinfo=UTC),
        fetched_at=datetime(2026, 6, 1, 23, tzinfo=UTC),
        metadata={"mapping_status": "resolved"},
    )


def _row(**overrides: Any) -> dict[str, Any]:
    event = _event()
    row = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "source_name": event.source_name,
        "source_url": event.source_url,
        "source_date": event.source_date,
        "trade_date": event.trade_date,
        "settlement_date": event.settlement_date,
        "symbol": event.symbol,
        "security_ticker": event.security_ticker,
        "security_exchange": event.security_exchange,
        "issuer_cik": event.issuer_cik,
        "issuer_name": event.issuer_name,
        "cusip": event.cusip,
        "market_code": event.market_code,
        "market_name": event.market_name,
        "short_volume": event.short_volume,
        "short_exempt_volume": event.short_exempt_volume,
        "total_volume": event.total_volume,
        "short_volume_ratio": event.short_volume_ratio,
        "short_exempt_ratio": event.short_exempt_ratio,
        "fail_quantity": event.fail_quantity,
        "fail_price": event.fail_price,
        "fail_notional": event.fail_notional,
        "signal_type": event.signal_type,
        "anomaly_level": event.anomaly_level,
        "persistence_count": event.persistence_count,
        "available_at": event.available_at,
        "fetched_at": event.fetched_at,
        "metadata": event.metadata,
        "created_at": datetime(2026, 6, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 1, tzinfo=UTC),
    }
    row.update(overrides)
    return row


class TestMarketStructureEventRepository:
    @pytest.mark.asyncio
    async def test_upsert_event_is_idempotent_by_event_id(self) -> None:
        database = AsyncMock()
        database.fetchrow.return_value = _row()
        repository = MarketStructureEventRepository(database)

        persisted = await repository.upsert_event(_event())

        args = database.fetchrow.call_args[0]
        sql = args[0]
        assert "ON CONFLICT (event_id) DO UPDATE SET" in sql
        assert "short_volume_ratio = $19" in sql
        assert "fail_notional = $23" in sql
        assert args[1] == "market-structure:event"
        assert args[11] == "0001045810"
        assert persisted.issuer_cik == "0001045810"

    @pytest.mark.asyncio
    async def test_list_events_filters_by_symbol_source_and_available_at(self) -> None:
        database = AsyncMock()
        database.fetch.return_value = [_row()]
        repository = MarketStructureEventRepository(database)

        events = await repository.list_events(
            symbol="nvda",
            as_of=datetime(2026, 6, 2, tzinfo=UTC),
            event_type="finra_short_volume",
        )

        args = database.fetch.call_args[0]
        sql = args[0]
        assert "symbol = $1" in sql
        assert "available_at <= $" in sql
        assert "event_type = $" in sql
        assert args[1] == "NVDA"
        assert events[0].event_id == "market-structure:event"

    @pytest.mark.asyncio
    async def test_unexpected_metadata_column_type_decodes_to_empty_object(self) -> None:
        database = AsyncMock()
        database.fetch.return_value = [_row(metadata=["not", "object"])]
        repository = MarketStructureEventRepository(database)

        events = await repository.list_events()

        assert events[0].metadata == {}


class TestMigration038:
    @pytest.fixture()
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_market_structure_event_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS market_structure_events" in sql

    def test_event_type_check_keeps_short_volume_and_ftd_separate(self, sql: str) -> None:
        assert "finra_short_volume" in sql
        assert "sec_fail_to_deliver" in sql

    def test_indexes_symbol_cusip_and_point_in_time_paths(self, sql: str) -> None:
        assert "idx_market_structure_events_symbol_as_of" in sql
        assert "idx_market_structure_events_cusip_as_of" in sql
        assert "idx_market_structure_events_event_type_as_of" in sql
