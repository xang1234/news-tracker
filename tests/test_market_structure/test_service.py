"""Tests for market-structure file ingestion orchestration."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.market_structure import MarketStructureIngestionService, MarketStructureSourceFile
from src.security_master.schemas import Security


class FakeSecurityRepository:
    def __init__(self) -> None:
        self.requested_keys: list[tuple[str, str]] = []

    async def get_by_keys(self, keys):
        self.requested_keys = list(keys)
        return {
            ("NVDA", "US"): Security(
                ticker="NVDA",
                exchange="US",
                name="NVIDIA Corporation",
                sec_cik="1045810",
                issuer_name="NVIDIA Corporation",
            )
        }


class FakeMarketStructureRepository:
    def __init__(self) -> None:
        self.upserted = []

    async def upsert_events(self, events):
        self.upserted = list(events)
        return list(events)


@pytest.mark.asyncio
async def test_ingest_source_files_maps_securities_applies_signals_and_persists() -> None:
    security_repository = FakeSecurityRepository()
    event_repository = FakeMarketStructureRepository()
    service = MarketStructureIngestionService(
        security_repository=security_repository,
        event_repository=event_repository,
    )

    result = await service.ingest_source_files(
        finra_short_volume_files=[
            MarketStructureSourceFile(
                source_name="FINRA",
                source_url="file:///CNMSshvol20260601.txt",
                content="Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
                "20260601|NVDA|600|0|1000|Q\n"
                "1\n",
                fetched_at=datetime(2026, 6, 1, 23, tzinfo=UTC),
            )
        ],
        sec_fails_to_deliver_files=[
            MarketStructureSourceFile(
                source_name="SEC",
                source_url="file:///cnsfails202606a.txt",
                content="SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
                "20260601|67066G104|NVDA|12000|NVIDIA CORP|123.45\n",
                fetched_at=datetime(2026, 6, 15, tzinfo=UTC),
            )
        ],
    )

    assert ("NVDA", "US") in security_repository.requested_keys
    assert result.total_events == 2
    assert result.finra_short_volume_count == 1
    assert result.sec_fails_to_deliver_count == 1
    assert result.upserted_count == 2
    assert result.unresolved_symbol_count == 0
    assert [event.issuer_cik for event in event_repository.upserted] == [
        "0001045810",
        "0001045810",
    ]
    assert event_repository.upserted[0].signal_type == "short_volume_ratio"
    assert event_repository.upserted[1].signal_type == "fails_to_deliver_notional"


@pytest.mark.asyncio
async def test_ingest_source_files_counts_unresolved_symbols() -> None:
    security_repository = FakeSecurityRepository()
    event_repository = FakeMarketStructureRepository()
    service = MarketStructureIngestionService(
        security_repository=security_repository,
        event_repository=event_repository,
    )

    result = await service.ingest_source_files(
        finra_short_volume_files=[
            MarketStructureSourceFile(
                source_name="FINRA",
                source_url="file:///CNMSshvol20260601.txt",
                content="Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
                "20260601|MISSING|600|0|1000|Q\n"
                "1\n",
            )
        ],
    )

    assert result.unresolved_symbol_count == 1
    assert event_repository.upserted[0].metadata["mapping_status"] == "unresolved"
