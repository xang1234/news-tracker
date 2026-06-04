"""Tests for SecurityMasterService."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.security_master.config import SecurityMasterConfig
from src.security_master.nasdaq_trader import NASDAQ_TRADER_EXTERNAL_KEY
from src.security_master.service import SecurityMasterService

NASDAQ_HEADER = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
)
OTHER_HEADER = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol"
)
NASDAQ_LISTED_TEXT = f"""{NASDAQ_HEADER}
NVDA|NVIDIA Corporation|Q|N|N|100|N|N
File Creation Time: 0601202616:01|||||||
"""

OTHER_LISTED_TEXT = f"""{OTHER_HEADER}
IBM|International Business Machines Corporation|N|IBM|N|100|N|IBM
File Creation Time: 0601202616:02|||||||
"""


@pytest.fixture
def config() -> SecurityMasterConfig:
    return SecurityMasterConfig(cache_ttl_seconds=60, fuzzy_threshold=0.3)


@pytest.fixture
def service(mock_database: AsyncMock, config: SecurityMasterConfig) -> SecurityMasterService:
    return SecurityMasterService(mock_database, config)


class TestGetAllTickers:
    """Tests for cached ticker retrieval."""

    @pytest.mark.asyncio
    async def test_fetches_from_db_on_first_call(self, service: SecurityMasterService) -> None:
        service.repository._db.fetch.return_value = [
            {"ticker": "NVDA"},
            {"ticker": "AMD"},
        ]

        result = await service.get_all_tickers()

        assert result == {"NVDA", "AMD"}

    @pytest.mark.asyncio
    async def test_returns_cached_on_second_call(self, service: SecurityMasterService) -> None:
        service.repository._db.fetch.return_value = [{"ticker": "NVDA"}]

        first = await service.get_all_tickers()
        # Change DB return — should NOT be used
        service.repository._db.fetch.return_value = [{"ticker": "AMD"}]
        second = await service.get_all_tickers()

        assert first == second == {"NVDA"}

    @pytest.mark.asyncio
    async def test_refetches_after_ttl(self, service: SecurityMasterService) -> None:
        service.repository._db.fetch.return_value = [{"ticker": "NVDA"}]
        await service.get_all_tickers()

        # Simulate TTL expiry
        service._tickers_cached_at = time.monotonic() - 120

        service.repository._db.fetch.return_value = [{"ticker": "AMD"}]
        result = await service.get_all_tickers()

        assert result == {"AMD"}

    @pytest.mark.asyncio
    async def test_invalidate_forces_refetch(self, service: SecurityMasterService) -> None:
        service.repository._db.fetch.return_value = [{"ticker": "NVDA"}]
        await service.get_all_tickers()

        service.invalidate_cache()
        service.repository._db.fetch.return_value = [{"ticker": "TSM"}]
        result = await service.get_all_tickers()

        assert result == {"TSM"}


class TestGetCompanyMap:
    """Tests for cached company mapping."""

    @pytest.mark.asyncio
    async def test_fetches_from_db(self, service: SecurityMasterService) -> None:
        service.repository._db.fetch.return_value = [
            {"ticker": "NVDA", "name": "NVIDIA", "aliases": ["nvidia"]},
        ]

        result = await service.get_company_map()

        assert result["nvidia"] == "NVDA"

    @pytest.mark.asyncio
    async def test_caches_result(self, service: SecurityMasterService) -> None:
        service.repository._db.fetch.return_value = [
            {"ticker": "NVDA", "name": "NVIDIA", "aliases": []},
        ]
        await service.get_company_map()

        service.repository._db.fetch.return_value = []
        result = await service.get_company_map()

        assert "nvidia" in result


class TestFuzzySearch:
    """Tests for fuzzy search passthrough."""

    @pytest.mark.asyncio
    async def test_delegates_to_repository(
        self, service: SecurityMasterService, sample_db_row: dict
    ) -> None:
        sample_db_row["sim"] = 0.75
        service.repository._db.fetch.return_value = [sample_db_row]

        results = await service.fuzzy_search("nvidia", limit=5)

        assert len(results) == 1
        assert results[0].ticker == "NVDA"


class TestSeedFromJson:
    """Tests for JSON seed loading."""

    @pytest.mark.asyncio
    async def test_loads_and_upserts(self, service: SecurityMasterService, tmp_path: Path) -> None:
        seed_data = [
            {
                "ticker": "TEST",
                "name": "Test Corp",
                "aliases": ["test"],
                "sec_cik": "CIK1234567",
                "issuer_name": "Test Corporation",
                "former_names": ["Old Test Corp"],
                "external_identifiers": {"sec_ticker": "TEST"},
                "identifier_lineage": [
                    {
                        "identifier_type": "sec_cik",
                        "value": "CIK1234567",
                        "source": "sec_ticker_company",
                    }
                ],
            },
        ]
        seed_file = tmp_path / "test_seed.json"
        seed_file.write_text(json.dumps(seed_data))

        result = await service.seed_from_json(seed_file)

        assert result == 1  # bulk_upsert returns len(securities)
        service.repository._db.execute.assert_called_once()
        args = service.repository._db.execute.call_args[0]
        assert args[9] == ["0001234567"]
        assert args[10] == ["Test Corporation"]
        lineage = json.loads(args[13][0])
        assert lineage[0]["value"] == "0001234567"

    @pytest.mark.asyncio
    async def test_invalidates_cache_after_seed(
        self, service: SecurityMasterService, tmp_path: Path
    ) -> None:
        # Prime cache
        service._tickers_cache = {"OLD"}
        service._tickers_cached_at = time.monotonic()

        seed_file = tmp_path / "test_seed.json"
        seed_file.write_text(json.dumps([{"ticker": "NEW", "name": "New"}]))

        await service.seed_from_json(seed_file)

        assert service._tickers_cache is None


class TestNasdaqTraderIngestion:
    """Tests for Nasdaq Trader symbol-directory ingestion."""

    @pytest.mark.asyncio
    async def test_reconciles_files_through_repository_and_invalidates_cache(
        self,
        service: SecurityMasterService,
    ) -> None:
        service.repository.get_by_keys = AsyncMock(return_value={})  # type: ignore[method-assign]
        service.repository.list_by_external_identifier = AsyncMock(  # type: ignore[method-assign]
            return_value=[],
        )
        service.repository.bulk_upsert = AsyncMock(return_value=2)  # type: ignore[method-assign]
        service._tickers_cache = {"OLD"}
        service._tickers_cached_at = time.monotonic()

        result = await service.ingest_nasdaq_trader_symbol_directory(
            NASDAQ_LISTED_TEXT,
            OTHER_LISTED_TEXT,
            observed_at=datetime(2026, 6, 1, 17, tzinfo=UTC),
        )

        service.repository.get_by_keys.assert_awaited_once()
        service.repository.list_by_external_identifier.assert_awaited_once_with(
            NASDAQ_TRADER_EXTERNAL_KEY,
        )
        service.repository.bulk_upsert.assert_awaited_once()
        upserted = service.repository.bulk_upsert.call_args[0][0]
        assert {security.ticker for security in upserted} == {"NVDA", "IBM"}
        nvda = next(security for security in upserted if security.ticker == "NVDA")
        assert nvda.external_identifiers[NASDAQ_TRADER_EXTERNAL_KEY]["last_seen_at"] == (
            "2026-06-01T17:00:00+00:00"
        )
        assert result.current_record_count == 2
        assert service._tickers_cache is None


class TestEnsureSeeded:
    """Tests for auto-seed on init."""

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, mock_database: AsyncMock) -> None:
        config = SecurityMasterConfig(seed_on_init=False)
        svc = SecurityMasterService(mock_database, config)

        await svc.ensure_seeded()

        mock_database.fetchval.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_table_has_data(self, service: SecurityMasterService) -> None:
        service.repository._db.fetchval.return_value = 35

        await service.ensure_seeded()

        # Should only have called count, not execute (bulk_upsert)
        service.repository._db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_seeds_when_table_empty(self, service: SecurityMasterService) -> None:
        service.repository._db.fetchval.return_value = 0

        with patch.object(service, "seed_from_json", new_callable=AsyncMock) as mock_seed:
            mock_seed.return_value = 35
            await service.ensure_seeded()
            mock_seed.assert_called_once()
