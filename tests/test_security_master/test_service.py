"""Tests for SecurityMasterService."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.security_master.config import SecurityMasterConfig
from src.security_master.service import SecurityMasterService


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
            {"ticker": "NVDA"}, {"ticker": "AMD"},
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
    async def test_delegates_to_repository(self, service: SecurityMasterService, sample_db_row: dict) -> None:
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
            {"ticker": "TEST", "name": "Test Corp", "aliases": ["test"]},
        ]
        seed_file = tmp_path / "test_seed.json"
        seed_file.write_text(json.dumps(seed_data))

        result = await service.seed_from_json(seed_file)

        assert result == 1  # bulk_upsert returns len(securities)
        service.repository._db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidates_cache_after_seed(self, service: SecurityMasterService, tmp_path: Path) -> None:
        # Prime cache
        service._tickers_cache = {"OLD"}
        service._tickers_cached_at = time.monotonic()

        seed_file = tmp_path / "test_seed.json"
        seed_file.write_text(json.dumps([{"ticker": "NEW", "name": "New"}]))

        await service.seed_from_json(seed_file)

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
