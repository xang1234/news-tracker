"""Tests for SourcesService."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.sources.config import SourcesConfig
from src.sources.service import SourcesService


@pytest.fixture
def config() -> SourcesConfig:
    return SourcesConfig(cache_ttl_seconds=60)


@pytest.fixture
def service(mock_database: AsyncMock, config: SourcesConfig) -> SourcesService:
    return SourcesService(mock_database, config)


class TestGetTwitterSources:
    """Tests for cached Twitter source retrieval."""

    @pytest.mark.asyncio
    async def test_fetches_from_db_on_first_call(
        self, service: SourcesService, sample_db_row: dict
    ) -> None:
        service.repository._db.fetch.return_value = [sample_db_row]

        result = await service.get_twitter_sources()

        assert result == ["SemiAnalysis"]

    @pytest.mark.asyncio
    async def test_returns_cached_on_second_call(
        self, service: SourcesService, sample_db_row: dict
    ) -> None:
        service.repository._db.fetch.return_value = [sample_db_row]

        first = await service.get_twitter_sources()
        # Change DB return — should NOT be used
        service.repository._db.fetch.return_value = []
        second = await service.get_twitter_sources()

        assert first == second == ["SemiAnalysis"]

    @pytest.mark.asyncio
    async def test_refetches_after_ttl(
        self, service: SourcesService, sample_db_row: dict
    ) -> None:
        service.repository._db.fetch.return_value = [sample_db_row]
        await service.get_twitter_sources()

        # Simulate TTL expiry
        service._twitter_cached_at = time.monotonic() - 120

        sample_db_row["identifier"] = "DeItaone"
        service.repository._db.fetch.return_value = [sample_db_row]
        result = await service.get_twitter_sources()

        assert result == ["DeItaone"]


class TestGetRedditSources:
    """Tests for cached Reddit source retrieval."""

    @pytest.mark.asyncio
    async def test_returns_identifiers(
        self, service: SourcesService, sample_reddit_row: dict
    ) -> None:
        service.repository._db.fetch.return_value = [sample_reddit_row]

        result = await service.get_reddit_sources()

        assert result == ["wallstreetbets"]


class TestGetSubstackSources:
    """Tests for cached Substack source retrieval."""

    @pytest.mark.asyncio
    async def test_returns_tuples(
        self, service: SourcesService, sample_substack_row: dict
    ) -> None:
        service.repository._db.fetch.return_value = [sample_substack_row]

        result = await service.get_substack_sources()

        assert result == [("semianalysis", "SemiAnalysis", "Semiconductor deep dives")]


class TestInvalidateCache:
    """Tests for cache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_forces_refetch(
        self, service: SourcesService, sample_db_row: dict
    ) -> None:
        service.repository._db.fetch.return_value = [sample_db_row]
        await service.get_twitter_sources()

        service.invalidate_cache()

        sample_db_row["identifier"] = "nvidia"
        service.repository._db.fetch.return_value = [sample_db_row]
        result = await service.get_twitter_sources()

        assert result == ["nvidia"]

    def test_clears_all_caches(self, service: SourcesService) -> None:
        service._twitter_cache = ["a"]
        service._reddit_cache = ["b"]
        service._substack_cache = [("c", "d", "e")]

        service.invalidate_cache()

        assert service._twitter_cache is None
        assert service._reddit_cache is None
        assert service._substack_cache is None


class TestSeedFromJson:
    """Tests for JSON seed loading."""

    @pytest.mark.asyncio
    async def test_loads_and_upserts(
        self, service: SourcesService, tmp_path: Path
    ) -> None:
        seed_data = [
            {
                "platform": "twitter",
                "identifier": "TestUser",
                "display_name": "Test User",
                "description": "Testing",
            },
        ]
        seed_file = tmp_path / "test_seed.json"
        seed_file.write_text(json.dumps(seed_data))

        result = await service.seed_from_json(seed_file)

        assert result == 1
        service.repository._db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidates_cache_after_seed(
        self, service: SourcesService, tmp_path: Path
    ) -> None:
        # Prime cache
        service._twitter_cache = ["OLD"]
        service._twitter_cached_at = time.monotonic()

        seed_file = tmp_path / "test_seed.json"
        seed_file.write_text(
            json.dumps(
                [{"platform": "twitter", "identifier": "NEW", "display_name": "New"}]
            )
        )

        await service.seed_from_json(seed_file)

        assert service._twitter_cache is None


class TestEnsureSeeded:
    """Tests for auto-seed on init."""

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, mock_database: AsyncMock) -> None:
        config = SourcesConfig(seed_on_init=False)
        svc = SourcesService(mock_database, config)

        await svc.ensure_seeded()

        mock_database.fetchval.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_table_has_data(
        self, service: SourcesService
    ) -> None:
        service.repository._db.fetchval.return_value = 32

        await service.ensure_seeded()

        # Should only have called count, not execute (bulk_upsert)
        service.repository._db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_seeds_when_table_empty(
        self, service: SourcesService
    ) -> None:
        service.repository._db.fetchval.return_value = 0

        with patch.object(
            service, "seed_from_json", new_callable=AsyncMock
        ) as mock_seed:
            mock_seed.return_value = 32
            await service.ensure_seeded()
            mock_seed.assert_called_once()
