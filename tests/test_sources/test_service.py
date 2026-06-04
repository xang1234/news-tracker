"""Tests for SourcesService."""

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.config.feeds import FEEDS, Feed
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
    async def test_refetches_after_ttl(self, service: SourcesService, sample_db_row: dict) -> None:
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
    async def test_returns_tuples(self, service: SourcesService, sample_substack_row: dict) -> None:
        service.repository._db.fetch.return_value = [sample_substack_row]

        result = await service.get_substack_sources()

        assert result == [("semianalysis", "SemiAnalysis", "Semiconductor deep dives")]


class TestGetRssFeeds:
    """Tests for cached RSS feed retrieval."""

    @pytest.mark.asyncio
    async def test_returns_feed_records(self, service: SourcesService) -> None:
        service.repository._db.fetch.return_value = [
            {
                "platform": "rss",
                "identifier": "nvidia-press-releases",
                "display_name": "NVIDIA Newsroom Press Releases",
                "description": "Official NVIDIA press releases",
                "is_active": True,
                "metadata": {
                    "url": "https://nvidianews.nvidia.com/cats/press_release.xml",
                    "category": "company_ir",
                    "authority": "official",
                    "full_text": True,
                },
                "created_at": None,
                "updated_at": None,
            }
        ]

        result = await service.get_rss_feeds()

        assert result == [
            Feed(
                slug="nvidia-press-releases",
                name="NVIDIA Newsroom Press Releases",
                url="https://nvidianews.nvidia.com/cats/press_release.xml",
                category="company_ir",
                authority="official",
                full_text=True,
                enabled=True,
            )
        ]

    @pytest.mark.asyncio
    async def test_returns_cached_on_second_call(self, service: SourcesService) -> None:
        service.repository._db.fetch.return_value = [
            {
                "platform": "rss",
                "identifier": "semiwiki",
                "display_name": "SemiWiki",
                "description": "Semiconductor trade analysis",
                "is_active": True,
                "metadata": {
                    "url": "https://semiwiki.com/feed/",
                    "category": "trade_press",
                    "authority": "specialist",
                    "full_text": True,
                },
                "created_at": None,
                "updated_at": None,
            }
        ]

        first = await service.get_rss_feeds()
        service.repository._db.fetch.return_value = []
        second = await service.get_rss_feeds()

        assert first == second
        assert second[0].slug == "semiwiki"


class TestRssSourceHealth:
    """Tests for RSS source health summaries and persistence."""

    @pytest.mark.asyncio
    async def test_returns_operator_health_statuses(self, service: SourcesService) -> None:
        now = datetime(2026, 6, 1, 12, tzinfo=UTC)
        stale_fetch_at = (now - timedelta(days=2)).isoformat()
        service.repository._db.fetch.return_value = [
            {
                "platform": "rss",
                "identifier": "healthy-feed",
                "display_name": "Healthy Feed",
                "description": "",
                "is_active": True,
                "metadata": {
                    "url": "https://example.com/healthy.xml",
                    "category": "trade_press",
                    "health": {
                        "status": "ok",
                        "last_fetch_at": now.isoformat(),
                        "last_success_at": now.isoformat(),
                        "recent_document_count": 3,
                    },
                },
                "created_at": None,
                "updated_at": None,
            },
            {
                "platform": "rss",
                "identifier": "stale-feed",
                "display_name": "Stale Feed",
                "description": "",
                "is_active": True,
                "metadata": {
                    "url": "https://example.com/stale.xml",
                    "category": "trade_press",
                    "health": {
                        "status": "ok",
                        "last_fetch_at": stale_fetch_at,
                        "recent_document_count": 0,
                    },
                },
                "created_at": None,
                "updated_at": None,
            },
            {
                "platform": "rss",
                "identifier": "failing-feed",
                "display_name": "Failing Feed",
                "description": "",
                "is_active": True,
                "metadata": {
                    "url": "https://example.com/failing.xml",
                    "category": "trade_press",
                    "health": {
                        "status": "error",
                        "last_fetch_at": now.isoformat(),
                        "last_error": "HTTP 500",
                        "recent_document_count": 0,
                    },
                },
                "created_at": None,
                "updated_at": None,
            },
            {
                "platform": "rss",
                "identifier": "inactive-feed",
                "display_name": "Inactive Feed",
                "description": "",
                "is_active": False,
                "metadata": {
                    "url": "https://example.com/inactive.xml",
                    "category": "trade_press",
                },
                "created_at": None,
                "updated_at": None,
            },
        ]

        result = await service.get_rss_source_health(now=now)
        by_slug = {item.slug: item for item in result}

        assert by_slug["healthy-feed"].status == "active"
        assert by_slug["healthy-feed"].is_producing is True
        assert by_slug["healthy-feed"].recent_document_count == 3
        assert by_slug["stale-feed"].status == "stale"
        assert by_slug["failing-feed"].status == "failing"
        assert by_slug["failing-feed"].last_error == "HTTP 500"
        assert by_slug["inactive-feed"].status == "inactive"

    @pytest.mark.asyncio
    async def test_records_rss_feed_health(self, service: SourcesService) -> None:
        await service.record_rss_feed_health(
            "healthy-feed",
            {
                "status": "ok",
                "last_fetch_at": "2026-06-01T12:00:00+00:00",
                "recent_document_count": 2,
            },
        )

        service.repository._db.execute.assert_called_once()
        args = service.repository._db.execute.call_args[0]
        assert "metadata" in args[0]
        assert args[1] == "rss"
        assert args[2] == "healthy-feed"
        assert json.loads(args[3]) == {
            "health": {
                "status": "ok",
                "last_fetch_at": "2026-06-01T12:00:00+00:00",
                "recent_document_count": 2,
            }
        }


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
        service._rss_cache = [
            Feed(
                slug="semiwiki",
                name="SemiWiki",
                url="https://semiwiki.com/feed/",
                category="trade_press",
            )
        ]

        service.invalidate_cache()

        assert service._twitter_cache is None
        assert service._reddit_cache is None
        assert service._substack_cache is None
        assert service._rss_cache is None


class TestSeedFromJson:
    """Tests for JSON seed loading."""

    @pytest.mark.asyncio
    async def test_loads_and_upserts(self, service: SourcesService, tmp_path: Path) -> None:
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
            json.dumps([{"platform": "twitter", "identifier": "NEW", "display_name": "New"}])
        )

        await service.seed_from_json(seed_file)

        assert service._twitter_cache is None

    @pytest.mark.asyncio
    async def test_default_seed_includes_static_rss_catalog(self, service: SourcesService) -> None:
        seed_count = len(json.loads(Path("src/sources/data/seed_sources.json").read_text()))

        result = await service.seed_from_json()

        assert result == seed_count + len(FEEDS)
        args = service.repository._db.execute.call_args[0]
        assert "rss" in args[1]
        assert "nvidia-press-releases" in args[2]


class TestEnsureSeeded:
    """Tests for auto-seed on init."""

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, mock_database: AsyncMock) -> None:
        config = SourcesConfig(seed_on_init=False)
        svc = SourcesService(mock_database, config)

        await svc.ensure_seeded()

        mock_database.fetchval.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_table_has_data(self, service: SourcesService) -> None:
        service.repository._db.fetchval.return_value = 32

        await service.ensure_seeded()

        # Should only have called count, not execute (bulk_upsert)
        service.repository._db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_seeds_when_table_empty(self, service: SourcesService) -> None:
        service.repository._db.fetchval.return_value = 0

        with patch.object(service, "seed_from_json", new_callable=AsyncMock) as mock_seed:
            mock_seed.return_value = 32
            await service.ensure_seeded()
            mock_seed.assert_called_once()
