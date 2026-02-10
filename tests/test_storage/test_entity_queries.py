"""Tests for entity-related repository methods.

These test the SQL query construction and parameter handling of
DocumentRepository entity methods using a mocked asyncpg database.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.storage.repository import DocumentRepository


@pytest.fixture
def mock_db():
    """Mock Database with asyncpg-like interface."""
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock(return_value=0)
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="UPDATE 0")
    return db


@pytest.fixture
def repo(mock_db):
    """DocumentRepository with mocked database."""
    return DocumentRepository(mock_db)


# ── list_entities ────────────────────────────────


class TestListEntities:
    """Tests for list_entities method."""

    @pytest.mark.asyncio
    async def test_empty_result(self, repo, mock_db):
        mock_db.fetch.return_value = []
        mock_db.fetchval.return_value = 0

        entities, total = await repo.list_entities()
        assert entities == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_returns_entities(self, repo, mock_db):
        mock_db.fetchval.return_value = 2
        mock_db.fetch.return_value = [
            {
                "type": "COMPANY",
                "normalized": "NVIDIA",
                "mention_count": 42,
                "first_seen": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "last_seen": datetime(2026, 2, 5, tzinfo=timezone.utc),
            },
            {
                "type": "TICKER",
                "normalized": "NVDA",
                "mention_count": 30,
                "first_seen": datetime(2026, 1, 5, tzinfo=timezone.utc),
                "last_seen": datetime(2026, 2, 4, tzinfo=timezone.utc),
            },
        ]

        entities, total = await repo.list_entities(limit=50, offset=0)
        assert total == 2
        assert len(entities) == 2
        assert entities[0]["normalized"] == "NVIDIA"

    @pytest.mark.asyncio
    async def test_with_type_filter(self, repo, mock_db):
        mock_db.fetchval.return_value = 0
        mock_db.fetch.return_value = []

        await repo.list_entities(entity_type="COMPANY")
        # Verify the SQL includes the entity_type filter
        call_args = mock_db.fetch.call_args
        sql = call_args[0][0]
        assert "entity_type" in sql.lower() or "type" in sql.lower()

    @pytest.mark.asyncio
    async def test_with_search(self, repo, mock_db):
        mock_db.fetchval.return_value = 0
        mock_db.fetch.return_value = []

        await repo.list_entities(search="nvidia")
        # Verify a query was made (search produces ILIKE filter)
        assert mock_db.fetch.called

    @pytest.mark.asyncio
    async def test_sort_by_recent(self, repo, mock_db):
        mock_db.fetchval.return_value = 0
        mock_db.fetch.return_value = []

        await repo.list_entities(sort="recent")
        assert mock_db.fetch.called


# ── get_entity_detail ────────────────────────────


class TestGetEntityDetail:
    """Tests for get_entity_detail method."""

    @pytest.mark.asyncio
    async def test_not_found(self, repo, mock_db):
        mock_db.fetchrow.return_value = None

        result = await repo.get_entity_detail("COMPANY", "NONEXISTENT")
        assert result is None

    @pytest.mark.asyncio
    async def test_found(self, repo, mock_db):
        mock_db.fetchrow.return_value = {
            "type": "COMPANY",
            "normalized": "NVIDIA",
            "mention_count": 42,
            "first_seen": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "last_seen": datetime(2026, 2, 5, tzinfo=timezone.utc),
        }
        mock_db.fetch.return_value = [
            {"platform": "twitter", "count": 20},
            {"platform": "newsfilter", "count": 22},
        ]

        result = await repo.get_entity_detail("COMPANY", "NVIDIA")
        assert result is not None
        assert result["mention_count"] == 42
        assert result["platforms"]["twitter"] == 20


# ── get_entity_sentiment ─────────────────────────


class TestGetEntitySentiment:
    """Tests for get_entity_sentiment method."""

    @pytest.mark.asyncio
    async def test_no_data(self, repo, mock_db):
        mock_db.fetchrow.return_value = None

        result = await repo.get_entity_sentiment("COMPANY", "NONEXISTENT")
        assert result is None

    @pytest.mark.asyncio
    async def test_with_data(self, repo, mock_db):
        # Single fetchrow returns all columns including recent_avg/baseline_avg
        mock_db.fetchrow.return_value = {
            "avg_score": 0.352,
            "pos_count": 20,
            "neg_count": 5,
            "neu_count": 10,
            "recent_avg": 0.45,
            "baseline_avg": 0.30,
        }

        result = await repo.get_entity_sentiment("COMPANY", "NVIDIA")
        assert result is not None
        assert result["avg_score"] == 0.352
        assert result["pos_count"] == 20
        assert result["trend"] == "improving"


# ── get_trending_entities ────────────────────────


class TestGetTrendingEntities:
    """Tests for get_trending_entities method."""

    @pytest.mark.asyncio
    async def test_empty(self, repo, mock_db):
        mock_db.fetch.return_value = []

        result = await repo.get_trending_entities()
        assert result == []

    @pytest.mark.asyncio
    async def test_with_results(self, repo, mock_db):
        mock_db.fetch.return_value = [
            {
                "type": "COMPANY",
                "normalized": "NVIDIA",
                "recent_count": 50,
                "baseline_count": 10,
                "spike_ratio": 5.0,
            },
        ]

        result = await repo.get_trending_entities(hours_recent=24, hours_baseline=168)
        assert len(result) == 1
        assert result[0]["spike_ratio"] == 5.0


# ── get_cooccurring_entities ─────────────────────


class TestGetCooccurringEntities:
    """Tests for get_cooccurring_entities method."""

    @pytest.mark.asyncio
    async def test_empty(self, repo, mock_db):
        mock_db.fetch.return_value = []

        result = await repo.get_cooccurring_entities("COMPANY", "NVIDIA")
        assert result == []

    @pytest.mark.asyncio
    async def test_with_results(self, repo, mock_db):
        mock_db.fetch.return_value = [
            {"type": "TICKER", "normalized": "NVDA", "cooccurrence_count": 30, "jaccard": 0.75},
            {"type": "COMPANY", "normalized": "AMD", "cooccurrence_count": 15, "jaccard": 0.45},
        ]

        result = await repo.get_cooccurring_entities("COMPANY", "NVIDIA", limit=20, min_count=2)
        assert len(result) == 2
        assert result[0]["jaccard"] == 0.75


# ── merge_entity ─────────────────────────────────


class TestMergeEntity:
    """Tests for merge_entity method."""

    @pytest.mark.asyncio
    async def test_no_affected_docs(self, repo, mock_db):
        # merge_entity uses fetchval (returns count from CTE)
        mock_db.fetchval.return_value = 0

        result = await repo.merge_entity(
            from_type="COMPANY",
            from_normalized="Nvidia Corp",
            to_type="COMPANY",
            to_normalized="NVIDIA",
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_affected_docs(self, repo, mock_db):
        mock_db.fetchval.return_value = 15

        result = await repo.merge_entity(
            from_type="COMPANY",
            from_normalized="Nvidia Corp",
            to_type="COMPANY",
            to_normalized="NVIDIA",
        )
        assert result == 15

    @pytest.mark.asyncio
    async def test_calls_db_fetchval(self, repo, mock_db):
        mock_db.fetchval.return_value = 3

        await repo.merge_entity("COMPANY", "Nvidia Corp", "COMPANY", "NVIDIA")
        assert mock_db.fetchval.called
        sql = mock_db.fetchval.call_args[0][0]
        assert "UPDATE" in sql or "update" in sql.lower()
