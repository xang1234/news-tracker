"""Tests for SourcesRepository."""

from unittest.mock import AsyncMock

import pytest

from src.sources.repository import SourcesRepository
from src.sources.schemas import Source


class TestUpsert:
    """Tests for single-source upsert."""

    @pytest.mark.asyncio
    async def test_passes_correct_params(
        self, mock_database: AsyncMock, sample_source: Source
    ) -> None:
        repo = SourcesRepository(mock_database)
        mock_database.fetch.return_value = [
            {"platform": "twitter", "identifier": "SemiAnalysis"}
        ]

        await repo.upsert(sample_source)

        args = mock_database.fetch.call_args[0]
        sql = args[0]
        assert "INSERT INTO sources" in sql
        assert "ON CONFLICT (platform, identifier) DO UPDATE" in sql
        assert args[1] == "twitter"
        assert args[2] == "SemiAnalysis"
        assert args[3] == "SemiAnalysis"
        assert args[4] == "Deep semiconductor analysis"


class TestBulkUpsert:
    """Tests for bulk upsert."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero(self, mock_database: AsyncMock) -> None:
        repo = SourcesRepository(mock_database)
        result = await repo.bulk_upsert([])
        assert result == 0
        mock_database.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_parallel_arrays(self, mock_database: AsyncMock) -> None:
        repo = SourcesRepository(mock_database)
        sources = [
            Source(platform="twitter", identifier="SemiAnalysis", display_name="SemiAnalysis"),
            Source(platform="reddit", identifier="wallstreetbets", display_name="WSB"),
        ]

        result = await repo.bulk_upsert(sources)

        assert result == 2
        args = mock_database.execute.call_args[0]
        sql = args[0]
        assert "unnest" in sql
        assert args[1] == ["twitter", "reddit"]  # platforms array
        assert args[2] == ["SemiAnalysis", "wallstreetbets"]  # identifiers array


class TestGetByKey:
    """Tests for single-source lookup."""

    @pytest.mark.asyncio
    async def test_found(
        self, mock_database: AsyncMock, sample_db_row: dict
    ) -> None:
        mock_database.fetchrow.return_value = sample_db_row
        repo = SourcesRepository(mock_database)

        result = await repo.get_by_key("twitter", "SemiAnalysis")

        assert result is not None
        assert result.platform == "twitter"
        assert result.identifier == "SemiAnalysis"
        assert result.display_name == "SemiAnalysis"

    @pytest.mark.asyncio
    async def test_not_found(self, mock_database: AsyncMock) -> None:
        mock_database.fetchrow.return_value = None
        repo = SourcesRepository(mock_database)

        result = await repo.get_by_key("twitter", "nonexistent")

        assert result is None


class TestListSources:
    """Tests for paginated list with filters."""

    @pytest.mark.asyncio
    async def test_no_filters(
        self, mock_database: AsyncMock, sample_db_row: dict
    ) -> None:
        mock_database.fetchval.return_value = 1
        mock_database.fetch.return_value = [sample_db_row]
        repo = SourcesRepository(mock_database)

        sources, total = await repo.list_sources()

        assert total == 1
        assert len(sources) == 1
        assert sources[0].identifier == "SemiAnalysis"

    @pytest.mark.asyncio
    async def test_platform_filter(
        self, mock_database: AsyncMock, sample_db_row: dict
    ) -> None:
        mock_database.fetchval.return_value = 1
        mock_database.fetch.return_value = [sample_db_row]
        repo = SourcesRepository(mock_database)

        sources, total = await repo.list_sources(platform="twitter")

        assert total == 1
        # Verify the SQL includes a platform filter
        count_sql = mock_database.fetchval.call_args[0][0]
        assert "platform = $" in count_sql

    @pytest.mark.asyncio
    async def test_search_filter(
        self, mock_database: AsyncMock, sample_db_row: dict
    ) -> None:
        mock_database.fetchval.return_value = 1
        mock_database.fetch.return_value = [sample_db_row]
        repo = SourcesRepository(mock_database)

        sources, total = await repo.list_sources(search="semi")

        assert total == 1
        count_sql = mock_database.fetchval.call_args[0][0]
        assert "ILIKE" in count_sql

    @pytest.mark.asyncio
    async def test_active_only_filter(self, mock_database: AsyncMock) -> None:
        mock_database.fetchval.return_value = 0
        mock_database.fetch.return_value = []
        repo = SourcesRepository(mock_database)

        sources, total = await repo.list_sources(active_only=True)

        count_sql = mock_database.fetchval.call_args[0][0]
        assert "is_active = TRUE" in count_sql

    @pytest.mark.asyncio
    async def test_pagination(
        self, mock_database: AsyncMock, sample_db_row: dict
    ) -> None:
        mock_database.fetchval.return_value = 50
        mock_database.fetch.return_value = [sample_db_row]
        repo = SourcesRepository(mock_database)

        sources, total = await repo.list_sources(limit=10, offset=20)

        assert total == 50
        data_sql = mock_database.fetch.call_args[0][0]
        assert "LIMIT" in data_sql
        assert "OFFSET" in data_sql

    @pytest.mark.asyncio
    async def test_empty_result(self, mock_database: AsyncMock) -> None:
        mock_database.fetchval.return_value = 0
        mock_database.fetch.return_value = []
        repo = SourcesRepository(mock_database)

        sources, total = await repo.list_sources()

        assert total == 0
        assert sources == []


class TestGetActiveByPlatform:
    """Tests for fetching active sources by platform."""

    @pytest.mark.asyncio
    async def test_returns_list(
        self, mock_database: AsyncMock, sample_db_row: dict
    ) -> None:
        mock_database.fetch.return_value = [sample_db_row]
        repo = SourcesRepository(mock_database)

        result = await repo.get_active_by_platform("twitter")

        assert len(result) == 1
        assert result[0].platform == "twitter"
        sql = mock_database.fetch.call_args[0][0]
        assert "is_active = TRUE" in sql

    @pytest.mark.asyncio
    async def test_empty(self, mock_database: AsyncMock) -> None:
        repo = SourcesRepository(mock_database)
        result = await repo.get_active_by_platform("twitter")
        assert result == []


class TestDeactivate:
    """Tests for soft deactivation."""

    @pytest.mark.asyncio
    async def test_success(self, mock_database: AsyncMock) -> None:
        mock_database.execute.return_value = "UPDATE 1"
        repo = SourcesRepository(mock_database)

        result = await repo.deactivate("twitter", "SemiAnalysis")

        assert result is True

    @pytest.mark.asyncio
    async def test_not_found(self, mock_database: AsyncMock) -> None:
        mock_database.execute.return_value = "UPDATE 0"
        repo = SourcesRepository(mock_database)

        result = await repo.deactivate("twitter", "nonexistent")

        assert result is False


class TestCount:
    """Tests for count."""

    @pytest.mark.asyncio
    async def test_returns_count(self, mock_database: AsyncMock) -> None:
        mock_database.fetchval.return_value = 32
        repo = SourcesRepository(mock_database)

        result = await repo.count()

        assert result == 32
