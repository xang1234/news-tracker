"""Tests for PointInTimeService temporal filtering."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.backtest.point_in_time import PointInTimeService
from src.themes.repository import ThemeRepository
from src.themes.schemas import Theme


@pytest.fixture
def mock_theme_repo(mock_database: AsyncMock) -> ThemeRepository:
    """ThemeRepository with mocked database."""
    return ThemeRepository(mock_database)


@pytest.fixture
def pit_service(
    mock_database: AsyncMock,
    mock_theme_repo: ThemeRepository,
) -> PointInTimeService:
    """PointInTimeService with mocked dependencies."""
    return PointInTimeService(
        database=mock_database,
        theme_repo=mock_theme_repo,
    )


class TestGetThemesAsOf:
    """Test PointInTimeService.get_themes_as_of()."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(
        self,
        pit_service: PointInTimeService,
        mock_theme_repo: ThemeRepository,
    ) -> None:
        """Calls ThemeRepository.get_all_as_of with correct params."""
        mock_theme_repo.get_all_as_of = AsyncMock(return_value=[])
        as_of = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        result = await pit_service.get_themes_as_of(
            as_of=as_of,
            lifecycle_stages=["emerging"],
            limit=50,
        )

        mock_theme_repo.get_all_as_of.assert_called_once_with(
            as_of=as_of,
            lifecycle_stages=["emerging"],
            limit=50,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_no_filter(
        self,
        pit_service: PointInTimeService,
        mock_theme_repo: ThemeRepository,
    ) -> None:
        """Works with no lifecycle stage filter."""
        mock_theme_repo.get_all_as_of = AsyncMock(return_value=[])
        as_of = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        await pit_service.get_themes_as_of(as_of=as_of)

        mock_theme_repo.get_all_as_of.assert_called_once_with(
            as_of=as_of,
            lifecycle_stages=None,
            limit=100,
        )


class TestGetDocumentsAsOf:
    """Test PointInTimeService.get_documents_as_of()."""

    @pytest.mark.asyncio
    async def test_filters_on_fetched_at(
        self,
        pit_service: PointInTimeService,
        mock_database: AsyncMock,
    ) -> None:
        """SQL filters on fetched_at, NOT timestamp."""
        mock_database.fetch.return_value = []
        as_of = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        await pit_service.get_documents_as_of(as_of=as_of)

        sql = mock_database.fetch.call_args[0][0]
        assert "fetched_at <= $1" in sql
        assert "timestamp" not in sql

    @pytest.mark.asyncio
    async def test_with_since_bound(
        self,
        pit_service: PointInTimeService,
        mock_database: AsyncMock,
    ) -> None:
        """Adds lower bound on fetched_at when since is provided."""
        mock_database.fetch.return_value = []
        as_of = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        since = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        await pit_service.get_documents_as_of(as_of=as_of, since=since)

        sql = mock_database.fetch.call_args[0][0]
        assert "fetched_at >= $2" in sql

    @pytest.mark.asyncio
    async def test_returns_parsed_rows(
        self,
        pit_service: PointInTimeService,
        mock_database: AsyncMock,
    ) -> None:
        """Results include properly parsed fields."""
        mock_database.fetch.return_value = [
            {
                "id": "doc_001",
                "content": "NVDA earnings beat",
                "embedding": "[0.1,0.2,0.3]",
                "authority_score": 0.85,
                "sentiment": '{"label": "positive", "score": 0.9}',
                "theme_ids": ["theme_abc"],
                "fetched_at": datetime(2025, 6, 14, tzinfo=timezone.utc),
            }
        ]
        as_of = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        result = await pit_service.get_documents_as_of(as_of=as_of)

        assert len(result) == 1
        doc = result[0]
        assert doc["id"] == "doc_001"
        assert doc["embedding"] == [0.1, 0.2, 0.3]
        assert doc["sentiment"]["label"] == "positive"
        assert doc["theme_ids"] == ["theme_abc"]

    @pytest.mark.asyncio
    async def test_requires_embedding(
        self,
        pit_service: PointInTimeService,
        mock_database: AsyncMock,
    ) -> None:
        """Only returns documents with embeddings."""
        mock_database.fetch.return_value = []
        as_of = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        await pit_service.get_documents_as_of(as_of=as_of)

        sql = mock_database.fetch.call_args[0][0]
        assert "embedding IS NOT NULL" in sql


class TestGetMetricsAsOf:
    """Test PointInTimeService.get_metrics_as_of()."""

    @pytest.mark.asyncio
    async def test_delegates_to_repo(
        self,
        pit_service: PointInTimeService,
        mock_theme_repo: ThemeRepository,
    ) -> None:
        """Calls get_metrics_range with correct date window."""
        mock_theme_repo.get_metrics_range = AsyncMock(return_value=[])
        as_of = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        await pit_service.get_metrics_as_of(
            theme_id="theme_abc",
            as_of=as_of,
            lookback_days=30,
        )

        call_args = mock_theme_repo.get_metrics_range.call_args
        assert call_args.kwargs["theme_id"] == "theme_abc"
        # end should be as_of.date()
        assert call_args.kwargs["end"] == as_of.date()
        # start should be 30 days before
        expected_start = (as_of - timedelta(days=30)).date()
        assert call_args.kwargs["start"] == expected_start


class TestGetThemeCentroidsAsOf:
    """Test PointInTimeService.get_theme_centroids_as_of()."""

    @pytest.mark.asyncio
    async def test_returns_centroid_dict(
        self,
        pit_service: PointInTimeService,
        mock_theme_repo: ThemeRepository,
    ) -> None:
        """Returns dict mapping theme_id to centroid list."""
        centroid = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        theme = Theme(
            theme_id="theme_abc",
            name="test",
            centroid=centroid,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        mock_theme_repo.get_all_as_of = AsyncMock(return_value=[theme])
        as_of = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        result = await pit_service.get_theme_centroids_as_of(as_of=as_of)

        assert "theme_abc" in result
        assert result["theme_abc"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
