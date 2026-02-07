"""Tests for FeedbackRepository SQL and parameter passing."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, call

import pytest

from src.feedback.repository import FeedbackRepository
from src.feedback.schemas import Feedback


@pytest.fixture
def repo(mock_database):
    """FeedbackRepository with a mock database."""
    return FeedbackRepository(mock_database)


class TestCreate:
    """Tests for FeedbackRepository.create()."""

    @pytest.mark.asyncio
    async def test_create_inserts_all_fields(self, repo, mock_database, sample_feedback):
        mock_database.fetchrow.return_value = {
            "feedback_id": sample_feedback.feedback_id,
            "entity_type": sample_feedback.entity_type,
            "entity_id": sample_feedback.entity_id,
            "rating": sample_feedback.rating,
            "quality_label": sample_feedback.quality_label,
            "comment": sample_feedback.comment,
            "user_id": sample_feedback.user_id,
            "created_at": sample_feedback.created_at,
        }

        result = await repo.create(sample_feedback)

        assert result.feedback_id == sample_feedback.feedback_id
        assert result.entity_type == "theme"
        assert result.rating == 4

        # Verify SQL was called with correct params
        args = mock_database.fetchrow.call_args
        sql = args[0][0]
        assert "INSERT INTO feedback" in sql
        assert "RETURNING *" in sql
        assert args[0][1] == sample_feedback.feedback_id
        assert args[0][2] == "theme"
        assert args[0][3] == "theme_xyz789"
        assert args[0][4] == 4

    @pytest.mark.asyncio
    async def test_create_with_none_optionals(self, repo, mock_database):
        feedback = Feedback(entity_type="alert", entity_id="alert_1", rating=3)
        mock_database.fetchrow.return_value = {
            "feedback_id": feedback.feedback_id,
            "entity_type": "alert",
            "entity_id": "alert_1",
            "rating": 3,
            "quality_label": None,
            "comment": None,
            "user_id": None,
            "created_at": feedback.created_at,
        }

        result = await repo.create(feedback)
        assert result.quality_label is None
        assert result.comment is None
        assert result.user_id is None


class TestListByEntity:
    """Tests for FeedbackRepository.list_by_entity()."""

    @pytest.mark.asyncio
    async def test_list_by_entity_params(self, repo, mock_database):
        mock_database.fetch.return_value = []

        await repo.list_by_entity("theme", "theme_abc", limit=10, offset=5)

        args = mock_database.fetch.call_args
        sql = args[0][0]
        assert "entity_type = $1" in sql
        assert "entity_id = $2" in sql
        assert "LIMIT $3 OFFSET $4" in sql
        assert args[0][1] == "theme"
        assert args[0][2] == "theme_abc"
        assert args[0][3] == 10
        assert args[0][4] == 5

    @pytest.mark.asyncio
    async def test_list_by_entity_returns_feedback(self, repo, mock_database):
        now = datetime.now(timezone.utc)
        mock_database.fetch.return_value = [
            {
                "feedback_id": "feedback_aaa111",
                "entity_type": "alert",
                "entity_id": "alert_1",
                "rating": 5,
                "quality_label": "useful",
                "comment": "Great alert",
                "user_id": "key-1",
                "created_at": now,
            },
        ]

        results = await repo.list_by_entity("alert", "alert_1")
        assert len(results) == 1
        assert results[0].feedback_id == "feedback_aaa111"
        assert results[0].rating == 5

    @pytest.mark.asyncio
    async def test_list_by_entity_default_pagination(self, repo, mock_database):
        mock_database.fetch.return_value = []

        await repo.list_by_entity("document", "doc_1")

        args = mock_database.fetch.call_args
        assert args[0][3] == 50  # default limit
        assert args[0][4] == 0   # default offset


class TestGetStats:
    """Tests for FeedbackRepository.get_stats()."""

    @pytest.mark.asyncio
    async def test_get_stats_no_filter(self, repo, mock_database):
        mock_database.fetch.return_value = [
            {
                "entity_type": "theme",
                "total_count": 42,
                "avg_rating": 3.5,
                "label_useful": 20,
                "label_noise": 10,
                "label_too_late": 7,
                "label_wrong_direction": 5,
            },
        ]

        results = await repo.get_stats()
        assert len(results) == 1
        assert results[0]["entity_type"] == "theme"
        assert results[0]["total_count"] == 42
        assert results[0]["avg_rating"] == 3.5
        assert results[0]["label_distribution"]["useful"] == 20
        assert results[0]["label_distribution"]["noise"] == 10

        # Verify no filtering WHERE clause (FILTER (WHERE ...) is aggregation syntax)
        args = mock_database.fetch.call_args
        sql = args[0][0]
        assert "entity_type = $1" not in sql
        # No params passed beyond the SQL itself
        assert len(args[0]) == 1

    @pytest.mark.asyncio
    async def test_get_stats_with_entity_type_filter(self, repo, mock_database):
        mock_database.fetch.return_value = []

        await repo.get_stats(entity_type="alert")

        args = mock_database.fetch.call_args
        sql = args[0][0]
        assert "WHERE" in sql
        assert "entity_type = $1" in sql
        assert args[0][1] == "alert"

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, repo, mock_database):
        mock_database.fetch.return_value = []

        results = await repo.get_stats()
        assert results == []
