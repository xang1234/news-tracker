"""Tests for feedback REST API endpoints."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_feedback_repository
from src.feedback.repository import FeedbackRepository
from src.feedback.schemas import Feedback


def _make_feedback(
    feedback_id: str = "feedback_abc123",
    entity_type: str = "theme",
    entity_id: str = "theme_xyz789",
    rating: int = 4,
    **kwargs,
) -> Feedback:
    """Helper to create a Feedback with sensible defaults."""
    return Feedback(
        feedback_id=feedback_id,
        entity_type=entity_type,
        entity_id=entity_id,
        rating=rating,
        quality_label=kwargs.pop("quality_label", "useful"),
        comment=kwargs.pop("comment", "Good insight"),
        user_id=kwargs.pop("user_id", "test-key"),
        created_at=kwargs.pop(
            "created_at", datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc)
        ),
        **kwargs,
    )


@pytest.fixture
def mock_feedback_repo():
    """Mock FeedbackRepository."""
    repo = AsyncMock(spec=FeedbackRepository)
    repo.create = AsyncMock()
    repo.list_by_entity = AsyncMock(return_value=[])
    repo.get_stats = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def client(mock_feedback_repo):
    """FastAPI TestClient with dependency overrides for feedback."""
    app = create_app()

    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_feedback_repository] = lambda: mock_feedback_repo

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── POST /feedback ──────────────────────────────────────


class TestCreateFeedback:
    """Tests for the create feedback endpoint."""

    def test_create_success(self, client, mock_feedback_repo):
        created = _make_feedback()
        mock_feedback_repo.create.return_value = created

        resp = client.post("/feedback", json={
            "entity_type": "theme",
            "entity_id": "theme_xyz789",
            "rating": 4,
            "quality_label": "useful",
            "comment": "Good insight",
        })

        assert resp.status_code == 201
        data = resp.json()
        assert data["feedback"]["entity_type"] == "theme"
        assert data["feedback"]["entity_id"] == "theme_xyz789"
        assert data["feedback"]["rating"] == 4
        assert data["feedback"]["quality_label"] == "useful"
        assert data["feedback"]["comment"] == "Good insight"
        assert "latency_ms" in data

    def test_create_minimal(self, client, mock_feedback_repo):
        created = _make_feedback(quality_label=None, comment=None)
        mock_feedback_repo.create.return_value = created

        resp = client.post("/feedback", json={
            "entity_type": "theme",
            "entity_id": "theme_xyz789",
            "rating": 4,
        })

        assert resp.status_code == 201

    def test_create_all_entity_types(self, client, mock_feedback_repo):
        for et in ("theme", "alert", "document"):
            created = _make_feedback(entity_type=et)
            mock_feedback_repo.create.return_value = created

            resp = client.post("/feedback", json={
                "entity_type": et,
                "entity_id": "id_1",
                "rating": 3,
            })
            assert resp.status_code == 201

    def test_create_invalid_entity_type(self, client, mock_feedback_repo):
        resp = client.post("/feedback", json={
            "entity_type": "user",
            "entity_id": "x",
            "rating": 3,
        })
        assert resp.status_code == 422
        assert "Invalid entity_type" in resp.json()["detail"]

    def test_create_invalid_quality_label(self, client, mock_feedback_repo):
        resp = client.post("/feedback", json={
            "entity_type": "theme",
            "entity_id": "x",
            "rating": 3,
            "quality_label": "bad",
        })
        assert resp.status_code == 422
        assert "Invalid quality_label" in resp.json()["detail"]

    def test_create_rating_too_low(self, client, mock_feedback_repo):
        resp = client.post("/feedback", json={
            "entity_type": "theme",
            "entity_id": "x",
            "rating": 0,
        })
        assert resp.status_code == 422

    def test_create_rating_too_high(self, client, mock_feedback_repo):
        resp = client.post("/feedback", json={
            "entity_type": "theme",
            "entity_id": "x",
            "rating": 6,
        })
        assert resp.status_code == 422

    def test_create_user_id_from_api_key(self, client, mock_feedback_repo):
        created = _make_feedback(user_id="test-key")
        mock_feedback_repo.create.return_value = created

        resp = client.post("/feedback", json={
            "entity_type": "theme",
            "entity_id": "x",
            "rating": 5,
        })

        assert resp.status_code == 201
        # Verify the Feedback passed to create() has user_id set
        call_args = mock_feedback_repo.create.call_args[0][0]
        assert call_args.user_id == "test-key"

    def test_create_server_error(self, client, mock_feedback_repo):
        mock_feedback_repo.create.side_effect = RuntimeError("DB down")

        resp = client.post("/feedback", json={
            "entity_type": "theme",
            "entity_id": "x",
            "rating": 3,
        })
        assert resp.status_code == 500
        assert "Failed to create feedback" in resp.json()["detail"]

    def test_create_missing_required_fields(self, client, mock_feedback_repo):
        resp = client.post("/feedback", json={
            "entity_type": "theme",
        })
        assert resp.status_code == 422


# ── GET /feedback/stats ─────────────────────────────────


class TestGetFeedbackStats:
    """Tests for the feedback stats endpoint."""

    def test_stats_empty(self, client, mock_feedback_repo):
        mock_feedback_repo.get_stats.return_value = []

        resp = client.get("/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"] == []
        assert data["total"] == 0
        assert "latency_ms" in data

    def test_stats_with_data(self, client, mock_feedback_repo):
        mock_feedback_repo.get_stats.return_value = [
            {
                "entity_type": "theme",
                "total_count": 42,
                "avg_rating": 3.5,
                "label_distribution": {
                    "useful": 20,
                    "noise": 10,
                    "too_late": 7,
                    "wrong_direction": 5,
                },
            },
            {
                "entity_type": "alert",
                "total_count": 15,
                "avg_rating": 4.2,
                "label_distribution": {
                    "useful": 12,
                    "noise": 2,
                    "too_late": 1,
                    "wrong_direction": 0,
                },
            },
        ]

        resp = client.get("/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["stats"][0]["entity_type"] == "theme"
        assert data["stats"][0]["total_count"] == 42
        assert data["stats"][0]["avg_rating"] == 3.5
        assert data["stats"][0]["label_distribution"]["useful"] == 20
        assert data["stats"][1]["entity_type"] == "alert"

    def test_stats_entity_type_filter(self, client, mock_feedback_repo):
        mock_feedback_repo.get_stats.return_value = []

        resp = client.get("/feedback/stats?entity_type=alert")
        assert resp.status_code == 200
        mock_feedback_repo.get_stats.assert_called_once_with(entity_type="alert")

    def test_stats_invalid_entity_type(self, client, mock_feedback_repo):
        resp = client.get("/feedback/stats?entity_type=bogus")
        assert resp.status_code == 422
        assert "Invalid entity_type" in resp.json()["detail"]

    def test_stats_server_error(self, client, mock_feedback_repo):
        mock_feedback_repo.get_stats.side_effect = RuntimeError("DB down")

        resp = client.get("/feedback/stats")
        assert resp.status_code == 500
        assert "Failed to get feedback stats" in resp.json()["detail"]
