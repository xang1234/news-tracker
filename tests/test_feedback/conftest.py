"""Shared fixtures for feedback tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.feedback.repository import FeedbackRepository
from src.feedback.schemas import Feedback


@pytest.fixture
def mock_database():
    """Mock Database with async fetch methods."""
    db = AsyncMock()
    db.fetchrow = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock()
    return db


@pytest.fixture
def sample_feedback():
    """A sample Feedback with all fields populated."""
    return Feedback(
        feedback_id="feedback_abc123def456",
        entity_type="theme",
        entity_id="theme_xyz789",
        rating=4,
        quality_label="useful",
        comment="Very actionable insight",
        user_id="test-api-key",
        created_at=datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_feedback_minimal():
    """A minimal Feedback with only required fields."""
    return Feedback(
        entity_type="alert",
        entity_id="alert_001",
        rating=2,
    )
