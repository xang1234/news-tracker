"""Shared fixtures for sources tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.sources.schemas import Source


@pytest.fixture
def mock_database() -> AsyncMock:
    """Mock Database instance matching the Database API."""
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock(return_value=None)
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="INSERT 0 1")
    return db


@pytest.fixture
def sample_source() -> Source:
    """A sample Source for testing."""
    return Source(
        platform="twitter",
        identifier="SemiAnalysis",
        display_name="SemiAnalysis",
        description="Deep semiconductor analysis",
        metadata={"category": "analyst"},
    )


@pytest.fixture
def sample_db_row() -> dict:
    """A dict mimicking an asyncpg Record for a source."""
    return {
        "platform": "twitter",
        "identifier": "SemiAnalysis",
        "display_name": "SemiAnalysis",
        "description": "Deep semiconductor analysis",
        "is_active": True,
        "metadata": {"category": "analyst"},
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }


@pytest.fixture
def sample_reddit_row() -> dict:
    """A dict mimicking an asyncpg Record for a Reddit source."""
    return {
        "platform": "reddit",
        "identifier": "wallstreetbets",
        "display_name": "WallStreetBets",
        "description": "Options and meme stocks",
        "is_active": True,
        "metadata": {},
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }


@pytest.fixture
def sample_substack_row() -> dict:
    """A dict mimicking an asyncpg Record for a Substack source."""
    return {
        "platform": "substack",
        "identifier": "semianalysis",
        "display_name": "SemiAnalysis",
        "description": "Semiconductor deep dives",
        "is_active": True,
        "metadata": {},
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }
