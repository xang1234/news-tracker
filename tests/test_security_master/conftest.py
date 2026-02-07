"""Shared fixtures for security master tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.security_master.schemas import Security


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
def sample_security() -> Security:
    """A sample Security for testing."""
    return Security(
        ticker="NVDA",
        exchange="US",
        name="NVIDIA Corporation",
        aliases=["nvidia", "nvda", "geforce", "jensen huang"],
        sector="gpu_ai",
        country="US",
        currency="USD",
    )


@pytest.fixture
def sample_db_row() -> dict:
    """A dict mimicking an asyncpg Record for a security."""
    return {
        "ticker": "NVDA",
        "exchange": "US",
        "name": "NVIDIA Corporation",
        "aliases": ["nvidia", "nvda", "geforce", "jensen huang"],
        "sector": "gpu_ai",
        "country": "US",
        "currency": "USD",
        "figi": None,
        "is_active": True,
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }


@pytest.fixture
def sample_korean_row() -> dict:
    """A dict mimicking an asyncpg Record for a Korean security."""
    return {
        "ticker": "005930.KS",
        "exchange": "KRX",
        "name": "Samsung Electronics",
        "aliases": ["samsung", "samsung semiconductor", "samsung memory"],
        "sector": "memory",
        "country": "KR",
        "currency": "KRW",
        "figi": "BBG000H7TBB4",
        "is_active": True,
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }
