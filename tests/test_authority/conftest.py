"""Shared fixtures for authority tests."""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

import pytest

from src.authority.config import AuthorityConfig
from src.authority.repository import AuthorityRepository
from src.authority.schemas import AuthorityProfile, AuthorTier
from src.authority.service import AuthorityService


@pytest.fixture
def config():
    """Default authority config."""
    return AuthorityConfig()


@pytest.fixture
def mock_database():
    """Mock Database with async fetch methods."""
    db = AsyncMock()
    db.fetchrow = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock()
    return db


@pytest.fixture
def repository(mock_database):
    """AuthorityRepository with mock database."""
    return AuthorityRepository(mock_database)


@pytest.fixture
def service(config):
    """AuthorityService without repository (compute-only)."""
    return AuthorityService(config=config)


@pytest.fixture
def service_with_repo(config, repository):
    """AuthorityService with repository."""
    return AuthorityService(config=config, repository=repository)


@pytest.fixture
def now():
    """Fixed reference time."""
    return datetime(2026, 2, 8, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def established_profile(now):
    """An established anonymous author with track record."""
    return AuthorityProfile(
        author_id="user_123",
        platform="twitter",
        tier=AuthorTier.ANONYMOUS.value,
        base_weight=1.0,
        total_calls=20,
        correct_calls=15,
        first_seen=now - timedelta(days=90),
        last_good_call=now - timedelta(days=2),
    )


@pytest.fixture
def new_profile(now):
    """A brand-new author with no track record."""
    return AuthorityProfile(
        author_id="new_user",
        platform="twitter",
        tier=AuthorTier.ANONYMOUS.value,
        base_weight=1.0,
        total_calls=0,
        correct_calls=0,
        first_seen=now - timedelta(days=5),
    )


@pytest.fixture
def verified_profile(now):
    """A verified professional author."""
    return AuthorityProfile(
        author_id="analyst_pro",
        platform="twitter",
        tier=AuthorTier.VERIFIED.value,
        base_weight=5.0,
        total_calls=50,
        correct_calls=40,
        first_seen=now - timedelta(days=180),
        last_good_call=now - timedelta(days=1),
    )


@pytest.fixture
def research_profile(now):
    """A specialized research outlet."""
    return AuthorityProfile(
        author_id="semianalysis",
        platform="substack",
        tier=AuthorTier.RESEARCH.value,
        base_weight=10.0,
        total_calls=100,
        correct_calls=85,
        first_seen=now - timedelta(days=365),
        last_good_call=now - timedelta(hours=12),
    )


@pytest.fixture
def stale_profile(now):
    """An author who hasn't had a good call in a long time."""
    return AuthorityProfile(
        author_id="stale_user",
        platform="reddit",
        tier=AuthorTier.ANONYMOUS.value,
        base_weight=1.0,
        total_calls=10,
        correct_calls=5,
        first_seen=now - timedelta(days=365),
        last_good_call=now - timedelta(days=200),
    )
