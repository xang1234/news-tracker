"""Pytest fixtures for theme tests."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.themes.schemas import Theme, ThemeMetrics


@pytest.fixture
def mock_database() -> AsyncMock:
    """Mock Database instance matching the Database API."""
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock(return_value=None)
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="UPDATE 1")
    return db


@pytest.fixture
def sample_centroid() -> np.ndarray:
    """Sample 768-dim centroid vector."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(768).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def sample_theme(sample_centroid: np.ndarray) -> Theme:
    """A fully populated Theme for testing."""
    return Theme(
        theme_id="theme_a1b2c3d4e5f6",
        name="gpu_nvidia_architecture",
        centroid=sample_centroid,
        top_keywords=["gpu", "nvidia", "architecture"],
        top_tickers=["NVDA", "AMD"],
        lifecycle_stage="emerging",
        document_count=25,
        created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 16, 8, 30, 0, tzinfo=timezone.utc),
        description="GPU architecture developments",
        top_entities=[
            {"type": "COMPANY", "normalized": "NVIDIA", "score": 0.95},
            {"type": "PRODUCT", "normalized": "H200", "score": 0.82},
        ],
        metadata={"bertopic_topic_id": 3, "lifecycle_stage": "emerging"},
    )


@pytest.fixture
def sample_db_row(sample_centroid: np.ndarray) -> dict:
    """A dict mimicking an asyncpg Record for _row_to_theme."""
    return {
        "theme_id": "theme_a1b2c3d4e5f6",
        "name": "gpu_nvidia_architecture",
        "centroid": f"[{','.join(str(float(x)) for x in sample_centroid)}]",
        "top_keywords": ["gpu", "nvidia", "architecture"],
        "top_tickers": ["NVDA", "AMD"],
        "lifecycle_stage": "emerging",
        "document_count": 25,
        "created_at": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 1, 16, 8, 30, 0, tzinfo=timezone.utc),
        "description": "GPU architecture developments",
        "top_entities": '[{"type": "COMPANY", "normalized": "NVIDIA", "score": 0.95}]',
        "metadata": '{"bertopic_topic_id": 3}',
    }


@pytest.fixture
def sample_similarity_row(sample_db_row: dict) -> dict:
    """A dict mimicking an asyncpg Record with a similarity column."""
    return {**sample_db_row, "similarity": 0.92}


@pytest.fixture
def sample_metrics() -> ThemeMetrics:
    """A fully populated ThemeMetrics for testing."""
    return ThemeMetrics(
        theme_id="theme_a1b2c3d4e5f6",
        date=date(2025, 6, 15),
        document_count=42,
        sentiment_score=0.35,
        volume_zscore=1.8,
        velocity=0.12,
        acceleration=0.03,
        avg_authority=0.65,
        bullish_ratio=0.72,
    )


@pytest.fixture
def sample_metrics_row() -> dict:
    """A dict mimicking an asyncpg Record for _row_to_metrics."""
    return {
        "theme_id": "theme_a1b2c3d4e5f6",
        "date": date(2025, 6, 15),
        "document_count": 42,
        "sentiment_score": 0.35,
        "volume_zscore": 1.8,
        "velocity": 0.12,
        "acceleration": 0.03,
        "avg_authority": 0.65,
        "bullish_ratio": 0.72,
    }
