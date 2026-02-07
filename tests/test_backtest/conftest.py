"""Pytest fixtures for backtest tests."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.backtest.audit import BacktestRun
from src.backtest.config import BacktestConfig
from src.backtest.model_versions import ModelVersion


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
def backtest_config() -> BacktestConfig:
    """Default BacktestConfig for tests."""
    return BacktestConfig()


@pytest.fixture
def sample_model_version() -> ModelVersion:
    """A fully populated ModelVersion for testing."""
    return ModelVersion(
        version_id="mv_abc123def456",
        embedding_model="ProsusAI/finbert",
        clustering_config={
            "hdbscan_min_cluster_size": 10,
            "umap_n_components": 10,
            "similarity_threshold_assign": 0.75,
        },
        config_snapshot={
            "embedding_model_name": "ProsusAI/finbert",
            "clustering_enabled": True,
        },
        created_at=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        description="Initial model version",
    )


@pytest.fixture
def sample_model_version_row() -> dict:
    """A dict mimicking an asyncpg Record for model_versions."""
    return {
        "version_id": "mv_abc123def456",
        "embedding_model": "ProsusAI/finbert",
        "clustering_config": '{"hdbscan_min_cluster_size": 10}',
        "config_snapshot": '{"embedding_model_name": "ProsusAI/finbert"}',
        "created_at": datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        "description": "Initial model version",
    }


@pytest.fixture
def sample_backtest_run() -> BacktestRun:
    """A fully populated BacktestRun for testing."""
    return BacktestRun(
        run_id="run_test_001",
        model_version_id="mv_abc123def456",
        date_range_start=date(2025, 1, 1),
        date_range_end=date(2025, 6, 30),
        parameters={"strategy": "swing", "forward_horizons": [1, 5, 10]},
        status="running",
        created_at=datetime(2025, 7, 1, 10, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_backtest_run_row() -> dict:
    """A dict mimicking an asyncpg Record for backtest_runs."""
    return {
        "run_id": "run_test_001",
        "model_version_id": "mv_abc123def456",
        "date_range_start": date(2025, 1, 1),
        "date_range_end": date(2025, 6, 30),
        "parameters": '{"strategy": "swing"}',
        "results": None,
        "status": "running",
        "created_at": datetime(2025, 7, 1, 10, 0, 0, tzinfo=timezone.utc),
        "completed_at": None,
        "error_message": None,
    }
