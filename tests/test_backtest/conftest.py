"""Pytest fixtures for backtest tests."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.backtest.audit import BacktestRun
from src.backtest.config import BacktestConfig
from src.backtest.engine import DailyBacktestResult
from src.backtest.model_versions import ModelVersion
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


# ── Engine/Metrics fixtures ──────────────────────────────


def _make_theme(
    theme_id: str,
    name: str,
    tickers: list[str],
    lifecycle: str = "emerging",
    compellingness: float = 5.0,
) -> Theme:
    """Helper to create a Theme with minimal boilerplate."""
    return Theme(
        theme_id=theme_id,
        name=name,
        centroid=np.zeros(768, dtype=np.float32),
        top_tickers=tickers,
        top_keywords=["semiconductor", "chip"],
        lifecycle_stage=lifecycle,
        document_count=50,
        metadata={"compellingness": compellingness},
    )


@pytest.fixture
def sample_themes() -> list[Theme]:
    """Three themes with distinct tickers for engine tests."""
    return [
        _make_theme("theme_nvda", "NVIDIA AI Theme", ["NVDA", "AMD", "INTC"], "accelerating"),
        _make_theme("theme_mem", "Memory Supply Theme", ["MU", "WDC", "STX"], "emerging"),
        _make_theme("theme_eda", "EDA Software Theme", ["SNPS", "CDNS", "MRVL"], "mature"),
    ]


@pytest.fixture
def sample_theme_metrics() -> dict[str, ThemeMetrics]:
    """Metrics map with varying z-scores for ranking differentiation."""
    return {
        "theme_nvda": ThemeMetrics(
            theme_id="theme_nvda",
            date=date(2025, 6, 15),
            document_count=100,
            volume_zscore=3.5,
            weighted_volume=1500.0,
        ),
        "theme_mem": ThemeMetrics(
            theme_id="theme_mem",
            date=date(2025, 6, 15),
            document_count=50,
            volume_zscore=1.0,
            weighted_volume=500.0,
        ),
        "theme_eda": ThemeMetrics(
            theme_id="theme_eda",
            date=date(2025, 6, 15),
            document_count=30,
            volume_zscore=-0.5,
            weighted_volume=200.0,
        ),
    }


@pytest.fixture
def sample_forward_returns() -> dict[str, dict[int, float | None]]:
    """Deterministic forward returns for test tickers."""
    return {
        "NVDA": {1: 0.02, 5: 0.05, 10: 0.08, 20: 0.12},
        "AMD": {1: -0.01, 5: 0.03, 10: 0.06, 20: 0.10},
        "INTC": {1: 0.005, 5: -0.02, 10: -0.03, 20: -0.05},
        "MU": {1: 0.03, 5: 0.04, 10: 0.07, 20: 0.09},
        "WDC": {1: -0.02, 5: -0.01, 10: 0.01, 20: 0.02},
        "STX": {1: 0.01, 5: 0.02, 10: 0.03, 20: 0.04},
    }


@pytest.fixture
def sample_daily_results() -> list[DailyBacktestResult]:
    """Pre-built daily results for metrics tests."""
    return [
        DailyBacktestResult(
            date=date(2025, 6, 2),
            ranked_themes=[{"score": 5.0, "theme_id": "t1"}],
            top_n_tickers=["NVDA", "AMD"],
            top_n_avg_return=0.03,
            direction_correct=True,
            theme_count=10,
        ),
        DailyBacktestResult(
            date=date(2025, 6, 3),
            ranked_themes=[{"score": 4.0, "theme_id": "t2"}],
            top_n_tickers=["NVDA", "MU"],
            top_n_avg_return=-0.01,
            direction_correct=False,
            theme_count=10,
        ),
        DailyBacktestResult(
            date=date(2025, 6, 4),
            ranked_themes=[{"score": 6.0, "theme_id": "t1"}],
            top_n_tickers=["NVDA"],
            top_n_avg_return=0.05,
            direction_correct=True,
            theme_count=10,
        ),
        DailyBacktestResult(
            date=date(2025, 6, 5),
            ranked_themes=[{"score": 3.0, "theme_id": "t3"}],
            top_n_tickers=["AMD", "INTC"],
            top_n_avg_return=-0.02,
            direction_correct=False,
            theme_count=10,
        ),
        DailyBacktestResult(
            date=date(2025, 6, 6),
            ranked_themes=[{"score": 7.0, "theme_id": "t1"}],
            top_n_tickers=["NVDA", "AMD", "MU"],
            top_n_avg_return=0.04,
            direction_correct=True,
            theme_count=10,
        ),
    ]


@pytest.fixture
def mock_theme_repo() -> AsyncMock:
    """AsyncMock of ThemeRepository for engine tests."""
    repo = AsyncMock()
    repo.get_all = AsyncMock(return_value=[])
    repo.get_all_as_of = AsyncMock(return_value=[])
    repo.get_metrics_range = AsyncMock(return_value=[])
    return repo
