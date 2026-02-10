"""Shared fixtures for API tests."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import (
    get_document_repository,
    get_ranking_service,
    get_sentiment_aggregator,
    get_theme_repository,
)
from src.themes.ranking import RankedTheme, ThemeRankingService
from src.themes.schemas import Theme, ThemeMetrics


def _make_theme(
    theme_id: str = "theme_abc123",
    name: str = "gpu_nvidia_hbm",
    lifecycle_stage: str = "emerging",
    document_count: int = 42,
    **kwargs,
) -> Theme:
    """Helper to create a Theme with sensible defaults."""
    return Theme(
        theme_id=theme_id,
        name=name,
        centroid=kwargs.pop("centroid", np.random.rand(768).astype(np.float32)),
        top_keywords=kwargs.pop("top_keywords", ["gpu", "nvidia", "hbm"]),
        top_tickers=kwargs.pop("top_tickers", ["NVDA", "AMD"]),
        top_entities=kwargs.pop("top_entities", [{"name": "NVIDIA", "score": 0.9}]),
        lifecycle_stage=lifecycle_stage,
        document_count=document_count,
        created_at=kwargs.pop(
            "created_at", datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ),
        updated_at=kwargs.pop(
            "updated_at", datetime(2026, 2, 5, 8, 30, 0, tzinfo=timezone.utc)
        ),
        description=kwargs.pop("description", "GPU and NVIDIA HBM memory theme"),
        metadata=kwargs.pop("metadata", {"bertopic_topic_id": 3}),
        **kwargs,
    )


def _make_metrics(theme_id: str = "theme_abc123", target_date: date | None = None) -> ThemeMetrics:
    """Helper to create a ThemeMetrics row."""
    return ThemeMetrics(
        theme_id=theme_id,
        date=target_date or date(2026, 2, 5),
        document_count=10,
        sentiment_score=0.3,
        volume_zscore=1.2,
        velocity=0.05,
        acceleration=0.01,
        avg_authority=0.65,
        bullish_ratio=0.6,
    )


@pytest.fixture
def mock_theme_repo():
    """Mock ThemeRepository."""
    repo = AsyncMock()
    repo.get_all = AsyncMock(return_value=[])
    repo.get_by_id = AsyncMock(return_value=None)
    repo.get_metrics_range = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_doc_repo():
    """Mock DocumentRepository."""
    repo = AsyncMock()
    repo.get_documents_by_theme = AsyncMock(return_value=[])
    repo.get_sentiments_for_theme = AsyncMock(return_value=[])
    repo.get_events_by_tickers = AsyncMock(return_value=[])
    # Document explorer endpoints
    repo.list_documents = AsyncMock(return_value=[])
    repo.list_documents_count = AsyncMock(return_value=0)
    repo.get_document_stats = AsyncMock(return_value={
        "total_count": 0,
        "platform_counts": [],
        "embedding_coverage": {"finbert_pct": 0.0, "minilm_pct": 0.0},
        "sentiment_coverage": 0.0,
        "earliest_document": None,
        "latest_document": None,
    })
    repo.get_by_id = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_aggregator():
    """Mock SentimentAggregator."""
    return MagicMock()


@pytest.fixture
def mock_ranking_service():
    """Mock ThemeRankingService."""
    service = AsyncMock(spec=ThemeRankingService)
    service.get_actionable = AsyncMock(return_value=[])
    return service


@pytest.fixture
def client(mock_theme_repo, mock_doc_repo, mock_aggregator, mock_ranking_service):
    """FastAPI TestClient with dependency overrides."""
    app = create_app()

    # Override auth to bypass API key checks
    app.dependency_overrides[verify_api_key] = lambda: "test-key"

    # Override data dependencies with mocks
    app.dependency_overrides[get_theme_repository] = lambda: mock_theme_repo
    app.dependency_overrides[get_document_repository] = lambda: mock_doc_repo
    app.dependency_overrides[get_sentiment_aggregator] = lambda: mock_aggregator
    app.dependency_overrides[get_ranking_service] = lambda: mock_ranking_service

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
