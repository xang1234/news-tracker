"""Pytest fixtures for scoring tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.scoring.config import ScoringConfig
from src.scoring.schemas import CompellingnessScore, DimensionScores
from src.themes.schemas import Theme


@pytest.fixture
def scoring_config() -> ScoringConfig:
    """Test config with high budgets and cache disabled."""
    return ScoringConfig(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        daily_budget_openai=100.0,
        daily_budget_anthropic=100.0,
        cache_enabled=False,
        tier2_min_rule_score=3.0,
        tier3_min_gpt_score=8.5,
        consensus_tolerance=1.5,
        circuit_failure_threshold=3,
        circuit_recovery_timeout=5.0,
    )


@pytest.fixture
def sample_centroid() -> np.ndarray:
    """768-dim normalized centroid."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(768).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def sample_theme(sample_centroid: np.ndarray) -> Theme:
    """A richly populated Theme for testing."""
    return Theme(
        theme_id="theme_a1b2c3d4e5f6",
        name="gpu_nvidia_architecture",
        centroid=sample_centroid,
        top_keywords=[
            "gpu", "nvidia", "architecture", "hbm", "chiplet",
            "revenue", "growth", "forecast", "capacity", "wafer",
        ],
        top_tickers=["NVDA", "AMD", "INTC"],
        lifecycle_stage="emerging",
        document_count=42,
        created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 16, 8, 30, 0, tzinfo=timezone.utc),
        description=(
            "NVIDIA's next-generation GPU architecture leverages chiplet design "
            "and HBM4 integration. Analyst reports indicate 30% revenue growth YoY. "
            "TSMC foundry capacity expansion supports volume ramp. Key risk: "
            "geopolitical tension around export controls to China."
        ),
        top_entities=[
            {"type": "COMPANY", "normalized": "NVIDIA", "score": 0.95},
            {"type": "COMPANY", "normalized": "TSMC", "score": 0.88},
            {"type": "PRODUCT", "normalized": "HBM4", "score": 0.82},
        ],
        metadata={"bertopic_topic_id": 3},
    )


@pytest.fixture
def minimal_theme(sample_centroid: np.ndarray) -> Theme:
    """A sparsely populated Theme for testing edge cases."""
    return Theme(
        theme_id="theme_minimal123456",
        name="sparse_theme",
        centroid=sample_centroid,
        top_keywords=[],
        top_tickers=[],
        lifecycle_stage="fading",
        document_count=1,
        description="Short text.",
        top_entities=[],
        metadata={},
    )


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Async mock for Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.incrbyfloat = AsyncMock(return_value="0.01")
    redis.ttl = AsyncMock(return_value=-1)
    redis.expire = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_gpt_score() -> CompellingnessScore:
    """A predictable GPT score for mocking Tier 2."""
    return CompellingnessScore(
        overall_score=7.5,
        dimensions=DimensionScores(
            authority=8.0, evidence=7.0, reasoning=7.5,
            risk=6.5, actionability=8.0, technical=8.0,
        ),
        summary="Strong semiconductor thesis with solid analyst backing",
        tickers=["NVDA"],
        time_horizon="medium-term",
        key_risks=["Export controls", "Competition from AMD"],
        tier_used="gpt",
        model_version="gpt-4o-mini",
    )


@pytest.fixture
def mock_claude_score() -> CompellingnessScore:
    """A predictable Claude score for mocking Tier 3."""
    return CompellingnessScore(
        overall_score=8.0,
        dimensions=DimensionScores(
            authority=8.5, evidence=7.5, reasoning=8.0,
            risk=7.0, actionability=8.5, technical=8.5,
        ),
        summary="Validates GPT assessment with stronger technical depth",
        tickers=["NVDA", "TSMC"],
        time_horizon="medium-term",
        key_risks=["Geopolitical risk", "HBM4 yield uncertainty"],
        tier_used="claude",
        model_version="claude-sonnet-4-5-20250929",
    )
