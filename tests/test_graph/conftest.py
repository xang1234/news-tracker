"""Pytest fixtures for causal graph tests."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.graph.schemas import CausalEdge, CausalNode


@pytest.fixture
def mock_database() -> AsyncMock:
    """Mock Database instance matching the Database API."""
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock(return_value=None)
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="DELETE 0")
    return db


@pytest.fixture
def sample_node() -> CausalNode:
    """A sample ticker node."""
    return CausalNode(
        node_id="NVDA",
        node_type="ticker",
        name="NVIDIA Corporation",
        metadata={"sector": "semiconductors"},
    )


@pytest.fixture
def sample_node_row() -> dict:
    """Dict mimicking an asyncpg Record for a node."""
    return {
        "node_id": "NVDA",
        "node_type": "ticker",
        "name": "NVIDIA Corporation",
        "metadata": '{"sector": "semiconductors"}',
        "created_at": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2025, 1, 16, 8, 30, 0, tzinfo=timezone.utc),
    }


@pytest.fixture
def sample_edge() -> CausalEdge:
    """A sample edge."""
    return CausalEdge(
        source="TSMC",
        target="NVDA",
        relation="supplies_to",
        confidence=0.9,
        source_doc_ids=["doc_001"],
    )


@pytest.fixture
def sample_edge_row() -> dict:
    """Dict mimicking an asyncpg Record for an edge."""
    return {
        "source": "TSMC",
        "target": "NVDA",
        "relation": "supplies_to",
        "confidence": 0.9,
        "source_doc_ids": ["doc_001"],
        "created_at": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        "metadata": "{}",
    }
