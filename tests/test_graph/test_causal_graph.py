"""Tests for the CausalGraph high-level service."""

from unittest.mock import AsyncMock, patch

import pytest

from src.graph.causal_graph import CausalGraph
from src.graph.config import GraphConfig
from src.graph.schemas import CausalNode


class TestEnsureNode:
    """Test CausalGraph.ensure_node()."""

    @pytest.mark.asyncio
    async def test_creates_node_with_defaults(
        self, mock_database: AsyncMock, sample_node_row: dict
    ) -> None:
        """ensure_node() creates a node using defaults for name/metadata."""
        mock_database.fetchrow.return_value = sample_node_row
        graph = CausalGraph(mock_database)

        result = await graph.ensure_node("NVDA")

        assert result.node_id == "NVDA"
        # Verify upsert was called
        sql = mock_database.fetchrow.call_args[0][0]
        assert "INSERT INTO causal_nodes" in sql

    @pytest.mark.asyncio
    async def test_passes_custom_fields(
        self, mock_database: AsyncMock, sample_node_row: dict
    ) -> None:
        """ensure_node() forwards name, node_type, metadata."""
        mock_database.fetchrow.return_value = sample_node_row
        graph = CausalGraph(mock_database)

        await graph.ensure_node(
            "HBM3E",
            node_type="technology",
            name="High Bandwidth Memory 3E",
            metadata={"generation": 3},
        )

        params = mock_database.fetchrow.call_args[0][1:]
        assert params[0] == "HBM3E"
        assert params[1] == "technology"
        assert params[2] == "High Bandwidth Memory 3E"


class TestAddEdge:
    """Test CausalGraph.add_edge()."""

    @pytest.mark.asyncio
    async def test_uses_config_default_confidence(
        self, mock_database: AsyncMock
    ) -> None:
        """add_edge() uses config default_confidence when not specified."""
        config = GraphConfig(default_confidence=0.75)
        graph = CausalGraph(mock_database, config=config)

        await graph.add_edge("A", "B", "drives")

        params = mock_database.execute.call_args[0][1:]
        assert params[3] == 0.75

    @pytest.mark.asyncio
    async def test_explicit_confidence_overrides_config(
        self, mock_database: AsyncMock
    ) -> None:
        """Explicit confidence overrides config default."""
        config = GraphConfig(default_confidence=0.75)
        graph = CausalGraph(mock_database, config=config)

        await graph.add_edge("A", "B", "drives", confidence=0.5)

        params = mock_database.execute.call_args[0][1:]
        assert params[3] == 0.5


class TestTraversalDepthClamping:
    """Test that traversal methods clamp depth to config max."""

    @pytest.mark.asyncio
    async def test_get_downstream_clamps_depth(
        self, mock_database: AsyncMock
    ) -> None:
        """get_downstream() clamps max_depth to config limit."""
        config = GraphConfig(max_traversal_depth=3)
        mock_database.fetch.return_value = []
        graph = CausalGraph(mock_database, config=config)

        await graph.get_downstream("TSMC", max_depth=10)

        # The depth param should be clamped to 3
        params = mock_database.fetch.call_args[0][1:]
        assert params[1] == 3

    @pytest.mark.asyncio
    async def test_get_upstream_clamps_depth(
        self, mock_database: AsyncMock
    ) -> None:
        """get_upstream() clamps max_depth to config limit."""
        config = GraphConfig(max_traversal_depth=3)
        mock_database.fetch.return_value = []
        graph = CausalGraph(mock_database, config=config)

        await graph.get_upstream("NVDA", max_depth=10)

        params = mock_database.fetch.call_args[0][1:]
        assert params[1] == 3

    @pytest.mark.asyncio
    async def test_find_path_clamps_depth(
        self, mock_database: AsyncMock
    ) -> None:
        """find_path() clamps max_depth to config limit."""
        config = GraphConfig(max_traversal_depth=3)
        mock_database.fetchrow.return_value = None
        graph = CausalGraph(mock_database, config=config)

        await graph.find_path("A", "Z", max_depth=10)

        params = mock_database.fetchrow.call_args[0][1:]
        assert params[2] == 3

    @pytest.mark.asyncio
    async def test_get_subgraph_clamps_depth(
        self, mock_database: AsyncMock
    ) -> None:
        """get_subgraph() clamps depth to config limit."""
        config = GraphConfig(max_traversal_depth=2)
        mock_database.fetch.return_value = []
        graph = CausalGraph(mock_database, config=config)

        await graph.get_subgraph("TSMC", depth=10)

        # get_subgraph calls get_downstream and get_upstream with clamped depth
        # The first two fetch calls are the recursive CTEs with (node_id, max_depth)
        cte_calls = mock_database.fetch.call_args_list[:2]
        for call in cte_calls:
            params = call[0][1:]
            assert params[1] == 2


class TestRemoveEdge:
    """Test CausalGraph.remove_edge()."""

    @pytest.mark.asyncio
    async def test_delegates_to_repository(
        self, mock_database: AsyncMock
    ) -> None:
        """remove_edge() delegates to GraphRepository."""
        mock_database.execute.return_value = "DELETE 1"
        graph = CausalGraph(mock_database)

        result = await graph.remove_edge("A", "B", "drives")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(
        self, mock_database: AsyncMock
    ) -> None:
        """remove_edge() returns False when edge doesn't exist."""
        mock_database.execute.return_value = "DELETE 0"
        graph = CausalGraph(mock_database)

        result = await graph.remove_edge("A", "B", "drives")
        assert result is False


class TestRepositoryAccess:
    """Test CausalGraph.repository property."""

    def test_exposes_repository(self, mock_database: AsyncMock) -> None:
        """repository property provides access to the underlying repo."""
        graph = CausalGraph(mock_database)
        repo = graph.repository
        assert repo is not None
        assert repo._db is mock_database
