"""Tests for GraphRepository CRUD and traversal operations."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.graph.schemas import CausalEdge, CausalNode
from src.graph.storage import GraphRepository, _row_to_edge, _row_to_node


class TestRowConversions:
    """Test module-level row conversion helpers."""

    def test_row_to_node_json_string_metadata(
        self, sample_node_row: dict
    ) -> None:
        """_row_to_node parses JSON string metadata."""
        node = _row_to_node(sample_node_row)
        assert node.node_id == "NVDA"
        assert node.node_type == "ticker"
        assert node.name == "NVIDIA Corporation"
        assert node.metadata == {"sector": "semiconductors"}

    def test_row_to_node_dict_metadata(self) -> None:
        """_row_to_node handles already-parsed dict metadata."""
        row = {
            "node_id": "AMD",
            "node_type": "ticker",
            "name": "AMD",
            "metadata": {"sector": "semiconductors"},
            "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        }
        node = _row_to_node(row)
        assert node.metadata == {"sector": "semiconductors"}

    def test_row_to_edge(self, sample_edge_row: dict) -> None:
        """_row_to_edge correctly parses all fields."""
        edge = _row_to_edge(sample_edge_row)
        assert edge.source == "TSMC"
        assert edge.target == "NVDA"
        assert edge.relation == "supplies_to"
        assert edge.confidence == 0.9
        assert edge.source_doc_ids == ["doc_001"]
        assert edge.metadata == {}

    def test_row_to_edge_dict_metadata(self) -> None:
        """_row_to_edge handles already-parsed dict metadata."""
        row = {
            "source": "A",
            "target": "B",
            "relation": "drives",
            "confidence": 1.0,
            "source_doc_ids": [],
            "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "metadata": {"note": "test"},
        }
        edge = _row_to_edge(row)
        assert edge.metadata == {"note": "test"}


class TestUpsertNode:
    """Test GraphRepository.upsert_node()."""

    @pytest.mark.asyncio
    async def test_insert_sql_and_params(
        self,
        mock_database: AsyncMock,
        sample_node: CausalNode,
        sample_node_row: dict,
    ) -> None:
        """upsert_node() passes correct SQL and params."""
        mock_database.fetchrow.return_value = sample_node_row
        repo = GraphRepository(mock_database)

        result = await repo.upsert_node(sample_node)

        args = mock_database.fetchrow.call_args
        sql = args[0][0]
        assert "INSERT INTO causal_nodes" in sql
        assert "ON CONFLICT (node_id) DO UPDATE" in sql
        assert "RETURNING *" in sql

        params = args[0][1:]
        assert params[0] == "NVDA"
        assert params[1] == "ticker"
        assert params[2] == "NVIDIA Corporation"
        assert json.loads(params[3]) == {"sector": "semiconductors"}

        assert result.node_id == "NVDA"


class TestGetNode:
    """Test GraphRepository.get_node()."""

    @pytest.mark.asyncio
    async def test_found(
        self,
        mock_database: AsyncMock,
        sample_node_row: dict,
    ) -> None:
        """get_node() returns CausalNode when found."""
        mock_database.fetchrow.return_value = sample_node_row
        repo = GraphRepository(mock_database)

        result = await repo.get_node("NVDA")
        assert result is not None
        assert result.node_id == "NVDA"

    @pytest.mark.asyncio
    async def test_not_found(self, mock_database: AsyncMock) -> None:
        """get_node() returns None when not found."""
        mock_database.fetchrow.return_value = None
        repo = GraphRepository(mock_database)

        result = await repo.get_node("NONEXISTENT")
        assert result is None


class TestDeleteNode:
    """Test GraphRepository.delete_node()."""

    @pytest.mark.asyncio
    async def test_deleted(self, mock_database: AsyncMock) -> None:
        """delete_node() returns True when a node was deleted."""
        mock_database.execute.return_value = "DELETE 1"
        repo = GraphRepository(mock_database)

        assert await repo.delete_node("NVDA") is True

    @pytest.mark.asyncio
    async def test_not_found(self, mock_database: AsyncMock) -> None:
        """delete_node() returns False when node doesn't exist."""
        mock_database.execute.return_value = "DELETE 0"
        repo = GraphRepository(mock_database)

        assert await repo.delete_node("NONEXISTENT") is False


class TestAddEdge:
    """Test GraphRepository.add_edge()."""

    @pytest.mark.asyncio
    async def test_insert_sql(self, mock_database: AsyncMock) -> None:
        """add_edge() uses ON CONFLICT with source_doc_ids merge."""
        repo = GraphRepository(mock_database)

        await repo.add_edge(
            source="TSMC",
            target="NVDA",
            relation="supplies_to",
            confidence=0.9,
            source_doc_ids=["doc_001"],
        )

        args = mock_database.execute.call_args
        sql = args[0][0]
        assert "INSERT INTO causal_edges" in sql
        assert "ON CONFLICT (source, target, relation) DO UPDATE" in sql
        assert "DISTINCT unnest" in sql

        params = args[0][1:]
        assert params[0] == "TSMC"
        assert params[1] == "NVDA"
        assert params[2] == "supplies_to"
        assert params[3] == 0.9
        assert params[4] == ["doc_001"]

    @pytest.mark.asyncio
    async def test_default_empty_lists(self, mock_database: AsyncMock) -> None:
        """add_edge() defaults to empty source_doc_ids and metadata."""
        repo = GraphRepository(mock_database)

        await repo.add_edge("A", "B", "drives")

        params = mock_database.execute.call_args[0][1:]
        assert params[4] == []  # source_doc_ids
        assert json.loads(params[5]) == {}  # metadata


class TestRemoveEdge:
    """Test GraphRepository.remove_edge()."""

    @pytest.mark.asyncio
    async def test_deleted(self, mock_database: AsyncMock) -> None:
        """remove_edge() returns True when edge existed."""
        mock_database.execute.return_value = "DELETE 1"
        repo = GraphRepository(mock_database)

        assert await repo.remove_edge("A", "B", "drives") is True

    @pytest.mark.asyncio
    async def test_not_found(self, mock_database: AsyncMock) -> None:
        """remove_edge() returns False when edge didn't exist."""
        mock_database.execute.return_value = "DELETE 0"
        repo = GraphRepository(mock_database)

        assert await repo.remove_edge("A", "B", "drives") is False


class TestGetDownstream:
    """Test GraphRepository.get_downstream()."""

    @pytest.mark.asyncio
    async def test_returns_node_depth_tuples(
        self, mock_database: AsyncMock
    ) -> None:
        """get_downstream() returns (node_id, depth) tuples from CTE."""
        mock_database.fetch.return_value = [
            {"node_id": "NVDA", "depth": 1},
            {"node_id": "AAPL", "depth": 2},
        ]
        repo = GraphRepository(mock_database)

        result = await repo.get_downstream("TSMC", max_depth=2)

        assert result == [("NVDA", 1), ("AAPL", 2)]

        # Verify recursive CTE is used
        sql = mock_database.fetch.call_args[0][0]
        assert "WITH RECURSIVE" in sql
        assert "NOT" in sql  # cycle detection

    @pytest.mark.asyncio
    async def test_empty_for_leaf_node(
        self, mock_database: AsyncMock
    ) -> None:
        """get_downstream() returns empty list for a leaf node."""
        mock_database.fetch.return_value = []
        repo = GraphRepository(mock_database)

        result = await repo.get_downstream("LEAF")
        assert result == []


class TestGetUpstream:
    """Test GraphRepository.get_upstream()."""

    @pytest.mark.asyncio
    async def test_returns_node_depth_tuples(
        self, mock_database: AsyncMock
    ) -> None:
        """get_upstream() follows edges in reverse direction."""
        mock_database.fetch.return_value = [
            {"node_id": "ASML", "depth": 1},
            {"node_id": "TSMC", "depth": 2},
        ]
        repo = GraphRepository(mock_database)

        result = await repo.get_upstream("NVDA", max_depth=2)

        assert result == [("ASML", 1), ("TSMC", 2)]

        # Verify reverse traversal (target → source)
        sql = mock_database.fetch.call_args[0][0]
        assert "WITH RECURSIVE" in sql


class TestGetNeighbors:
    """Test GraphRepository.get_neighbors()."""

    @pytest.mark.asyncio
    async def test_without_relation_filter(
        self, mock_database: AsyncMock
    ) -> None:
        """get_neighbors() returns all neighbors when no filter."""
        mock_database.fetch.return_value = [
            {"neighbor": "TSMC", "relation": "supplies_to"},
            {"neighbor": "AMD", "relation": "competes_with"},
        ]
        repo = GraphRepository(mock_database)

        result = await repo.get_neighbors("NVDA")

        assert result == [("TSMC", "supplies_to"), ("AMD", "competes_with")]

        # Both directions queried (UNION ALL)
        sql = mock_database.fetch.call_args[0][0]
        assert "UNION ALL" in sql

    @pytest.mark.asyncio
    async def test_with_relation_filter(
        self, mock_database: AsyncMock
    ) -> None:
        """get_neighbors() filters by relation types."""
        mock_database.fetch.return_value = [
            {"neighbor": "AMD", "relation": "competes_with"},
        ]
        repo = GraphRepository(mock_database)

        result = await repo.get_neighbors(
            "NVDA", relations=["competes_with"]
        )

        assert result == [("AMD", "competes_with")]

        # Verify ANY($2) filter is present
        sql = mock_database.fetch.call_args[0][0]
        assert "ANY($2)" in sql


class TestFindPath:
    """Test GraphRepository.find_path()."""

    @pytest.mark.asyncio
    async def test_path_found(self, mock_database: AsyncMock) -> None:
        """find_path() returns shortest path when one exists."""
        mock_database.fetchrow.return_value = {
            "path": ["ASML", "TSMC", "NVDA"]
        }
        repo = GraphRepository(mock_database)

        result = await repo.find_path("ASML", "NVDA", max_depth=3)

        assert result == ["ASML", "TSMC", "NVDA"]

        sql = mock_database.fetchrow.call_args[0][0]
        assert "WITH RECURSIVE" in sql
        assert "ORDER BY depth" in sql
        assert "LIMIT 1" in sql

    @pytest.mark.asyncio
    async def test_no_path(self, mock_database: AsyncMock) -> None:
        """find_path() returns None when no path exists."""
        mock_database.fetchrow.return_value = None
        repo = GraphRepository(mock_database)

        result = await repo.find_path("A", "Z")
        assert result is None


class TestGetSubgraph:
    """Test GraphRepository.get_subgraph()."""

    @pytest.mark.asyncio
    async def test_subgraph_structure(
        self, mock_database: AsyncMock
    ) -> None:
        """get_subgraph() returns dict with nodes and edges."""
        # Mock downstream and upstream CTEs to return some node IDs
        mock_database.fetch.side_effect = [
            # get_downstream CTE
            [{"node_id": "NVDA", "depth": 1}],
            # get_upstream CTE
            [{"node_id": "ASML", "depth": 1}],
            # Node fetch: SELECT * FROM causal_nodes WHERE node_id = ANY($1)
            [
                {
                    "node_id": "TSMC",
                    "node_type": "ticker",
                    "name": "TSMC",
                    "metadata": "{}",
                    "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                },
                {
                    "node_id": "NVDA",
                    "node_type": "ticker",
                    "name": "NVIDIA",
                    "metadata": "{}",
                    "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                },
                {
                    "node_id": "ASML",
                    "node_type": "ticker",
                    "name": "ASML",
                    "metadata": "{}",
                    "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                },
            ],
            # Edge fetch: SELECT * FROM causal_edges WHERE source/target ANY
            [
                {
                    "source": "TSMC",
                    "target": "NVDA",
                    "relation": "supplies_to",
                    "confidence": 0.9,
                    "source_doc_ids": [],
                    "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "metadata": "{}",
                },
                {
                    "source": "ASML",
                    "target": "TSMC",
                    "relation": "supplies_to",
                    "confidence": 0.85,
                    "source_doc_ids": [],
                    "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
                    "metadata": "{}",
                },
            ],
        ]

        repo = GraphRepository(mock_database)
        result = await repo.get_subgraph("TSMC", depth=1)

        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 2

        node_ids = {n["node_id"] for n in result["nodes"]}
        assert node_ids == {"TSMC", "NVDA", "ASML"}

        edge_pairs = {(e["source"], e["target"]) for e in result["edges"]}
        assert ("TSMC", "NVDA") in edge_pairs
        assert ("ASML", "TSMC") in edge_pairs
