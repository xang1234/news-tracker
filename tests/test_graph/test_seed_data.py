"""Tests for the causal graph seed data module."""

from unittest.mock import AsyncMock, call

import pytest

from src.graph.schemas import VALID_NODE_TYPES, VALID_RELATION_TYPES
from src.graph.seed_data import (
    ALL_EDGES,
    ALL_NODES,
    COMPETITION_EDGES,
    DEMAND_DRIVER_EDGES,
    EDA_SUPPLY_EDGES,
    EQUIPMENT_SUPPLY_EDGES,
    FOUNDRY_SUPPLY_EDGES,
    MEMORY_SUPPLY_EDGES,
    SEED_VERSION,
    TECHNOLOGY_EDGES,
    TECHNOLOGY_NODES,
    THEME_NODES,
    TICKER_NODES,
    seed_graph,
)


class TestSeedDataIntegrity:
    """Validate seed data definitions are consistent and complete."""

    def test_at_least_100_edges(self) -> None:
        """Task requires ~100 key relationships."""
        assert len(ALL_EDGES) >= 100

    def test_all_node_types_valid(self) -> None:
        """Every node uses a valid node_type."""
        for node in ALL_NODES:
            assert node.node_type in VALID_NODE_TYPES, (
                f"Node {node.node_id} has invalid type {node.node_type!r}"
            )

    def test_all_edge_relations_valid(self) -> None:
        """Every edge uses a valid relation type."""
        for edge in ALL_EDGES:
            assert edge.relation in VALID_RELATION_TYPES, (
                f"Edge ({edge.source}, {edge.target}) has invalid "
                f"relation {edge.relation!r}"
            )

    def test_all_edge_confidence_in_range(self) -> None:
        """Every edge confidence is between 0.0 and 1.0."""
        for edge in ALL_EDGES:
            assert 0.0 <= edge.confidence <= 1.0, (
                f"Edge ({edge.source}, {edge.target}, {edge.relation}) "
                f"has confidence {edge.confidence} out of range"
            )

    def test_no_self_loops(self) -> None:
        """No edge connects a node to itself."""
        for edge in ALL_EDGES:
            assert edge.source != edge.target, (
                f"Self-loop found: {edge.source} → {edge.source} "
                f"({edge.relation})"
            )

    def test_node_ids_unique(self) -> None:
        """All node IDs are unique."""
        ids = [n.node_id for n in ALL_NODES]
        assert len(ids) == len(set(ids))

    def test_edge_sources_reference_defined_nodes(self) -> None:
        """Every edge source references a defined node."""
        node_ids = {n.node_id for n in ALL_NODES}
        for edge in ALL_EDGES:
            assert edge.source in node_ids, (
                f"Edge source {edge.source!r} not in defined nodes"
            )

    def test_edge_targets_reference_defined_nodes(self) -> None:
        """Every edge target references a defined node."""
        node_ids = {n.node_id for n in ALL_NODES}
        for edge in ALL_EDGES:
            assert edge.target in node_ids, (
                f"Edge target {edge.target!r} not in defined nodes"
            )

    def test_no_duplicate_edges(self) -> None:
        """No duplicate (source, target, relation) triples."""
        seen = set()
        for edge in ALL_EDGES:
            key = (edge.source, edge.target, edge.relation)
            assert key not in seen, f"Duplicate edge: {key}"
            seen.add(key)

    def test_seed_version_is_positive_int(self) -> None:
        """SEED_VERSION is a positive integer."""
        assert isinstance(SEED_VERSION, int)
        assert SEED_VERSION >= 1


class TestSeedDataCoverage:
    """Validate domain coverage of the seed data."""

    def test_ticker_nodes_cover_major_segments(self) -> None:
        """Ticker nodes include GPU, foundry, memory, equipment, and EDA."""
        ids = {n.node_id for n in TICKER_NODES}
        # GPU / AI
        assert "NVDA" in ids
        assert "AMD" in ids
        assert "INTC" in ids
        # Foundry
        assert "TSM" in ids
        assert "SAMSUNG" in ids
        # Memory
        assert "SK_HYNIX" in ids
        assert "MU" in ids
        # Equipment
        assert "ASML" in ids
        assert "AMAT" in ids
        assert "LRCX" in ids
        # EDA / IP
        assert "SNPS" in ids
        assert "CDNS" in ids
        assert "ARM" in ids

    def test_technology_nodes_present(self) -> None:
        """Key semiconductor technologies are represented."""
        ids = {n.node_id for n in TECHNOLOGY_NODES}
        assert "EUV" in ids
        assert "HBM3E" in ids
        assert "CoWoS" in ids

    def test_theme_nodes_present(self) -> None:
        """Key demand themes are represented."""
        ids = {n.node_id for n in THEME_NODES}
        assert "theme_ai_training" in ids
        assert "theme_hbm_demand" in ids
        assert "theme_automotive_chips" in ids

    def test_competition_edges_are_bidirectional(self) -> None:
        """Competition edges have both A→B and B→A."""
        pairs = set()
        for edge in COMPETITION_EDGES:
            pairs.add((edge.source, edge.target))
        for edge in COMPETITION_EDGES:
            reverse = (edge.target, edge.source)
            assert reverse in pairs, (
                f"Competition edge ({edge.source}, {edge.target}) "
                f"is missing reverse direction"
            )

    def test_supply_chain_categories_non_empty(self) -> None:
        """Each supply chain category has edges."""
        assert len(FOUNDRY_SUPPLY_EDGES) > 0
        assert len(EQUIPMENT_SUPPLY_EDGES) > 0
        assert len(MEMORY_SUPPLY_EDGES) > 0
        assert len(EDA_SUPPLY_EDGES) > 0
        assert len(TECHNOLOGY_EDGES) > 0
        assert len(DEMAND_DRIVER_EDGES) > 0

    def test_tsmc_has_supply_edges(self) -> None:
        """TSMC (TSM) is a major supplier — should have multiple supplies_to."""
        tsm_supply = [
            e for e in ALL_EDGES
            if e.source == "TSM" and e.relation == "supplies_to"
        ]
        assert len(tsm_supply) >= 5

    def test_nvidia_has_upstream_dependencies(self) -> None:
        """NVIDIA receives supplies from foundries, memory, and EDA."""
        nvda_targets = [
            e for e in ALL_EDGES
            if e.target == "NVDA" and e.relation == "supplies_to"
        ]
        sources = {e.source for e in nvda_targets}
        # Should have foundry, memory, and EDA suppliers
        assert "TSM" in sources
        assert "SK_HYNIX" in sources
        assert "SNPS" in sources or "CDNS" in sources


class TestSeedGraphFunction:
    """Test the seed_graph() async function."""

    @pytest.mark.asyncio
    async def test_returns_summary(self, mock_database: AsyncMock) -> None:
        """seed_graph() returns a summary dict with counts."""
        # Mock for ensure_node upsert (fetchrow)
        mock_database.fetchrow.return_value = {
            "node_id": "X",
            "node_type": "ticker",
            "name": "X",
            "metadata": "{}",
            "created_at": None,
            "updated_at": None,
        }

        result = await seed_graph(mock_database)

        assert result["seed_version"] == SEED_VERSION
        assert result["node_count"] == len(ALL_NODES)
        assert result["edge_count"] == len(ALL_EDGES)

    @pytest.mark.asyncio
    async def test_creates_all_nodes(self, mock_database: AsyncMock) -> None:
        """seed_graph() calls ensure_node for every defined node."""
        mock_database.fetchrow.return_value = {
            "node_id": "X",
            "node_type": "ticker",
            "name": "X",
            "metadata": "{}",
            "created_at": None,
            "updated_at": None,
        }

        await seed_graph(mock_database)

        # Each node triggers a fetchrow (upsert_node)
        assert mock_database.fetchrow.call_count == len(ALL_NODES)

    @pytest.mark.asyncio
    async def test_creates_all_edges(self, mock_database: AsyncMock) -> None:
        """seed_graph() calls add_edge for every defined edge."""
        mock_database.fetchrow.return_value = {
            "node_id": "X",
            "node_type": "ticker",
            "name": "X",
            "metadata": "{}",
            "created_at": None,
            "updated_at": None,
        }

        await seed_graph(mock_database)

        # Each edge triggers an execute (add_edge)
        assert mock_database.execute.call_count == len(ALL_EDGES)

    @pytest.mark.asyncio
    async def test_idempotent_uses_upsert(
        self, mock_database: AsyncMock
    ) -> None:
        """seed_graph() uses ON CONFLICT for idempotency."""
        mock_database.fetchrow.return_value = {
            "node_id": "X",
            "node_type": "ticker",
            "name": "X",
            "metadata": "{}",
            "created_at": None,
            "updated_at": None,
        }

        await seed_graph(mock_database)

        # Node upsert uses ON CONFLICT
        node_sql = mock_database.fetchrow.call_args_list[0][0][0]
        assert "ON CONFLICT" in node_sql

        # Edge upsert uses ON CONFLICT
        edge_sql = mock_database.execute.call_args_list[0][0][0]
        assert "ON CONFLICT" in edge_sql
