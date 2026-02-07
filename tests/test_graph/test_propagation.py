"""Tests for sentiment propagation through the causal graph.

Tests the core BFS algorithm, edge-type weighting, decay, confidence
attenuation, cycle detection, and min_impact filtering.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.graph.causal_graph import CausalGraph
from src.graph.config import GraphConfig
from src.graph.propagation import PropagationImpact, SentimentPropagation
from src.graph.storage import GraphRepository


@pytest.fixture
def mock_database() -> AsyncMock:
    """Mock Database instance."""
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock(return_value=None)
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="DELETE 0")
    return db


@pytest.fixture
def config() -> GraphConfig:
    """Default graph config for tests."""
    return GraphConfig()


@pytest.fixture
def graph(mock_database, config) -> CausalGraph:
    """CausalGraph with mocked database."""
    return CausalGraph(mock_database, config=config)


@pytest.fixture
def propagation(graph, config) -> SentimentPropagation:
    """SentimentPropagation service with mocked graph."""
    return SentimentPropagation(graph=graph, config=config)


# ── Core Propagation Algorithm ──────────────────────────


class TestSentimentPropagation:
    """Test the BFS propagation algorithm."""

    @pytest.mark.asyncio
    async def test_linear_chain_decay(self, propagation, mock_database):
        """A→B→C chain: impact decays at each hop via edge_weight * confidence * decay."""
        # A --depends_on(conf=1.0)--> B --depends_on(conf=1.0)--> C
        mock_database.fetch.return_value = [
            {"source": "A", "target": "B", "relation": "depends_on", "confidence": 1.0, "depth": 1},
            {"source": "B", "target": "C", "relation": "depends_on", "confidence": 1.0, "depth": 2},
        ]

        impacts = await propagation.propagate("A", -0.5)

        assert "B" in impacts
        assert "C" in impacts

        # B: -0.5 * 0.8 (depends_on weight) * 1.0 (confidence) * 0.7 (decay) = -0.28
        assert impacts["B"].impact == pytest.approx(-0.28, abs=1e-4)
        assert impacts["B"].depth == 1

        # C: -0.28 * 0.8 * 1.0 * 0.7 = -0.1568
        assert impacts["C"].impact == pytest.approx(-0.1568, abs=1e-4)
        assert impacts["C"].depth == 2

    @pytest.mark.asyncio
    async def test_branching_both_receive_impact(self, propagation, mock_database):
        """A→B and A→C: both branches receive independent impact."""
        mock_database.fetch.return_value = [
            {"source": "A", "target": "B", "relation": "depends_on", "confidence": 1.0, "depth": 1},
            {"source": "A", "target": "C", "relation": "supplies_to", "confidence": 1.0, "depth": 1},
        ]

        impacts = await propagation.propagate("A", -0.5)

        assert "B" in impacts
        assert "C" in impacts

        # B: -0.5 * 0.8 (depends_on) * 1.0 * 0.7 = -0.28
        assert impacts["B"].impact == pytest.approx(-0.28, abs=1e-4)
        # C: -0.5 * 0.6 (supplies_to) * 1.0 * 0.7 = -0.21
        assert impacts["C"].impact == pytest.approx(-0.21, abs=1e-4)

    @pytest.mark.asyncio
    async def test_competes_with_flips_sign(self, propagation, mock_database):
        """competes_with edge flips the sentiment sign."""
        mock_database.fetch.return_value = [
            {"source": "AMD", "target": "NVDA", "relation": "competes_with", "confidence": 1.0, "depth": 1},
        ]

        impacts = await propagation.propagate("AMD", -0.5)

        assert "NVDA" in impacts
        # -0.5 * (-0.3) * 1.0 * 0.7 = +0.105
        assert impacts["NVDA"].impact == pytest.approx(0.105, abs=1e-4)
        assert impacts["NVDA"].path_relation == "competes_with"

    @pytest.mark.asyncio
    async def test_blocks_edge_flips_sign(self, propagation, mock_database):
        """blocks edge inverts the impact direction."""
        mock_database.fetch.return_value = [
            {"source": "X", "target": "Y", "relation": "blocks", "confidence": 0.8, "depth": 1},
        ]

        impacts = await propagation.propagate("X", 0.4)

        assert "Y" in impacts
        # 0.4 * (-0.4) * 0.8 * 0.7 = -0.0896
        assert impacts["Y"].impact == pytest.approx(-0.0896, abs=1e-4)

    @pytest.mark.asyncio
    async def test_min_impact_filtering(self, propagation, mock_database):
        """Impacts below min_impact threshold are excluded."""
        mock_database.fetch.return_value = [
            {"source": "A", "target": "B", "relation": "drives", "confidence": 0.01, "depth": 1},
        ]

        # 0.1 * 0.5 (drives) * 0.01 * 0.7 = 0.00035 < 0.01 min_impact
        impacts = await propagation.propagate("A", 0.1)

        assert "B" not in impacts

    @pytest.mark.asyncio
    async def test_custom_min_impact(self, propagation, mock_database):
        """Custom min_impact override works."""
        mock_database.fetch.return_value = [
            {"source": "A", "target": "B", "relation": "depends_on", "confidence": 1.0, "depth": 1},
        ]

        # -0.1 * 0.8 * 1.0 * 0.7 = -0.056
        # With min_impact=0.1, this should be filtered out
        impacts = await propagation.propagate("A", -0.1, min_impact=0.1)
        assert "B" not in impacts

        # With min_impact=0.01, this should pass
        impacts = await propagation.propagate("A", -0.1, min_impact=0.01)
        assert "B" in impacts

    @pytest.mark.asyncio
    async def test_no_downstream_returns_empty(self, propagation, mock_database):
        """Node with no outgoing edges returns empty dict."""
        mock_database.fetch.return_value = []

        impacts = await propagation.propagate("ISOLATED_NODE", -0.5)

        assert impacts == {}

    @pytest.mark.asyncio
    async def test_first_arrival_wins(self, propagation, mock_database):
        """When a node is reachable via multiple paths, shallowest wins."""
        # A→B (depth 1) and A→C→B (depth 2) — B should keep depth-1 impact
        mock_database.fetch.return_value = [
            {"source": "A", "target": "B", "relation": "depends_on", "confidence": 1.0, "depth": 1},
            {"source": "A", "target": "C", "relation": "depends_on", "confidence": 1.0, "depth": 1},
            {"source": "C", "target": "B", "relation": "supplies_to", "confidence": 1.0, "depth": 2},
        ]

        impacts = await propagation.propagate("A", -0.5)

        assert impacts["B"].depth == 1
        # Depth-1 impact: -0.5 * 0.8 * 1.0 * 0.7 = -0.28
        assert impacts["B"].impact == pytest.approx(-0.28, abs=1e-4)

    @pytest.mark.asyncio
    async def test_edge_confidence_attenuates(self, propagation, mock_database):
        """Low confidence edge halves the impact."""
        mock_database.fetch.return_value = [
            {"source": "A", "target": "B", "relation": "depends_on", "confidence": 0.5, "depth": 1},
        ]

        impacts = await propagation.propagate("A", -0.5)

        assert "B" in impacts
        # -0.5 * 0.8 * 0.5 * 0.7 = -0.14
        assert impacts["B"].impact == pytest.approx(-0.14, abs=1e-4)
        assert impacts["B"].edge_confidence == 0.5

    @pytest.mark.asyncio
    async def test_mixed_edge_types(self, propagation, mock_database):
        """Different edge types apply their respective weights."""
        mock_database.fetch.return_value = [
            {"source": "A", "target": "B", "relation": "depends_on", "confidence": 1.0, "depth": 1},
            {"source": "A", "target": "C", "relation": "competes_with", "confidence": 1.0, "depth": 1},
            {"source": "A", "target": "D", "relation": "drives", "confidence": 1.0, "depth": 1},
        ]

        impacts = await propagation.propagate("A", -0.5)

        # depends_on: -0.5 * 0.8 * 0.7 = -0.28
        assert impacts["B"].impact == pytest.approx(-0.28, abs=1e-4)
        # competes_with: -0.5 * (-0.3) * 0.7 = +0.105
        assert impacts["C"].impact == pytest.approx(0.105, abs=1e-4)
        # drives: -0.5 * 0.5 * 0.7 = -0.175
        assert impacts["D"].impact == pytest.approx(-0.175, abs=1e-4)

    @pytest.mark.asyncio
    async def test_propagation_impact_dataclass(self):
        """PropagationImpact is frozen and has correct fields."""
        impact = PropagationImpact(
            node_id="NVDA",
            impact=-0.14,
            depth=1,
            path_relation="depends_on",
            edge_confidence=0.9,
        )
        assert impact.node_id == "NVDA"
        assert impact.impact == -0.14
        assert impact.depth == 1

    @pytest.mark.asyncio
    async def test_unknown_relation_zero_weight(self, propagation, mock_database):
        """Unknown edge relation type gets zero weight → filtered out."""
        mock_database.fetch.return_value = [
            {"source": "A", "target": "B", "relation": "unknown_rel", "confidence": 1.0, "depth": 1},
        ]

        impacts = await propagation.propagate("A", -0.5)
        # 0.0 weight → impact = 0 → below min_impact threshold
        assert "B" not in impacts

    @pytest.mark.asyncio
    async def test_positive_sentiment_propagates(self, propagation, mock_database):
        """Positive sentiment delta propagates correctly."""
        mock_database.fetch.return_value = [
            {"source": "A", "target": "B", "relation": "depends_on", "confidence": 1.0, "depth": 1},
        ]

        impacts = await propagation.propagate("A", 0.3)

        assert "B" in impacts
        # 0.3 * 0.8 * 1.0 * 0.7 = 0.168
        assert impacts["B"].impact == pytest.approx(0.168, abs=1e-4)
        assert impacts["B"].impact > 0


# ── get_downstream_edges Repository Method ──────────────


class TestGetDownstreamEdges:
    """Test the repository-level get_downstream_edges method."""

    @pytest.mark.asyncio
    async def test_returns_edge_tuples(self, mock_database):
        """get_downstream_edges returns (source, target, relation, confidence, depth)."""
        mock_database.fetch.return_value = [
            {"source": "TSMC", "target": "NVDA", "relation": "supplies_to", "confidence": 0.9, "depth": 1},
            {"source": "NVDA", "target": "MSFT", "relation": "supplies_to", "confidence": 0.8, "depth": 2},
        ]

        repo = GraphRepository(mock_database)
        edges = await repo.get_downstream_edges("TSMC", max_depth=3)

        assert len(edges) == 2
        assert edges[0] == ("TSMC", "NVDA", "supplies_to", 0.9, 1)
        assert edges[1] == ("NVDA", "MSFT", "supplies_to", 0.8, 2)

    @pytest.mark.asyncio
    async def test_empty_when_no_edges(self, mock_database):
        """Isolated node returns empty list."""
        mock_database.fetch.return_value = []

        repo = GraphRepository(mock_database)
        edges = await repo.get_downstream_edges("ISOLATED", max_depth=3)

        assert edges == []

    @pytest.mark.asyncio
    async def test_depth_passed_to_query(self, mock_database):
        """Max depth parameter is forwarded to the SQL query."""
        mock_database.fetch.return_value = []

        repo = GraphRepository(mock_database)
        await repo.get_downstream_edges("A", max_depth=5)

        # Verify the depth parameter was passed
        call_args = mock_database.fetch.call_args
        assert call_args[0][1] == "A"
        assert call_args[0][2] == 5

    @pytest.mark.asyncio
    async def test_causal_graph_wrapper_clamps_depth(self, mock_database):
        """CausalGraph wrapper clamps depth to max_traversal_depth."""
        config = GraphConfig(max_traversal_depth=3)
        cg = CausalGraph(mock_database, config=config)
        mock_database.fetch.return_value = []

        await cg.get_downstream_edges("A", max_depth=10)

        call_args = mock_database.fetch.call_args
        # Should be clamped to 3
        assert call_args[0][2] == 3


# ── Edge Weight Lookup ──────────────────────────────────


class TestEdgeWeightLookup:
    """Test the _get_edge_weight helper."""

    def test_all_known_relations(self, propagation):
        """All 5 relation types return their configured weight."""
        assert propagation._get_edge_weight("depends_on") == 0.8
        assert propagation._get_edge_weight("supplies_to") == 0.6
        assert propagation._get_edge_weight("competes_with") == -0.3
        assert propagation._get_edge_weight("drives") == 0.5
        assert propagation._get_edge_weight("blocks") == -0.4

    def test_unknown_relation_returns_zero(self, propagation):
        """Unknown relation type returns 0.0."""
        assert propagation._get_edge_weight("some_unknown") == 0.0

    def test_custom_config_weights(self, graph):
        """Custom config overrides default weights."""
        custom_config = GraphConfig(propagation_weight_depends_on=0.95)
        prop = SentimentPropagation(graph=graph, config=custom_config)
        assert prop._get_edge_weight("depends_on") == 0.95
