"""Tests for /graph/* endpoints."""

import pytest
from unittest.mock import MagicMock


class _MockNode:
    """Simple mock for GraphNode dataclass."""
    def __init__(self, node_id, node_type="ticker", name="NVDA", metadata=None):
        self.node_id = node_id
        self.node_type = node_type
        self.name = name
        self.metadata = metadata or {}


class TestGraphNodesRoute:
    """Test GET /graph/nodes."""

    def test_list_nodes_empty(self, client, mock_graph_repo):
        """GET /graph/nodes returns empty list when no nodes."""
        resp = client.get("/graph/nodes")
        assert resp.status_code == 200
        body = resp.json()
        assert body["nodes"] == []
        assert body["total"] == 0

    def test_list_nodes_with_data(self, client, mock_graph_repo):
        """GET /graph/nodes returns nodes."""
        mock_graph_repo.get_all_nodes.return_value = [
            _MockNode("nvda", "ticker", "NVIDIA"),
            _MockNode("amd", "ticker", "AMD"),
        ]

        resp = client.get("/graph/nodes")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["nodes"][0]["node_id"] == "nvda"

    def test_list_nodes_with_type_filter(self, client, mock_graph_repo):
        """GET /graph/nodes?node_type=ticker uses filter."""
        mock_graph_repo.get_all_nodes.return_value = []

        resp = client.get("/graph/nodes?node_type=ticker")
        assert resp.status_code == 200
        mock_graph_repo.get_all_nodes.assert_called_once_with(
            node_type="ticker", limit=200
        )

    def test_list_nodes_error_sanitized(self, client, mock_graph_repo):
        """Error response doesn't leak internal details."""
        mock_graph_repo.get_all_nodes.side_effect = RuntimeError("DB pool exhausted")

        resp = client.get("/graph/nodes")
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Failed to list graph nodes"
        assert "DB pool" not in body["detail"]


class TestGraphSubgraphRoute:
    """Test GET /graph/nodes/{node_id}/subgraph."""

    def test_subgraph_not_found(self, client, mock_graph_repo):
        """GET /graph/nodes/xxx/subgraph returns 404 when node doesn't exist."""
        mock_graph_repo.get_node.return_value = None

        resp = client.get("/graph/nodes/xxx/subgraph")
        assert resp.status_code == 404

    def test_subgraph_happy_path(self, client, mock_graph_repo):
        """GET /graph/nodes/nvda/subgraph returns subgraph."""
        mock_graph_repo.get_node.return_value = _MockNode("nvda")
        mock_graph_repo.get_subgraph.return_value = {
            "nodes": [
                {"node_id": "nvda", "node_type": "ticker", "name": "NVIDIA", "metadata": {}},
                {"node_id": "hbm", "node_type": "technology", "name": "HBM", "metadata": {}},
            ],
            "edges": [
                {"source": "nvda", "target": "hbm", "relation": "uses", "confidence": 0.9},
            ],
        }

        resp = client.get("/graph/nodes/nvda/subgraph?depth=2")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) == 2
        assert len(body["edges"]) == 1
        assert body["center_node"] == "nvda"


class TestGraphPropagateRoute:
    """Test POST /graph/propagate."""

    def test_propagate_happy_path(self, client, mock_propagation_service):
        """POST /graph/propagate returns impact results."""
        mock_impact = MagicMock()
        mock_impact.node_id = "amd"
        mock_impact.impact = 0.35
        mock_impact.depth = 1
        mock_impact.path_relation = "competes_with"
        mock_impact.edge_confidence = 0.8
        mock_propagation_service.propagate.return_value = {"amd": mock_impact}

        resp = client.post("/graph/propagate", json={
            "source_node": "nvda",
            "sentiment_delta": 0.5,
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["source_node"] == "nvda"
        assert body["sentiment_delta"] == 0.5
        assert body["total_affected"] == 1
        assert body["impacts"][0]["node_id"] == "amd"

    def test_propagate_error_sanitized(self, client, mock_propagation_service):
        """Error response doesn't leak internal details."""
        mock_propagation_service.propagate.side_effect = RuntimeError("recursive CTE timeout")

        resp = client.post("/graph/propagate", json={
            "source_node": "nvda",
            "sentiment_delta": 0.3,
        })

        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Failed to propagate sentiment"
        assert "CTE" not in body["detail"]
