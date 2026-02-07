"""Tests for causal graph schema validation."""

import pytest

from src.graph.schemas import (
    VALID_NODE_TYPES,
    VALID_RELATION_TYPES,
    CausalEdge,
    CausalNode,
)


class TestCausalNode:
    """Test CausalNode dataclass."""

    def test_valid_node_types(self) -> None:
        """All valid node types are accepted."""
        for node_type in VALID_NODE_TYPES:
            node = CausalNode(node_id="x", node_type=node_type, name="X")
            assert node.node_type == node_type

    def test_invalid_node_type_raises(self) -> None:
        """Invalid node_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid node_type"):
            CausalNode(node_id="x", node_type="invalid", name="X")

    def test_equality_by_id(self) -> None:
        """Two nodes with the same ID are equal."""
        a = CausalNode(node_id="NVDA", node_type="ticker", name="NVIDIA")
        b = CausalNode(node_id="NVDA", node_type="ticker", name="Different")
        assert a == b

    def test_inequality(self) -> None:
        """Two nodes with different IDs are not equal."""
        a = CausalNode(node_id="NVDA", node_type="ticker", name="NVIDIA")
        b = CausalNode(node_id="AMD", node_type="ticker", name="AMD")
        assert a != b

    def test_hash_by_id(self) -> None:
        """Hash is based on node_id, enabling set/dict usage."""
        a = CausalNode(node_id="NVDA", node_type="ticker", name="NVIDIA")
        b = CausalNode(node_id="NVDA", node_type="ticker", name="Different")
        assert {a, b} == {a}

    def test_default_metadata(self) -> None:
        """Metadata defaults to empty dict."""
        node = CausalNode(node_id="x", node_type="ticker", name="X")
        assert node.metadata == {}

    def test_not_equal_to_other_types(self) -> None:
        """Comparison with non-CausalNode returns NotImplemented."""
        node = CausalNode(node_id="x", node_type="ticker", name="X")
        assert node != "x"


class TestCausalEdge:
    """Test CausalEdge dataclass."""

    def test_valid_relation_types(self) -> None:
        """All valid relation types are accepted."""
        for relation in VALID_RELATION_TYPES:
            edge = CausalEdge(source="A", target="B", relation=relation)
            assert edge.relation == relation

    def test_invalid_relation_raises(self) -> None:
        """Invalid relation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid relation"):
            CausalEdge(source="A", target="B", relation="invalid")

    def test_confidence_bounds(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        CausalEdge(source="A", target="B", relation="drives", confidence=0.0)
        CausalEdge(source="A", target="B", relation="drives", confidence=1.0)

        with pytest.raises(ValueError, match="confidence"):
            CausalEdge(source="A", target="B", relation="drives", confidence=-0.1)

        with pytest.raises(ValueError, match="confidence"):
            CausalEdge(source="A", target="B", relation="drives", confidence=1.1)

    def test_default_confidence(self) -> None:
        """Default confidence is 1.0."""
        edge = CausalEdge(source="A", target="B", relation="drives")
        assert edge.confidence == 1.0

    def test_default_source_doc_ids(self) -> None:
        """Default source_doc_ids is empty list."""
        edge = CausalEdge(source="A", target="B", relation="drives")
        assert edge.source_doc_ids == []
