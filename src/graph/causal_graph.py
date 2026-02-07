"""High-level causal graph service.

Thin orchestration layer over GraphRepository, providing the public API
described in the task spec. Ensures nodes exist before edge operations
and encapsulates config defaults.
"""

import logging
from typing import Any

from src.storage.database import Database

from .config import GraphConfig
from .schemas import CausalNode, NodeType
from .storage import GraphRepository

logger = logging.getLogger(__name__)


class CausalGraph:
    """Causal graph service for modeling semiconductor supply chain relationships.

    Wraps GraphRepository with auto-node-creation and config-driven defaults.

    Usage:
        graph = CausalGraph(database)
        await graph.add_edge("TSMC", "NVDA", "supplies_to", confidence=0.9)
        downstream = await graph.get_downstream("TSMC", max_depth=3)
    """

    def __init__(
        self,
        database: Database,
        config: GraphConfig | None = None,
    ) -> None:
        self._repo = GraphRepository(database)
        self._config = config or GraphConfig()

    @property
    def repository(self) -> GraphRepository:
        """Access the underlying repository for direct node/edge CRUD."""
        return self._repo

    async def ensure_node(
        self,
        node_id: str,
        node_type: NodeType = "ticker",
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CausalNode:
        """Ensure a node exists, creating it if necessary."""
        node = CausalNode(
            node_id=node_id,
            node_type=node_type,
            name=name or node_id,
            metadata=metadata or {},
        )
        return await self._repo.upsert_node(node)

    async def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        confidence: float | None = None,
        source_doc_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an edge between two existing nodes.

        Args:
            source: Source node ID (must exist).
            target: Target node ID (must exist).
            relation: One of depends_on, supplies_to, competes_with, drives, blocks.
            confidence: Edge confidence (0.0-1.0). Uses config default if None.
            source_doc_ids: Document IDs supporting this edge.
            metadata: Optional metadata dict.
        """
        conf = confidence if confidence is not None else self._config.default_confidence
        await self._repo.add_edge(
            source=source,
            target=target,
            relation=relation,
            confidence=conf,
            source_doc_ids=source_doc_ids,
            metadata=metadata,
        )

    async def remove_edge(
        self, source: str, target: str, relation: str
    ) -> bool:
        """Remove an edge. Returns True if an edge was actually deleted."""
        return await self._repo.remove_edge(source, target, relation)

    async def get_downstream(
        self, node_id: str, max_depth: int = 2
    ) -> list[tuple[str, int]]:
        """Get all nodes reachable by following outgoing edges.

        Returns:
            List of (node_id, depth) tuples.
        """
        depth = min(max_depth, self._config.max_traversal_depth)
        return await self._repo.get_downstream(node_id, max_depth=depth)

    async def get_upstream(
        self, node_id: str, max_depth: int = 2
    ) -> list[tuple[str, int]]:
        """Get all nodes that can reach this node via outgoing edges.

        Returns:
            List of (node_id, depth) tuples.
        """
        depth = min(max_depth, self._config.max_traversal_depth)
        return await self._repo.get_upstream(node_id, max_depth=depth)

    async def get_neighbors(
        self,
        node_id: str,
        relations: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Get immediate neighbors with their relation type.

        Returns:
            List of (neighbor_node_id, relation) tuples.
        """
        return await self._repo.get_neighbors(node_id, relations=relations)

    async def find_path(
        self, source: str, target: str, max_depth: int = 5
    ) -> list[str] | None:
        """Find shortest path between two nodes.

        Returns:
            List of node IDs from source to target, or None if no path exists.
        """
        depth = min(max_depth, self._config.max_traversal_depth)
        return await self._repo.find_path(source, target, max_depth=depth)

    async def get_subgraph(
        self, node_id: str, depth: int = 2
    ) -> dict[str, Any]:
        """Extract a local subgraph around a node.

        Returns:
            Dict with "nodes" and "edges" lists.
        """
        d = min(depth, self._config.max_traversal_depth)
        return await self._repo.get_subgraph(node_id, depth=d)
