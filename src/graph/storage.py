"""Persistence layer for causal graph nodes and edges.

Provides asyncpg-based CRUD and traversal operations on the causal_nodes
and causal_edges tables. Graph traversals use recursive CTEs with depth
limiting and cycle detection.
"""

import json
import logging
from typing import Any

from src.storage.database import Database

from .schemas import CausalEdge, CausalNode

logger = logging.getLogger(__name__)


def _row_to_node(row: Any) -> CausalNode:
    """Convert an asyncpg Record to a CausalNode."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return CausalNode(
        node_id=row["node_id"],
        node_type=row["node_type"],
        name=row["name"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_edge(row: Any) -> CausalEdge:
    """Convert an asyncpg Record to a CausalEdge."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return CausalEdge(
        source=row["source"],
        target=row["target"],
        relation=row["relation"],
        confidence=row["confidence"],
        source_doc_ids=list(row["source_doc_ids"]),
        created_at=row["created_at"],
        metadata=metadata,
    )


class GraphRepository:
    """CRUD and traversal operations for the causal graph."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def upsert_node(self, node: CausalNode) -> CausalNode:
        """Insert or update a node. Returns the persisted node."""
        row = await self._db.fetchrow(
            """
            INSERT INTO causal_nodes (node_id, node_type, name, metadata)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (node_id) DO UPDATE
                SET node_type = EXCLUDED.node_type,
                    name = EXCLUDED.name,
                    metadata = EXCLUDED.metadata
            RETURNING *
            """,
            node.node_id,
            node.node_type,
            node.name,
            json.dumps(node.metadata),
        )
        return _row_to_node(row)

    async def get_node(self, node_id: str) -> CausalNode | None:
        """Get a node by ID, or None if not found."""
        row = await self._db.fetchrow(
            "SELECT * FROM causal_nodes WHERE node_id = $1",
            node_id,
        )
        return _row_to_node(row) if row else None

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node (cascades to edges). Returns True if deleted."""
        result = await self._db.execute(
            "DELETE FROM causal_nodes WHERE node_id = $1",
            node_id,
        )
        return result == "DELETE 1"

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        confidence: float = 1.0,
        source_doc_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update an edge between two nodes.

        Uses ON CONFLICT to merge source_doc_ids and update confidence
        on re-insertion of the same (source, target, relation) triple.
        """
        await self._db.execute(
            """
            INSERT INTO causal_edges
                (source, target, relation, confidence, source_doc_ids, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (source, target, relation) DO UPDATE
                SET confidence = EXCLUDED.confidence,
                    source_doc_ids = ARRAY(
                        SELECT DISTINCT unnest(
                            causal_edges.source_doc_ids || EXCLUDED.source_doc_ids
                        )
                    ),
                    metadata = EXCLUDED.metadata
            """,
            source,
            target,
            relation,
            confidence,
            source_doc_ids or [],
            json.dumps(metadata or {}),
        )

    async def remove_edge(
        self, source: str, target: str, relation: str
    ) -> bool:
        """Remove an edge. Returns True if an edge was actually deleted."""
        result = await self._db.execute(
            """
            DELETE FROM causal_edges
            WHERE source = $1 AND target = $2 AND relation = $3
            """,
            source,
            target,
            relation,
        )
        return result == "DELETE 1"

    async def get_edge(
        self, source: str, target: str, relation: str
    ) -> CausalEdge | None:
        """Get a specific edge, or None if not found."""
        row = await self._db.fetchrow(
            """
            SELECT * FROM causal_edges
            WHERE source = $1 AND target = $2 AND relation = $3
            """,
            source,
            target,
            relation,
        )
        return _row_to_edge(row) if row else None

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    async def get_downstream_edges(
        self, node_id: str, max_depth: int = 2
    ) -> list[tuple[str, str, str, float, int]]:
        """Get all edges reachable by following outgoing edges with full edge info.

        Returns edges grouped by depth for level-by-level propagation.
        Uses a recursive CTE with cycle detection via path tracking.

        Returns:
            List of (source, target, relation, confidence, depth) tuples,
            ordered by depth.
        """
        rows = await self._db.fetch(
            """
            WITH RECURSIVE downstream AS (
                SELECT source, target, relation, confidence, 1 AS depth,
                       ARRAY[source, target] AS path
                FROM causal_edges
                WHERE source = $1

                UNION ALL

                SELECT e.source, e.target, e.relation, e.confidence, d.depth + 1,
                       d.path || e.target
                FROM causal_edges e
                JOIN downstream d ON e.source = d.target
                WHERE d.depth < $2
                  AND NOT e.target = ANY(d.path)
            )
            SELECT source, target, relation, confidence, depth
            FROM downstream
            ORDER BY depth
            """,
            node_id,
            max_depth,
        )
        return [
            (row["source"], row["target"], row["relation"], row["confidence"], row["depth"])
            for row in rows
        ]

    async def get_downstream(
        self, node_id: str, max_depth: int = 2
    ) -> list[tuple[str, int]]:
        """Get all nodes reachable by following outgoing edges.

        Uses a recursive CTE with cycle detection via path tracking.

        Returns:
            List of (node_id, depth) tuples, ordered by depth then ID.
        """
        rows = await self._db.fetch(
            """
            WITH RECURSIVE downstream AS (
                SELECT target AS node_id, 1 AS depth, ARRAY[source, target] AS path
                FROM causal_edges
                WHERE source = $1

                UNION ALL

                SELECT e.target, d.depth + 1, d.path || e.target
                FROM causal_edges e
                JOIN downstream d ON e.source = d.node_id
                WHERE d.depth < $2
                  AND NOT e.target = ANY(d.path)
            )
            SELECT DISTINCT ON (node_id) node_id, depth
            FROM downstream
            ORDER BY node_id, depth
            """,
            node_id,
            max_depth,
        )
        return [(row["node_id"], row["depth"]) for row in rows]

    async def get_upstream(
        self, node_id: str, max_depth: int = 2
    ) -> list[tuple[str, int]]:
        """Get all nodes that can reach this node via outgoing edges.

        Follows edges in reverse direction (target→source).

        Returns:
            List of (node_id, depth) tuples, ordered by depth then ID.
        """
        rows = await self._db.fetch(
            """
            WITH RECURSIVE upstream AS (
                SELECT source AS node_id, 1 AS depth, ARRAY[target, source] AS path
                FROM causal_edges
                WHERE target = $1

                UNION ALL

                SELECT e.source, u.depth + 1, u.path || e.source
                FROM causal_edges e
                JOIN upstream u ON e.target = u.node_id
                WHERE u.depth < $2
                  AND NOT e.source = ANY(u.path)
            )
            SELECT DISTINCT ON (node_id) node_id, depth
            FROM upstream
            ORDER BY node_id, depth
            """,
            node_id,
            max_depth,
        )
        return [(row["node_id"], row["depth"]) for row in rows]

    async def get_neighbors(
        self,
        node_id: str,
        relations: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Get immediate neighbors (both directions) with their relation type.

        Args:
            node_id: The node to query.
            relations: Optional filter to specific relation types.

        Returns:
            List of (neighbor_node_id, relation) tuples.
        """
        if relations:
            rows = await self._db.fetch(
                """
                SELECT target AS neighbor, relation FROM causal_edges
                WHERE source = $1 AND relation = ANY($2)
                UNION ALL
                SELECT source AS neighbor, relation FROM causal_edges
                WHERE target = $1 AND relation = ANY($2)
                """,
                node_id,
                relations,
            )
        else:
            rows = await self._db.fetch(
                """
                SELECT target AS neighbor, relation FROM causal_edges
                WHERE source = $1
                UNION ALL
                SELECT source AS neighbor, relation FROM causal_edges
                WHERE target = $1
                """,
                node_id,
            )
        return [(row["neighbor"], row["relation"]) for row in rows]

    async def find_path(
        self, source: str, target: str, max_depth: int = 5
    ) -> list[str] | None:
        """Find shortest path between two nodes (BFS via recursive CTE).

        Returns:
            List of node IDs from source to target, or None if no path exists.
        """
        row = await self._db.fetchrow(
            """
            WITH RECURSIVE search AS (
                SELECT
                    e.target AS node_id,
                    ARRAY[e.source, e.target] AS path,
                    1 AS depth
                FROM causal_edges e
                WHERE e.source = $1

                UNION ALL

                SELECT
                    e.target,
                    s.path || e.target,
                    s.depth + 1
                FROM causal_edges e
                JOIN search s ON e.source = s.node_id
                WHERE s.depth < $3
                  AND NOT e.target = ANY(s.path)
            )
            SELECT path
            FROM search
            WHERE node_id = $2
            ORDER BY depth
            LIMIT 1
            """,
            source,
            target,
            max_depth,
        )
        return list(row["path"]) if row else None

    async def get_subgraph(
        self, node_id: str, depth: int = 2
    ) -> dict[str, Any]:
        """Extract a local subgraph around a node.

        Returns:
            Dict with "nodes" (list of CausalNode dicts) and
            "edges" (list of CausalEdge dicts).
        """
        # Collect reachable node IDs in both directions
        downstream = await self.get_downstream(node_id, max_depth=depth)
        upstream = await self.get_upstream(node_id, max_depth=depth)

        node_ids = {node_id}
        for nid, _ in downstream:
            node_ids.add(nid)
        for nid, _ in upstream:
            node_ids.add(nid)

        if not node_ids:
            return {"nodes": [], "edges": []}

        id_list = list(node_ids)

        # Fetch node records
        node_rows = await self._db.fetch(
            "SELECT * FROM causal_nodes WHERE node_id = ANY($1)",
            id_list,
        )
        nodes = [_row_to_node(r) for r in node_rows]

        # Fetch edges between the collected nodes
        edge_rows = await self._db.fetch(
            """
            SELECT * FROM causal_edges
            WHERE source = ANY($1) AND target = ANY($1)
            """,
            id_list,
        )
        edges = [_row_to_edge(r) for r in edge_rows]

        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "name": n.name,
                    "metadata": n.metadata,
                }
                for n in nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "confidence": e.confidence,
                }
                for e in edges
            ],
        }
