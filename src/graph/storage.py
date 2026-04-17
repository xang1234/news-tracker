"""Persistence layer for causal graph nodes and edges.

Provides asyncpg-based CRUD and traversal operations on the causal_nodes
and causal_edges tables. Graph traversals use recursive CTEs with depth
limiting and cycle detection.
"""

import json
import logging
from typing import Any

from src.storage.database import Database

from .schemas import CausalEdge, CausalEdgeSupport, CausalNode

logger = logging.getLogger(__name__)


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value)


def _row_to_node(row: Any) -> CausalNode:
    """Convert an asyncpg Record to a CausalNode."""
    return CausalNode(
        node_id=row["node_id"],
        node_type=row["node_type"],
        name=row["name"],
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_edge(row: Any) -> CausalEdge:
    """Convert an asyncpg Record to a CausalEdge."""
    return CausalEdge(
        source=row["source"],
        target=row["target"],
        relation=row["relation"],
        confidence=row["confidence"],
        source_doc_ids=list(row["source_doc_ids"]),
        created_at=row["created_at"],
        metadata=_parse_json(row["metadata"]),
    )


def _row_to_edge_support(row: Any) -> CausalEdgeSupport:
    """Convert an asyncpg Record to a CausalEdgeSupport."""
    return CausalEdgeSupport(
        source=row["source"],
        target=row["target"],
        relation=row["relation"],
        support_key=row["support_key"],
        origin_kind=row["origin_kind"],
        confidence=row["confidence"],
        source_doc_ids=list(row["source_doc_ids"] or []),
        active=bool(row["active"]),
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class GraphRepository:
    """CRUD and traversal operations for the causal graph."""

    def __init__(self, database: Database) -> None:
        self._db = database

    @staticmethod
    def _legacy_support_key(source: str, target: str, relation: str) -> str:
        return f"legacy::{source}::{target}::{relation}"

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
        """Add or update a legacy/manual edge support and refresh the projection."""
        support_key = self._legacy_support_key(source, target, relation)
        await self.upsert_edge_support(
            source,
            target,
            relation,
            support_key=support_key,
            origin_kind="legacy",
            confidence=confidence,
            source_doc_ids=source_doc_ids,
            metadata=metadata,
        )
        await self.refresh_edge(source, target, relation)

    async def remove_edge(self, source: str, target: str, relation: str) -> bool:
        """Remove the legacy/manual support for an edge and refresh the projection."""
        support_key = self._legacy_support_key(source, target, relation)
        removed = await self.deactivate_edge_support(
            source,
            target,
            relation,
            support_key=support_key,
        )
        await self.refresh_edge(source, target, relation)
        return removed

    async def upsert_edge_support(
        self,
        source: str,
        target: str,
        relation: str,
        *,
        support_key: str,
        origin_kind: str,
        confidence: float = 1.0,
        source_doc_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        active: bool = True,
    ) -> None:
        """Upsert one support row contributing to an aggregated edge."""
        await self._db.execute(
            """
            INSERT INTO causal_edge_supports (
                source, target, relation, support_key, origin_kind,
                confidence, source_doc_ids, active, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (source, target, relation, support_key) DO UPDATE
                SET origin_kind = EXCLUDED.origin_kind,
                    confidence = EXCLUDED.confidence,
                    source_doc_ids = EXCLUDED.source_doc_ids,
                    active = EXCLUDED.active,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """,
            source,
            target,
            relation,
            support_key,
            origin_kind,
            confidence,
            source_doc_ids or [],
            active,
            json.dumps(metadata or {}),
        )

    async def deactivate_edge_support(
        self,
        source: str,
        target: str,
        relation: str,
        *,
        support_key: str,
    ) -> bool:
        """Deactivate one support row and return whether it existed."""
        result = await self._db.execute(
            """
            UPDATE causal_edge_supports
            SET active = FALSE,
                updated_at = NOW()
            WHERE source = $1
              AND target = $2
              AND relation = $3
              AND support_key = $4
              AND active = TRUE
            """,
            source,
            target,
            relation,
            support_key,
        )
        return result == "UPDATE 1"

    async def list_edge_supports(
        self,
        source: str,
        target: str,
        relation: str,
        *,
        active_only: bool = False,
    ) -> list[CausalEdgeSupport]:
        """List support rows for an edge triple."""
        query = """
        SELECT *
        FROM causal_edge_supports
        WHERE source = $1 AND target = $2 AND relation = $3
        """
        if active_only:
            query += " AND active = TRUE"
        query += " ORDER BY support_key"
        rows = await self._db.fetch(query, source, target, relation)
        return [_row_to_edge_support(row) for row in rows]

    async def refresh_edge(self, source: str, target: str, relation: str) -> bool:
        """Recompute the aggregated edge from active supports."""
        supports = await self.list_edge_supports(source, target, relation, active_only=True)
        if not supports:
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

        aggregated_source_ids = sorted(
            {
                source_doc_id
                for support in supports
                for source_doc_id in support.source_doc_ids
                if source_doc_id
            }
        )
        metadata = {
            "support_count": len(supports),
            "support_keys": [support.support_key for support in supports],
            "origin_kinds": sorted({support.origin_kind for support in supports}),
            "supports": [
                {
                    "support_key": support.support_key,
                    "origin_kind": support.origin_kind,
                    "active": support.active,
                    "confidence": support.confidence,
                    "source_doc_ids": support.source_doc_ids,
                    "metadata": support.metadata,
                }
                for support in supports
            ],
        }

        await self._db.execute(
            """
            INSERT INTO causal_edges (
                source, target, relation, confidence, source_doc_ids, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (source, target, relation) DO UPDATE
                SET confidence = EXCLUDED.confidence,
                    source_doc_ids = EXCLUDED.source_doc_ids,
                    metadata = EXCLUDED.metadata
            """,
            source,
            target,
            relation,
            max(support.confidence for support in supports),
            aggregated_source_ids,
            json.dumps(metadata),
        )
        return True

    async def get_edge(self, source: str, target: str, relation: str) -> CausalEdge | None:
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

    async def get_downstream(self, node_id: str, max_depth: int = 2) -> list[tuple[str, int]]:
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

    async def get_upstream(self, node_id: str, max_depth: int = 2) -> list[tuple[str, int]]:
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

    async def find_path(self, source: str, target: str, max_depth: int = 5) -> list[str] | None:
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

    async def get_all_nodes(
        self,
        node_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[CausalNode]:
        """List all graph nodes with optional type filter.

        Args:
            node_type: Optional filter to a specific node type (ticker, theme, technology).
            limit: Maximum number of nodes to return.
            offset: Row offset for pagination.

        Returns:
            List of CausalNode objects.
        """
        if node_type:
            rows = await self._db.fetch(
                (
                    "SELECT * FROM causal_nodes WHERE node_type = $1 "
                    "ORDER BY name, node_id LIMIT $2 OFFSET $3"
                ),
                node_type,
                limit,
                offset,
            )
        else:
            rows = await self._db.fetch(
                "SELECT * FROM causal_nodes ORDER BY name, node_id LIMIT $1 OFFSET $2",
                limit,
                offset,
            )
        return [_row_to_node(r) for r in rows]

    async def get_subgraph(self, node_id: str, depth: int = 2) -> dict[str, Any]:
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
