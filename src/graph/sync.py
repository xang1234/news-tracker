"""Sync assertion-derived edges into the causal graph.

Reads resolved assertions, calls ``derive_edges()``, and persists
current edges to the ``causal_edges`` table via ``GraphRepository``.

Design decisions:
- Seed edges (from ``seed_data.py``) are bootstrap priors with no
  ``source_doc_ids``.  Evidence-backed edges carry document lineage.
- When an evidence edge matches a seed edge (same source/target/relation),
  the evidence confidence is used if ``support_count >= min_evidence_to_supersede``.
- Retracted assertions deactivate their support rows and recompute the
  aggregate edge projection.
- The sync is idempotent — support rows are keyed per assertion.

Usage:
    service = GraphSyncService(database)
    result = await service.sync()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from src.assertions.edges import DerivedEdge, derive_edges
from src.assertions.repository import AssertionRepository
from src.assertions.schemas import ResolvedAssertion
from src.graph.causal_graph import CausalGraph
from src.graph.schemas import VALID_RELATION_TYPES
from src.storage.database import Database

logger = structlog.get_logger(__name__)

# Structural predicates that map to existing graph relation types
_PREDICATE_TO_RELATION: dict[str, str] = {
    # Direct mappings (already in VALID_RELATION_TYPES)
    "depends_on": "depends_on",
    "supplies_to": "supplies_to",
    "competes_with": "competes_with",
    "drives": "drives",
    "blocks": "blocks",
    # Extended predicates → closest graph relation
    "customer_of": "depends_on",
    "uses_technology": "depends_on",
    "develops_technology": "drives",
    "produces": "supplies_to",
    "consumes": "depends_on",
    "component_of": "depends_on",
    "contains_component": "supplies_to",
    "subsidiary_of": "depends_on",
    "parent_of": "drives",
    "operates_facility": "drives",
    "located_at": "drives",
}

# Minimum supporting claims before evidence supersedes seed priors
MIN_EVIDENCE_TO_SUPERSEDE = 3


@dataclass
class SyncResult:
    """Summary of a graph sync operation.

    Attributes:
        assertions_read: Total assertions fetched.
        edges_derived: Edges that passed the confidence threshold.
        edges_synced: Edges actually written to causal_edges.
        edges_removed: Edges removed due to retracted assertions.
        edges_skipped: Edges skipped (unmappable predicate, etc.).
        errors: Per-edge error descriptions.
    """

    assertions_read: int = 0
    edges_derived: int = 0
    edges_synced: int = 0
    edges_removed: int = 0
    edges_skipped: int = 0
    errors: list[str] = field(default_factory=list)


class GraphSyncService:
    """Syncs assertion-derived edges into the causal graph."""

    def __init__(
        self,
        database: Database,
        *,
        min_evidence: int = MIN_EVIDENCE_TO_SUPERSEDE,
    ) -> None:
        self._assertion_repo = AssertionRepository(database)
        self._graph = CausalGraph(database)
        self._graph_repo = self._graph.repository
        self._min_evidence = min_evidence

    def _map_predicate(self, predicate: str) -> str | None:
        """Map a structural predicate to a valid graph relation type.

        Returns None if the predicate cannot be mapped.
        """
        # Direct match
        if predicate in VALID_RELATION_TYPES:
            return predicate
        # Extended mapping
        return _PREDICATE_TO_RELATION.get(predicate)

    async def sync(self, *, limit: int = 1000) -> SyncResult:
        """Fetch all active assertions, derive edges, and sync to graph.

        Args:
            limit: Maximum assertions to process per sync.

        Returns:
            SyncResult with counts and any errors.
        """
        result = SyncResult()

        # Fetch all active and disputed assertions
        assertions: list[ResolvedAssertion] = []
        for status in ("active", "disputed"):
            batch = await self._assertion_repo.list_assertions(
                status=status,
                limit=limit,
            )
            assertions.extend(batch)

        # Also fetch retracted to handle edge removal
        retracted = await self._assertion_repo.list_assertions(
            status="retracted",
            limit=limit,
        )

        result.assertions_read = len(assertions) + len(retracted)

        # Derive current edges from active/disputed assertions
        current_edges, _history = derive_edges(assertions)
        result.edges_derived = len(current_edges)

        # Sync current edges
        for edge in current_edges:
            try:
                await self._sync_edge(edge, result)
            except Exception as e:
                result.errors.append(
                    f"Error syncing edge {edge.source_concept_id}→{edge.target_concept_id}: {e}"
                )

        # Remove supports from retracted assertions
        _, retracted_edges = derive_edges(retracted, confidence_threshold=0.0)
        for edge in retracted_edges:
            try:
                result.edges_removed += await self._remove_edge_support(edge)
            except Exception as e:
                result.errors.append(
                    f"Error removing edge {edge.source_concept_id}→{edge.target_concept_id}: {e}"
                )

        logger.info(
            "Graph sync complete",
            assertions_read=result.assertions_read,
            edges_derived=result.edges_derived,
            edges_synced=result.edges_synced,
            edges_removed=result.edges_removed,
            edges_skipped=result.edges_skipped,
            errors=len(result.errors),
        )

        return result

    async def _sync_edge(self, edge: DerivedEdge, result: SyncResult) -> None:
        """Persist a single derived edge to the causal graph."""
        relation = self._map_predicate(edge.predicate)
        if relation is None:
            result.edges_skipped += 1
            return

        # Only supersede seed edges when we have enough evidence
        supports = await self._graph_repo.list_edge_supports(
            edge.source_concept_id,
            edge.target_concept_id,
            relation,
            active_only=True,
        )
        has_legacy_support = any(support.origin_kind == "legacy" for support in supports)
        if has_legacy_support and edge.support_count < self._min_evidence:
            # This is a seed edge — only override with sufficient evidence
            result.edges_skipped += 1
            return

        # Ensure both nodes exist (prevents FK violations for non-seeded concepts).
        # Only create if missing — don't overwrite existing seed nodes' type/name/metadata.
        for concept_id in (edge.source_concept_id, edge.target_concept_id):
            existing_node = await self._graph_repo.get_node(concept_id)
            if existing_node is None:
                await self._graph.ensure_node(concept_id)

        source_doc_ids = self._extract_source_doc_ids(edge)
        metadata: dict[str, Any] = {
            "assertion_id": edge.assertion_id,
            "support_count": edge.support_count,
            "contradiction_count": edge.contradiction_count,
            "source_diversity": edge.source_diversity,
            "synced_from": "assertion",
            "assertion_metadata": edge.metadata,
        }

        await self._graph_repo.upsert_edge_support(
            source=edge.source_concept_id,
            target=edge.target_concept_id,
            relation=relation,
            support_key=edge.assertion_id,
            origin_kind="assertion",
            confidence=edge.confidence,
            source_doc_ids=source_doc_ids,
            metadata=metadata,
        )
        await self._graph_repo.refresh_edge(
            edge.source_concept_id,
            edge.target_concept_id,
            relation,
        )

        # competes_with requires explicit bidirectional edges (A→B and B→A)
        if relation == "competes_with":
            reverse_supports = await self._graph_repo.list_edge_supports(
                edge.target_concept_id,
                edge.source_concept_id,
                relation,
                active_only=True,
            )
            reverse_has_legacy = any(support.origin_kind == "legacy" for support in reverse_supports)
            if not (reverse_has_legacy and edge.support_count < self._min_evidence):
                await self._graph_repo.upsert_edge_support(
                    source=edge.target_concept_id,
                    target=edge.source_concept_id,
                    relation=relation,
                    support_key=edge.assertion_id,
                    origin_kind="assertion",
                    confidence=edge.confidence,
                    source_doc_ids=source_doc_ids,
                    metadata=metadata,
                )
                await self._graph_repo.refresh_edge(
                    edge.target_concept_id,
                    edge.source_concept_id,
                    relation,
                )

        result.edges_synced += 1

    @staticmethod
    def _extract_source_doc_ids(edge: DerivedEdge) -> list[str]:
        """Best-effort lineage extraction for edge supports."""
        metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
        lineage_ids = metadata.get("source_doc_ids") or metadata.get("source_ids")
        if isinstance(lineage_ids, list):
            parsed = [str(lineage_id) for lineage_id in lineage_ids if lineage_id]
            if parsed:
                return parsed
        return [edge.assertion_id] if edge.assertion_id else []

    async def _remove_edge_support(self, edge: DerivedEdge) -> int:
        """Deactivate one assertion support and refresh affected triples."""
        relation = self._map_predicate(edge.predicate)
        if relation is None:
            return 0

        removed_count = 0
        removed = await self._graph_repo.deactivate_edge_support(
            edge.source_concept_id,
            edge.target_concept_id,
            relation,
            support_key=edge.assertion_id,
        )
        await self._graph_repo.refresh_edge(
            edge.source_concept_id,
            edge.target_concept_id,
            relation,
        )
        if removed:
            removed_count += 1

        if relation == "competes_with":
            reverse_removed = await self._graph_repo.deactivate_edge_support(
                edge.target_concept_id,
                edge.source_concept_id,
                relation,
                support_key=edge.assertion_id,
            )
            await self._graph_repo.refresh_edge(
                edge.target_concept_id,
                edge.source_concept_id,
                relation,
            )
            if reverse_removed:
                removed_count += 1

        return removed_count
