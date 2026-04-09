"""Tests for the graph sync service.

Uses mock repositories to validate edge derivation, predicate mapping,
seed-prior merging, and retraction handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.assertions.schemas import ResolvedAssertion
from src.graph.sync import (
    _PREDICATE_TO_RELATION,
    MIN_EVIDENCE_TO_SUPERSEDE,
    GraphSyncService,
    SyncResult,
)


def _make_assertion(
    subject: str,
    predicate: str,
    obj: str | None = None,
    confidence: float = 0.8,
    status: str = "active",
    support_count: int = 5,
    contradiction_count: int = 0,
) -> ResolvedAssertion:
    """Helper to build a minimal ResolvedAssertion."""
    from src.assertions.schemas import make_assertion_id

    return ResolvedAssertion(
        assertion_id=make_assertion_id(subject, predicate, obj),
        subject_concept_id=subject,
        predicate=predicate,
        object_concept_id=obj,
        confidence=confidence,
        status=status,
        support_count=support_count,
        contradiction_count=contradiction_count,
        source_diversity=1,
    )


@pytest.fixture
def mock_database():
    """Create a mock Database."""
    db = MagicMock()
    return db


@pytest.fixture
def sync_service(mock_database):
    """Create a GraphSyncService with mocked internals."""
    service = GraphSyncService.__new__(GraphSyncService)
    service._min_evidence = MIN_EVIDENCE_TO_SUPERSEDE
    # Replace repos/graph with mocks
    service._assertion_repo = AsyncMock()
    service._graph_repo = AsyncMock()
    service._graph = AsyncMock()
    service._graph.repository = service._graph_repo
    service._graph.ensure_node = AsyncMock()
    return service


class TestPredicateMapping:
    def test_direct_relations_map_to_themselves(self):
        service = GraphSyncService.__new__(GraphSyncService)
        for relation in ("depends_on", "supplies_to", "competes_with", "drives", "blocks"):
            assert service._map_predicate(relation) == relation

    def test_extended_predicates_mapped(self):
        service = GraphSyncService.__new__(GraphSyncService)
        assert service._map_predicate("customer_of") == "depends_on"
        assert service._map_predicate("uses_technology") == "depends_on"
        assert service._map_predicate("develops_technology") == "drives"
        assert service._map_predicate("produces") == "supplies_to"
        assert service._map_predicate("subsidiary_of") == "depends_on"

    def test_unknown_predicate_returns_none(self):
        service = GraphSyncService.__new__(GraphSyncService)
        assert service._map_predicate("completely_unknown") is None

    def test_all_mapped_predicates_valid(self):
        from src.graph.schemas import VALID_RELATION_TYPES

        for predicate, relation in _PREDICATE_TO_RELATION.items():
            assert relation in VALID_RELATION_TYPES, (
                f"{predicate} maps to {relation} which is not a valid relation type"
            )


class TestGraphSync:
    @pytest.mark.asyncio
    async def test_syncs_active_assertions(self, sync_service):
        assertion = _make_assertion("TSMC", "supplies_to", "NVDA")

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: ([assertion] if status == "active" else [])
        )
        sync_service._graph_repo.get_edge = AsyncMock(return_value=None)
        sync_service._graph_repo.add_edge = AsyncMock()

        result = await sync_service.sync()

        assert result.assertions_read >= 1
        assert result.edges_synced == 1
        sync_service._graph_repo.add_edge.assert_called_once()
        call_kwargs = sync_service._graph_repo.add_edge.call_args
        assert call_kwargs.kwargs["source"] == "TSMC"
        assert call_kwargs.kwargs["target"] == "NVDA"
        assert call_kwargs.kwargs["relation"] == "supplies_to"

    @pytest.mark.asyncio
    async def test_skips_unmappable_predicate(self, sync_service):
        assertion = _make_assertion("A", "totally_unknown", "B")

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: ([assertion] if status == "active" else [])
        )

        result = await sync_service.sync()

        assert result.edges_skipped >= 1
        assert result.edges_synced == 0

    @pytest.mark.asyncio
    async def test_respects_seed_edge_threshold(self, sync_service):
        """Seed edges (no source_doc_ids) should not be overridden
        unless evidence has sufficient support."""
        assertion = _make_assertion(
            "TSMC",
            "supplies_to",
            "NVDA",
            confidence=0.6,
            support_count=1,  # Below MIN_EVIDENCE_TO_SUPERSEDE
        )

        # Mock an existing seed edge (no source_doc_ids)
        seed_edge = MagicMock()
        seed_edge.source_doc_ids = []  # Seed edge indicator

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: ([assertion] if status == "active" else [])
        )
        sync_service._graph_repo.get_edge = AsyncMock(return_value=seed_edge)

        result = await sync_service.sync()

        assert result.edges_skipped >= 1
        assert result.edges_synced == 0

    @pytest.mark.asyncio
    async def test_overrides_seed_with_sufficient_evidence(self, sync_service):
        assertion = _make_assertion(
            "TSMC",
            "supplies_to",
            "NVDA",
            confidence=0.9,
            support_count=5,  # Above MIN_EVIDENCE_TO_SUPERSEDE
        )

        seed_edge = MagicMock()
        seed_edge.source_doc_ids = []

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: ([assertion] if status == "active" else [])
        )
        sync_service._graph_repo.get_edge = AsyncMock(return_value=seed_edge)
        sync_service._graph_repo.add_edge = AsyncMock()

        result = await sync_service.sync()

        assert result.edges_synced == 1

    @pytest.mark.asyncio
    async def test_removes_retracted_edges(self, sync_service):
        retracted = _make_assertion(
            "TSMC",
            "supplies_to",
            "NVDA",
            status="retracted",
            confidence=0.1,
        )

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: ([retracted] if status == "retracted" else [])
        )
        sync_service._graph_repo.remove_edge = AsyncMock(return_value=True)

        result = await sync_service.sync()

        assert result.edges_removed == 1
        sync_service._graph_repo.remove_edge.assert_called_once_with(
            "TSMC",
            "NVDA",
            "supplies_to",
        )

    @pytest.mark.asyncio
    async def test_maps_extended_predicate_during_sync(self, sync_service):
        assertion = _make_assertion(
            "Apple",
            "customer_of",
            "TSMC",
            support_count=5,
        )

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: ([assertion] if status == "active" else [])
        )
        sync_service._graph_repo.get_edge = AsyncMock(return_value=None)
        sync_service._graph_repo.add_edge = AsyncMock()

        result = await sync_service.sync()

        assert result.edges_synced == 1
        call_kwargs = sync_service._graph_repo.add_edge.call_args.kwargs
        assert call_kwargs["relation"] == "depends_on"  # customer_of → depends_on

    @pytest.mark.asyncio
    async def test_sync_result_counts(self, sync_service):
        sync_service._assertion_repo.list_assertions = AsyncMock(return_value=[])

        result = await sync_service.sync()

        assert isinstance(result, SyncResult)
        assert result.assertions_read == 0
        assert result.edges_derived == 0
        assert result.edges_synced == 0
        assert result.edges_removed == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_ensures_nodes_exist_before_edge(self, sync_service):
        assertion = _make_assertion("NEW_TICKER", "supplies_to", "NVDA")

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: ([assertion] if status == "active" else [])
        )
        sync_service._graph_repo.get_edge = AsyncMock(return_value=None)
        sync_service._graph_repo.add_edge = AsyncMock()

        await sync_service.sync()

        # ensure_node called for both source and target
        calls = sync_service._graph.ensure_node.call_args_list
        ensured_ids = {c.args[0] for c in calls}
        assert "NEW_TICKER" in ensured_ids
        assert "NVDA" in ensured_ids

    @pytest.mark.asyncio
    async def test_competes_with_creates_bidirectional_edges(self, sync_service):
        assertion = _make_assertion("AMD", "competes_with", "INTC", support_count=5)

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: ([assertion] if status == "active" else [])
        )
        sync_service._graph_repo.get_edge = AsyncMock(return_value=None)
        sync_service._graph_repo.add_edge = AsyncMock()

        result = await sync_service.sync()

        assert result.edges_synced == 1
        # add_edge called twice: AMD→INTC and INTC→AMD
        assert sync_service._graph_repo.add_edge.call_count == 2
        call_pairs = [
            (c.kwargs["source"], c.kwargs["target"])
            for c in sync_service._graph_repo.add_edge.call_args_list
        ]
        assert ("AMD", "INTC") in call_pairs
        assert ("INTC", "AMD") in call_pairs

    @pytest.mark.asyncio
    async def test_competes_with_retraction_removes_both_directions(self, sync_service):
        retracted = _make_assertion(
            "AMD",
            "competes_with",
            "INTC",
            status="retracted",
            confidence=0.1,
        )

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: ([retracted] if status == "retracted" else [])
        )
        sync_service._graph_repo.remove_edge = AsyncMock(return_value=True)

        result = await sync_service.sync()

        assert result.edges_removed == 2
        remove_calls = [
            (c.args[0], c.args[1], c.args[2])
            for c in sync_service._graph_repo.remove_edge.call_args_list
        ]
        assert ("AMD", "INTC", "competes_with") in remove_calls
        assert ("INTC", "AMD", "competes_with") in remove_calls
