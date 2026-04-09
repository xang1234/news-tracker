"""Tests for the graph sync service.

Uses mock repositories to validate edge derivation, predicate mapping,
seed-prior merging, and retraction handling.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.assertions.schemas import ResolvedAssertion
from src.graph.sync import (
    MIN_EVIDENCE_TO_SUPERSEDE,
    GraphSyncService,
    SyncResult,
    _PREDICATE_TO_RELATION,
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
    service = GraphSyncService(mock_database)
    # Replace repos with mocks
    service._assertion_repo = AsyncMock()
    service._graph_repo = AsyncMock()
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
            side_effect=lambda status, limit: (
                [assertion] if status == "active" else []
            )
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
            side_effect=lambda status, limit: (
                [assertion] if status == "active" else []
            )
        )

        result = await sync_service.sync()

        assert result.edges_skipped >= 1
        assert result.edges_synced == 0

    @pytest.mark.asyncio
    async def test_respects_seed_edge_threshold(self, sync_service):
        """Seed edges (no source_doc_ids) should not be overridden
        unless evidence has sufficient support."""
        assertion = _make_assertion(
            "TSMC", "supplies_to", "NVDA",
            confidence=0.6,
            support_count=1,  # Below MIN_EVIDENCE_TO_SUPERSEDE
        )

        # Mock an existing seed edge (no source_doc_ids)
        seed_edge = MagicMock()
        seed_edge.source_doc_ids = []  # Seed edge indicator

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: (
                [assertion] if status == "active" else []
            )
        )
        sync_service._graph_repo.get_edge = AsyncMock(return_value=seed_edge)

        result = await sync_service.sync()

        assert result.edges_skipped >= 1
        assert result.edges_synced == 0

    @pytest.mark.asyncio
    async def test_overrides_seed_with_sufficient_evidence(self, sync_service):
        assertion = _make_assertion(
            "TSMC", "supplies_to", "NVDA",
            confidence=0.9,
            support_count=5,  # Above MIN_EVIDENCE_TO_SUPERSEDE
        )

        seed_edge = MagicMock()
        seed_edge.source_doc_ids = []

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: (
                [assertion] if status == "active" else []
            )
        )
        sync_service._graph_repo.get_edge = AsyncMock(return_value=seed_edge)
        sync_service._graph_repo.add_edge = AsyncMock()

        result = await sync_service.sync()

        assert result.edges_synced == 1

    @pytest.mark.asyncio
    async def test_removes_retracted_edges(self, sync_service):
        retracted = _make_assertion(
            "TSMC", "supplies_to", "NVDA",
            status="retracted",
            confidence=0.1,
        )

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: (
                [retracted] if status == "retracted" else []
            )
        )
        sync_service._graph_repo.remove_edge = AsyncMock(return_value=True)

        result = await sync_service.sync()

        assert result.edges_removed == 1
        sync_service._graph_repo.remove_edge.assert_called_once_with(
            "TSMC", "NVDA", "supplies_to",
        )

    @pytest.mark.asyncio
    async def test_maps_extended_predicate_during_sync(self, sync_service):
        assertion = _make_assertion(
            "Apple", "customer_of", "TSMC",
            support_count=5,
        )

        sync_service._assertion_repo.list_assertions = AsyncMock(
            side_effect=lambda status, limit: (
                [assertion] if status == "active" else []
            )
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
