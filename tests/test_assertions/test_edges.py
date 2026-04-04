"""Tests for assertion-derived edges, exposures, and path-cache inputs.

Verifies that graph consumers can read assertion-derived edges instead
of touching raw claims. Tests cover current/history edge derivation,
per-concept exposure aggregation, and adjacency cache construction.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.assertions.edges import (
    DEFAULT_EDGE_CONFIDENCE_THRESHOLD,
    ConceptExposure,
    DerivedEdge,
    PathCacheEntry,
    build_path_cache,
    compute_exposures,
    derive_edge,
    derive_edges,
)
from src.assertions.schemas import ResolvedAssertion


# -- Helpers ---------------------------------------------------------------


def _make_assertion(
    assertion_id: str = "asrt_test",
    *,
    subject: str = "concept_tsmc",
    predicate: str = "supplies_to",
    obj: str | None = "concept_nvda",
    confidence: float = 0.8,
    status: str = "active",
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    support_count: int = 3,
    contradiction_count: int = 0,
    source_diversity: int = 2,
) -> ResolvedAssertion:
    return ResolvedAssertion(
        assertion_id=assertion_id,
        subject_concept_id=subject,
        predicate=predicate,
        object_concept_id=obj,
        confidence=confidence,
        status=status,
        valid_from=valid_from,
        valid_to=valid_to,
        support_count=support_count,
        contradiction_count=contradiction_count,
        source_diversity=source_diversity,
    )


# -- derive_edge tests -----------------------------------------------------


class TestDeriveEdge:
    """Single-assertion edge derivation."""

    def test_active_above_threshold(self) -> None:
        a = _make_assertion(confidence=0.8)
        edge = derive_edge(a)
        assert edge is not None
        assert edge.is_current is True
        assert edge.source_concept_id == "concept_tsmc"
        assert edge.target_concept_id == "concept_nvda"
        assert edge.predicate == "supplies_to"
        assert edge.assertion_id == "asrt_test"

    def test_active_below_threshold(self) -> None:
        a = _make_assertion(confidence=0.1)
        edge = derive_edge(a)
        # Below threshold AND active → doesn't produce an edge
        # (not current, and status is active not retracted/superseded/disputed)
        assert edge is None

    def test_retracted_produces_history_edge(self) -> None:
        a = _make_assertion(status="retracted", confidence=0.7)
        edge = derive_edge(a)
        assert edge is not None
        assert edge.is_current is False
        assert edge.assertion_status == "retracted"

    def test_superseded_produces_history_edge(self) -> None:
        a = _make_assertion(status="superseded", confidence=0.6)
        edge = derive_edge(a)
        assert edge is not None
        assert edge.is_current is False

    def test_disputed_above_threshold_is_current(self) -> None:
        """Disputed assertions are actively contested, not historical."""
        a = _make_assertion(status="disputed", confidence=0.5)
        edge = derive_edge(a)
        assert edge is not None
        assert edge.is_current is True
        assert edge.assertion_status == "disputed"

    def test_disputed_below_threshold_is_history(self) -> None:
        a = _make_assertion(status="disputed", confidence=0.1)
        edge = derive_edge(a)
        assert edge is not None
        assert edge.is_current is False

    def test_unary_assertion_skipped(self) -> None:
        a = _make_assertion(obj=None)
        edge = derive_edge(a)
        assert edge is None

    def test_custom_threshold(self) -> None:
        a = _make_assertion(confidence=0.2)
        edge = derive_edge(a, confidence_threshold=0.1)
        assert edge is not None
        assert edge.is_current is True

    def test_carries_lineage(self) -> None:
        a = _make_assertion(
            support_count=5,
            contradiction_count=1,
            source_diversity=3,
        )
        edge = derive_edge(a)
        assert edge is not None
        assert edge.support_count == 5
        assert edge.contradiction_count == 1
        assert edge.source_diversity == 3

    def test_carries_validity_window(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 12, 31, tzinfo=timezone.utc)
        a = _make_assertion(valid_from=t1, valid_to=t2)
        edge = derive_edge(a)
        assert edge is not None
        assert edge.valid_from == t1
        assert edge.valid_to == t2


# -- derive_edges tests ----------------------------------------------------


class TestDeriveEdges:
    """Batch edge derivation with current/history split."""

    def test_splits_current_and_history(self) -> None:
        assertions = [
            _make_assertion("a1", confidence=0.8, status="active"),
            _make_assertion("a2", confidence=0.7, status="retracted"),
            _make_assertion("a3", confidence=0.9, status="active"),
        ]
        current, history = derive_edges(assertions)
        assert len(current) == 2
        assert len(history) == 1
        assert all(e.is_current for e in current)
        assert all(not e.is_current for e in history)

    def test_skips_unary(self) -> None:
        assertions = [
            _make_assertion("a1", obj=None),
            _make_assertion("a2", obj="concept_nvda"),
        ]
        current, history = derive_edges(assertions)
        assert len(current) == 1

    def test_empty_input(self) -> None:
        current, history = derive_edges([])
        assert current == []
        assert history == []

    def test_all_below_threshold(self) -> None:
        assertions = [
            _make_assertion("a1", confidence=0.1),
            _make_assertion("a2", confidence=0.05),
        ]
        current, history = derive_edges(assertions)
        assert len(current) == 0


# -- compute_exposures tests -----------------------------------------------


class TestComputeExposures:
    """Per-concept exposure aggregation."""

    def test_basic_exposure(self) -> None:
        assertions = [
            _make_assertion("a1", subject="concept_tsmc", obj="concept_nvda"),
        ]
        exposures = compute_exposures(assertions)
        assert "concept_tsmc" in exposures
        assert "concept_nvda" in exposures

        tsmc = exposures["concept_tsmc"]
        assert tsmc.total_assertions == 1
        assert tsmc.active_assertions == 1
        assert tsmc.as_subject_count == 1
        assert tsmc.as_object_count == 0

        nvda = exposures["concept_nvda"]
        assert nvda.as_object_count == 1

    def test_multiple_assertions(self) -> None:
        assertions = [
            _make_assertion("a1", subject="c_a", obj="c_b",
                            confidence=0.8, predicate="supplies_to"),
            _make_assertion("a2", subject="c_a", obj="c_c",
                            confidence=0.6, predicate="competes_with"),
        ]
        exposures = compute_exposures(assertions)
        a = exposures["c_a"]
        assert a.total_assertions == 2
        assert a.as_subject_count == 2
        assert a.weighted_confidence == round(0.8 + 0.6, 4)
        assert "supplies_to" in a.predicates
        assert "competes_with" in a.predicates

    def test_disputed_counted_separately(self) -> None:
        assertions = [
            _make_assertion("a1", status="disputed", confidence=0.5),
        ]
        exposures = compute_exposures(assertions)
        tsmc = exposures["concept_tsmc"]
        assert tsmc.disputed_assertions == 1
        assert tsmc.active_assertions == 0

    def test_below_threshold_excluded(self) -> None:
        assertions = [
            _make_assertion("a1", confidence=0.05),
        ]
        exposures = compute_exposures(assertions, exposure_threshold=0.1)
        assert len(exposures) == 0

    def test_unary_assertion(self) -> None:
        assertions = [
            _make_assertion("a1", obj=None),
        ]
        exposures = compute_exposures(assertions)
        assert "concept_tsmc" in exposures
        assert len(exposures) == 1  # only subject, no object

    def test_empty_input(self) -> None:
        exposures = compute_exposures([])
        assert exposures == {}


# -- build_path_cache tests ------------------------------------------------


class TestBuildPathCache:
    """Adjacency cache construction."""

    def test_basic_cache(self) -> None:
        a = _make_assertion(confidence=0.8)
        current, _ = derive_edges([a])
        cache = build_path_cache(current)

        assert "concept_tsmc" in cache
        assert "concept_nvda" in cache

        tsmc = cache["concept_tsmc"]
        assert len(tsmc.outgoing) == 1
        assert len(tsmc.incoming) == 0
        assert tsmc.outgoing[0].target_concept_id == "concept_nvda"

        nvda = cache["concept_nvda"]
        assert len(nvda.incoming) == 1
        assert len(nvda.outgoing) == 0

    def test_degree_property(self) -> None:
        assertions = [
            _make_assertion("a1", subject="c_a", obj="c_b"),
            _make_assertion("a2", subject="c_a", obj="c_c"),
            _make_assertion("a3", subject="c_d", obj="c_a"),
        ]
        current, _ = derive_edges(assertions)
        cache = build_path_cache(current)
        # c_a: 2 outgoing + 1 incoming = 3
        assert cache["c_a"].degree == 3

    def test_neighbors_property(self) -> None:
        assertions = [
            _make_assertion("a1", subject="c_a", obj="c_b"),
            _make_assertion("a2", subject="c_c", obj="c_a"),
        ]
        current, _ = derive_edges(assertions)
        cache = build_path_cache(current)
        assert cache["c_a"].neighbors == frozenset({"c_b", "c_c"})

    def test_sorted_by_confidence(self) -> None:
        assertions = [
            _make_assertion("a1", subject="c_a", obj="c_b", confidence=0.5),
            _make_assertion("a2", subject="c_a", obj="c_c", confidence=0.9),
        ]
        current, _ = derive_edges(assertions)
        cache = build_path_cache(current)
        # Outgoing sorted by confidence DESC
        assert cache["c_a"].outgoing[0].confidence > cache["c_a"].outgoing[1].confidence

    def test_empty_input(self) -> None:
        cache = build_path_cache([])
        assert cache == {}

    def test_excludes_history_edges(self) -> None:
        """Only current edges feed into path cache."""
        assertions = [
            _make_assertion("a1", confidence=0.8, status="active"),
            _make_assertion("a2", confidence=0.7, status="retracted",
                            subject="c_x", obj="c_y"),
        ]
        current, history = derive_edges(assertions)
        cache = build_path_cache(current)
        # Retracted edge not in cache
        assert "c_x" not in cache
