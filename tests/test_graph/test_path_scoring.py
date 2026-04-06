"""Tests for 1-hop and 2-hop path scoring.

Verifies decomposed factor computation, path construction, sign
compounding, hop decay, and the full scoring pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.graph.path_scoring import (
    DEFAULT_FRESHNESS_HALF_LIFE_DAYS,
    DEFAULT_FRESHNESS_UNKNOWN,
    DEFAULT_HOP_DECAY,
    compute_corroboration_factor,
    compute_freshness_factor,
    score_edge,
    score_paths_from,
)
from src.graph.structural import StructuralRelation, StructuralSnapshot

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Helpers ---------------------------------------------------------------


def _rel(
    source: str = "A",
    target: str = "B",
    predicate: str = "supplies_to",
    confidence: float = 0.8,
    sign: int = 1,
    assertion_id: str = "asrt_001",
    valid_from: datetime | None = None,
    support_count: int = 5,
    contradiction_count: int = 0,
    source_diversity: int = 3,
) -> StructuralRelation:
    return StructuralRelation(
        source_concept_id=source,
        target_concept_id=target,
        predicate=predicate,
        confidence=confidence,
        sign=sign,
        assertion_id=assertion_id,
        is_current=True,
        assertion_status="active",
        valid_from=valid_from or NOW - timedelta(days=30),
        support_count=support_count,
        contradiction_count=contradiction_count,
        source_diversity=source_diversity,
    )


def _snap(relations: list[StructuralRelation]) -> StructuralSnapshot:
    return StructuralSnapshot(
        current=relations,
        history=[],
        predicate_counts={},
        concept_count=0,
        computed_at=NOW,
    )


# -- Freshness tests -------------------------------------------------------


class TestFreshness:
    """Exponential decay freshness factor."""

    def test_very_fresh(self) -> None:
        factor = compute_freshness_factor(NOW - timedelta(days=1), NOW)
        assert factor > 0.99

    def test_at_half_life(self) -> None:
        factor = compute_freshness_factor(
            NOW - timedelta(days=DEFAULT_FRESHNESS_HALF_LIFE_DAYS), NOW,
        )
        assert abs(factor - 0.5) < 0.01

    def test_very_old(self) -> None:
        factor = compute_freshness_factor(NOW - timedelta(days=365), NOW)
        assert factor < 0.1

    def test_unknown_validity(self) -> None:
        factor = compute_freshness_factor(None, NOW)
        assert factor == DEFAULT_FRESHNESS_UNKNOWN

    def test_future_valid_from(self) -> None:
        """Future valid_from clamps to 0 days, giving ~1.0."""
        factor = compute_freshness_factor(NOW + timedelta(days=10), NOW)
        assert factor > 0.99

    def test_custom_half_life(self) -> None:
        factor = compute_freshness_factor(
            NOW - timedelta(days=30), NOW, half_life_days=30.0,
        )
        assert abs(factor - 0.5) < 0.01

    def test_zero_days(self) -> None:
        factor = compute_freshness_factor(NOW, NOW)
        assert factor == 1.0


# -- Corroboration tests ---------------------------------------------------


class TestCorroboration:
    """Evidence quality from diversity, volume, and agreement."""

    def test_perfect_corroboration(self) -> None:
        factor = compute_corroboration_factor(
            support_count=5, contradiction_count=0, source_diversity=3,
        )
        assert factor == 1.0

    def test_no_evidence(self) -> None:
        factor = compute_corroboration_factor(
            support_count=0, contradiction_count=0, source_diversity=0,
        )
        assert factor == 0.0

    def test_contradictions_reduce(self) -> None:
        full = compute_corroboration_factor(5, 0, 3)
        with_contra = compute_corroboration_factor(5, 5, 3)
        assert with_contra < full

    def test_low_diversity_reduces(self) -> None:
        full = compute_corroboration_factor(5, 0, 3)
        low_div = compute_corroboration_factor(5, 0, 1)
        assert low_div < full

    def test_low_volume_reduces(self) -> None:
        full = compute_corroboration_factor(5, 0, 3)
        low_vol = compute_corroboration_factor(1, 0, 3)
        assert low_vol < full

    def test_support_ratio(self) -> None:
        """50% support / 50% contradiction = 0.5 support ratio."""
        factor = compute_corroboration_factor(5, 5, 3)
        assert abs(factor - 0.5) < 0.01  # 0.5 * 1.0 * 1.0

    def test_custom_ceilings(self) -> None:
        factor = compute_corroboration_factor(
            10, 0, 5,
            diversity_ceiling=5, volume_ceiling=10,
        )
        assert factor == 1.0


# -- Edge scoring tests ----------------------------------------------------


class TestScoreEdge:
    """Score individual structural relations."""

    def test_basic_score(self) -> None:
        rel = _rel(confidence=0.8)
        se = score_edge(rel, NOW)
        assert se.relation is rel
        assert se.freshness_factor > 0
        assert se.corroboration_factor > 0
        assert se.edge_score > 0

    def test_score_components(self) -> None:
        """edge_score = confidence * freshness * corroboration."""
        rel = _rel(confidence=1.0, valid_from=NOW, support_count=5,
                   contradiction_count=0, source_diversity=3)
        se = score_edge(rel, NOW)
        # confidence=1.0, freshness=1.0 (just now), corroboration=1.0 (perfect)
        assert se.edge_score == 1.0

    def test_low_confidence_reduces(self) -> None:
        high = score_edge(_rel(confidence=0.9), NOW)
        low = score_edge(_rel(confidence=0.3), NOW)
        assert low.edge_score < high.edge_score

    def test_old_evidence_reduces(self) -> None:
        fresh = score_edge(_rel(valid_from=NOW - timedelta(days=1)), NOW)
        stale = score_edge(_rel(valid_from=NOW - timedelta(days=300)), NOW)
        assert stale.edge_score < fresh.edge_score


# -- Path scoring tests (1-hop) --------------------------------------------


class TestOneHopPaths:
    """Score 1-hop paths from a source concept."""

    def test_single_path(self) -> None:
        snap = _snap([_rel("A", "B")])
        paths = score_paths_from(snap, "A", now=NOW)
        assert len(paths) == 1
        assert paths[0].source_concept_id == "A"
        assert paths[0].target_concept_id == "B"
        assert paths[0].hops == 1
        assert paths[0].path_sign == 1
        assert paths[0].intermediate_concept_id is None

    def test_no_outgoing(self) -> None:
        snap = _snap([_rel("B", "C")])
        paths = score_paths_from(snap, "A", now=NOW)
        assert paths == []

    def test_multiple_targets(self) -> None:
        snap = _snap([
            _rel("A", "B", assertion_id="a1"),
            _rel("A", "C", assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        assert len(paths) == 2
        targets = {p.target_concept_id for p in paths}
        assert targets == {"B", "C"}

    def test_sorted_by_score_descending(self) -> None:
        snap = _snap([
            _rel("A", "B", confidence=0.3, assertion_id="a1"),
            _rel("A", "C", confidence=0.9, assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        assert paths[0].target_concept_id == "C"
        assert paths[0].path_score >= paths[1].path_score

    def test_competitive_sign(self) -> None:
        snap = _snap([_rel("A", "B", predicate="competes_with", sign=-1)])
        paths = score_paths_from(snap, "A", now=NOW)
        assert paths[0].path_sign == -1

    def test_no_hop_decay_for_1hop(self) -> None:
        """1-hop paths have decay^0 = 1.0."""
        snap = _snap([_rel("A", "B")])
        paths = score_paths_from(snap, "A", now=NOW)
        assert paths[0].breakdown.hop_decay == 1.0

    def test_min_score_filter(self) -> None:
        snap = _snap([_rel("A", "B", confidence=0.01, support_count=1,
                          source_diversity=1)])
        paths = score_paths_from(snap, "A", min_path_score=0.5, now=NOW)
        assert paths == []


# -- Path scoring tests (2-hop) --------------------------------------------


class TestTwoHopPaths:
    """Score 2-hop paths from a source concept."""

    def test_basic_2hop(self) -> None:
        snap = _snap([
            _rel("A", "B", assertion_id="a1"),
            _rel("B", "C", assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        two_hop = [p for p in paths if p.hops == 2]
        assert len(two_hop) == 1
        assert two_hop[0].source_concept_id == "A"
        assert two_hop[0].target_concept_id == "C"
        assert two_hop[0].intermediate_concept_id == "B"

    def test_2hop_has_decay(self) -> None:
        snap = _snap([
            _rel("A", "B", assertion_id="a1"),
            _rel("B", "C", assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        two_hop = [p for p in paths if p.hops == 2]
        assert two_hop[0].breakdown.hop_decay == DEFAULT_HOP_DECAY

    def test_2hop_lower_than_1hop(self) -> None:
        """2-hop paths score lower due to decay and compounding."""
        snap = _snap([
            _rel("A", "B", confidence=0.8, assertion_id="a1"),
            _rel("A", "C", confidence=0.8, assertion_id="a2"),
            _rel("B", "C", confidence=0.8, assertion_id="a3"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        one_hop_to_c = next(p for p in paths if p.target_concept_id == "C" and p.hops == 1)
        two_hop_to_c = next(p for p in paths if p.target_concept_id == "C" and p.hops == 2)
        assert one_hop_to_c.path_score > two_hop_to_c.path_score

    def test_sign_compounds(self) -> None:
        """positive * negative = negative."""
        snap = _snap([
            _rel("A", "B", sign=1, assertion_id="a1"),
            _rel("B", "C", sign=-1, predicate="competes_with", assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        two_hop = [p for p in paths if p.hops == 2]
        assert two_hop[0].path_sign == -1

    def test_double_negative_is_positive(self) -> None:
        """negative * negative = positive."""
        snap = _snap([
            _rel("A", "B", sign=-1, predicate="competes_with", assertion_id="a1"),
            _rel("B", "C", sign=-1, predicate="blocks", assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        two_hop = [p for p in paths if p.hops == 2]
        assert two_hop[0].path_sign == 1

    def test_no_loops_to_source(self) -> None:
        """2-hop paths don't loop back to the source."""
        snap = _snap([
            _rel("A", "B", assertion_id="a1"),
            _rel("B", "A", assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        two_hop = [p for p in paths if p.hops == 2]
        assert two_hop == []

    def test_no_loops_to_intermediate(self) -> None:
        """2-hop paths don't loop back to the intermediate."""
        snap = _snap([
            _rel("A", "B", assertion_id="a1"),
            _rel("B", "B", assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        two_hop = [p for p in paths if p.hops == 2]
        assert two_hop == []

    def test_max_hops_1(self) -> None:
        """max_hops=1 excludes 2-hop paths."""
        snap = _snap([
            _rel("A", "B", assertion_id="a1"),
            _rel("B", "C", assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", max_hops=1, now=NOW)
        assert all(p.hops == 1 for p in paths)

    def test_custom_hop_decay(self) -> None:
        snap = _snap([
            _rel("A", "B", assertion_id="a1"),
            _rel("B", "C", assertion_id="a2"),
        ])
        paths = score_paths_from(snap, "A", hop_decay=0.5, now=NOW)
        two_hop = [p for p in paths if p.hops == 2]
        assert two_hop[0].breakdown.hop_decay == 0.5


# -- Integration tests -----------------------------------------------------


class TestPathScoringIntegration:
    """Full pipeline from snapshot to scored paths."""

    def test_diamond_graph(self) -> None:
        """A → B, A → C, B → D, C → D. Two 2-hop paths to D."""
        snap = _snap([
            _rel("A", "B", assertion_id="a1"),
            _rel("A", "C", assertion_id="a2"),
            _rel("B", "D", assertion_id="a3"),
            _rel("C", "D", assertion_id="a4"),
        ])
        paths = score_paths_from(snap, "A", now=NOW)
        two_hop_to_d = [p for p in paths if p.target_concept_id == "D" and p.hops == 2]
        assert len(two_hop_to_d) == 2
        intermediates = {p.intermediate_concept_id for p in two_hop_to_d}
        assert intermediates == {"B", "C"}

    def test_empty_snapshot(self) -> None:
        snap = _snap([])
        paths = score_paths_from(snap, "A", now=NOW)
        assert paths == []

    def test_to_dict(self) -> None:
        snap = _snap([_rel("A", "B")])
        paths = score_paths_from(snap, "A", now=NOW)
        d = paths[0].to_dict()
        assert d["source_concept_id"] == "A"
        assert d["target_concept_id"] == "B"
        assert "breakdown" in d
        assert "confidence_product" in d["breakdown"]
        assert "edge_predicates" in d

    def test_breakdown_preserved(self) -> None:
        snap = _snap([_rel("A", "B", confidence=0.8)])
        paths = score_paths_from(snap, "A", now=NOW)
        b = paths[0].breakdown
        assert b.confidence_product == 0.8
        assert b.freshness_product > 0
        assert b.corroboration_product > 0
        assert paths[0].path_score > 0


# -- Dataclass tests -------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass invariants."""

    def test_scored_path_frozen(self) -> None:
        snap = _snap([_rel("A", "B")])
        paths = score_paths_from(snap, "A", now=NOW)
        try:
            paths[0].path_score = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_scored_edge_frozen(self) -> None:
        se = score_edge(_rel(), NOW)
        try:
            se.edge_score = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass
