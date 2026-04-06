"""Tests for assertion-derived typed structural relations.

Verifies that DerivedEdges translate correctly into StructuralRelations
with predicate signs, current/history split, and snapshot metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.assertions.edges import DerivedEdge
from src.graph.structural import (
    DEFAULT_SIGN,
    PREDICATE_SIGNS,
    build_structural_snapshot,
    get_predicate_sign,
    translate_derived_edge,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Helpers ---------------------------------------------------------------


def _make_edge(
    source: str = "concept_issuer_aaa",
    target: str = "concept_issuer_bbb",
    predicate: str = "supplies_to",
    confidence: float = 0.8,
    assertion_id: str = "asrt_001",
    is_current: bool = True,
    assertion_status: str = "active",
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    support_count: int = 3,
    contradiction_count: int = 0,
    source_diversity: int = 2,
    metadata: dict[str, Any] | None = None,
) -> DerivedEdge:
    return DerivedEdge(
        source_concept_id=source,
        target_concept_id=target,
        predicate=predicate,
        confidence=confidence,
        assertion_id=assertion_id,
        is_current=is_current,
        assertion_status=assertion_status,
        valid_from=valid_from or NOW - timedelta(days=30),
        valid_to=valid_to,
        support_count=support_count,
        contradiction_count=contradiction_count,
        source_diversity=source_diversity,
        metadata=metadata or {},
    )


# -- Predicate sign tests --------------------------------------------------


class TestPredicateSigns:
    """Sign mapping for propagation direction."""

    def test_positive_predicates(self) -> None:
        for pred in (
            "supplies_to",
            "customer_of",
            "depends_on",
            "drives",
            "uses_technology",
            "develops_technology",
            "produces",
            "consumes",
            "component_of",
            "contains_component",
            "subsidiary_of",
            "parent_of",
            "operates_facility",
            "located_at",
        ):
            assert get_predicate_sign(pred) == 1, f"{pred} should be positive"

    def test_negative_predicates(self) -> None:
        assert get_predicate_sign("competes_with") == -1
        assert get_predicate_sign("blocks") == -1

    def test_unknown_predicate_defaults_positive(self) -> None:
        assert get_predicate_sign("unknown_relation") == DEFAULT_SIGN

    def test_all_known_predicates_mapped(self) -> None:
        """Every predicate in PREDICATE_SIGNS has a valid sign."""
        for pred, sign in PREDICATE_SIGNS.items():
            assert sign in (1, -1), f"{pred} has invalid sign {sign}"

    def test_signs_cover_concept_relationship_types(self) -> None:
        """All valid concept relationship types should have signs."""
        from src.security_master.concept_schemas import VALID_RELATIONSHIP_TYPES

        for rt in VALID_RELATIONSHIP_TYPES:
            assert rt in PREDICATE_SIGNS, f"{rt} missing from PREDICATE_SIGNS"

    def test_signs_cover_graph_relation_types(self) -> None:
        """All manual graph relation types should have signs."""
        from src.graph.schemas import VALID_RELATION_TYPES

        for rt in VALID_RELATION_TYPES:
            assert rt in PREDICATE_SIGNS, f"{rt} missing from PREDICATE_SIGNS"


# -- Translation tests -----------------------------------------------------


class TestTranslateDerivedEdge:
    """DerivedEdge → StructuralRelation translation."""

    def test_basic_translation(self) -> None:
        edge = _make_edge()
        rel = translate_derived_edge(edge)
        assert rel.source_concept_id == edge.source_concept_id
        assert rel.target_concept_id == edge.target_concept_id
        assert rel.predicate == "supplies_to"
        assert rel.confidence == 0.8
        assert rel.sign == 1  # supplies_to is positive
        assert rel.assertion_id == "asrt_001"
        assert rel.is_current is True

    def test_competitive_predicate_gets_negative_sign(self) -> None:
        edge = _make_edge(predicate="competes_with")
        rel = translate_derived_edge(edge)
        assert rel.sign == -1

    def test_blocking_predicate_gets_negative_sign(self) -> None:
        edge = _make_edge(predicate="blocks")
        rel = translate_derived_edge(edge)
        assert rel.sign == -1

    def test_unknown_predicate_gets_default_sign(self) -> None:
        edge = _make_edge(predicate="novel_relation")
        rel = translate_derived_edge(edge)
        assert rel.sign == DEFAULT_SIGN

    def test_preserves_temporal_validity(self) -> None:
        vf = NOW - timedelta(days=60)
        vt = NOW - timedelta(days=10)
        edge = _make_edge(valid_from=vf, valid_to=vt)
        rel = translate_derived_edge(edge)
        assert rel.valid_from == vf
        assert rel.valid_to == vt

    def test_preserves_evidence_counts(self) -> None:
        edge = _make_edge(support_count=5, contradiction_count=2, source_diversity=3)
        rel = translate_derived_edge(edge)
        assert rel.support_count == 5
        assert rel.contradiction_count == 2
        assert rel.source_diversity == 3

    def test_preserves_assertion_status(self) -> None:
        edge = _make_edge(assertion_status="disputed", is_current=True)
        rel = translate_derived_edge(edge)
        assert rel.assertion_status == "disputed"
        assert rel.is_current is True

    def test_history_edge_preserved(self) -> None:
        edge = _make_edge(is_current=False, assertion_status="retracted")
        rel = translate_derived_edge(edge)
        assert rel.is_current is False
        assert rel.assertion_status == "retracted"

    def test_metadata_copied(self) -> None:
        edge = _make_edge(metadata={"source": "filing"})
        rel = translate_derived_edge(edge)
        assert rel.metadata == {"source": "filing"}
        assert rel.metadata is not edge.metadata  # shallow copy

    def test_to_dict(self) -> None:
        edge = _make_edge()
        rel = translate_derived_edge(edge)
        d = rel.to_dict()
        assert d["source_concept_id"] == edge.source_concept_id
        assert d["predicate"] == "supplies_to"
        assert d["sign"] == 1
        assert isinstance(d["valid_from"], str)


# -- Structural snapshot tests ---------------------------------------------


class TestBuildStructuralSnapshot:
    """Build complete structural state from derived edges."""

    def test_basic_snapshot(self) -> None:
        edges = [
            _make_edge(source="A", target="B", predicate="supplies_to", is_current=True),
            _make_edge(source="B", target="C", predicate="customer_of", is_current=True),
            _make_edge(
                source="A",
                target="C",
                predicate="competes_with",
                is_current=False,
                assertion_status="retracted",
            ),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        assert len(snap.current) == 2
        assert len(snap.history) == 1
        assert snap.computed_at == NOW

    def test_predicate_counts(self) -> None:
        edges = [
            _make_edge(source="A", target="B", predicate="supplies_to", assertion_id="a1"),
            _make_edge(source="C", target="D", predicate="supplies_to", assertion_id="a2"),
            _make_edge(source="E", target="F", predicate="competes_with", assertion_id="a3"),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        assert snap.predicate_counts["supplies_to"] == 2
        assert snap.predicate_counts["competes_with"] == 1

    def test_concept_count(self) -> None:
        edges = [
            _make_edge(source="A", target="B", assertion_id="a1"),
            _make_edge(source="B", target="C", assertion_id="a2"),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        assert snap.concept_count == 3  # A, B, C

    def test_concept_count_deduplicates(self) -> None:
        edges = [
            _make_edge(source="A", target="B", assertion_id="a1"),
            _make_edge(source="A", target="C", assertion_id="a2"),
            _make_edge(source="B", target="A", assertion_id="a3"),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        assert snap.concept_count == 3  # A, B, C

    def test_empty_edges(self) -> None:
        snap = build_structural_snapshot([], now=NOW)
        assert snap.current == []
        assert snap.history == []
        assert snap.predicate_counts == {}
        assert snap.concept_count == 0

    def test_all_current(self) -> None:
        edges = [
            _make_edge(assertion_id="a1"),
            _make_edge(assertion_id="a2"),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        assert snap.total_current == 2
        assert snap.total_history == 0

    def test_all_history(self) -> None:
        edges = [
            _make_edge(is_current=False, assertion_status="retracted", assertion_id="a1"),
            _make_edge(is_current=False, assertion_status="superseded", assertion_id="a2"),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        assert snap.total_current == 0
        assert snap.total_history == 2
        assert snap.concept_count == 0  # only current edges count

    def test_history_excluded_from_metadata(self) -> None:
        """Predicate counts and concept count only include current edges."""
        edges = [
            _make_edge(
                source="A", target="B", predicate="supplies_to", is_current=True, assertion_id="a1"
            ),
            _make_edge(
                source="C",
                target="D",
                predicate="competes_with",
                is_current=False,
                assertion_status="retracted",
                assertion_id="a2",
            ),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        assert snap.predicate_counts == {"supplies_to": 1}
        assert snap.concept_count == 2  # A, B only

    def test_signs_carried_through(self) -> None:
        edges = [
            _make_edge(predicate="competes_with", assertion_id="a1"),
            _make_edge(predicate="supplies_to", assertion_id="a2"),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        signs = {r.predicate: r.sign for r in snap.current}
        assert signs["competes_with"] == -1
        assert signs["supplies_to"] == 1

    def test_to_dict_summary(self) -> None:
        edges = [
            _make_edge(assertion_id="a1"),
            _make_edge(is_current=False, assertion_status="retracted", assertion_id="a2"),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        d = snap.to_dict()
        assert d["total_current"] == 1
        assert d["total_history"] == 1
        assert isinstance(d["computed_at"], str)

    def test_snapshot_properties(self) -> None:
        edges = [
            _make_edge(assertion_id="a1"),
            _make_edge(assertion_id="a2"),
            _make_edge(is_current=False, assertion_status="retracted", assertion_id="a3"),
        ]
        snap = build_structural_snapshot(edges, now=NOW)
        assert snap.total_current == 2
        assert snap.total_history == 1


# -- Dataclass tests -------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass invariants."""

    def test_relation_frozen(self) -> None:
        rel = translate_derived_edge(_make_edge())
        try:
            rel.confidence = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_snapshot_frozen(self) -> None:
        snap = build_structural_snapshot([], now=NOW)
        try:
            snap.concept_count = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass
