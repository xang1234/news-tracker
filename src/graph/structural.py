"""Assertion-derived typed structural relations with history.

Translates DerivedEdges from the assertion layer into typed
structural relations that the graph subsystem can consume for
traversal and propagation.

Key differences from manual CausalEdge:
    - Assertion lineage (assertion_id, support/contradiction counts)
    - Temporal validity (valid_from/valid_to)
    - Current vs history distinction
    - Broader predicate set (not limited to 5 manual types)
    - Sign mapping for propagation direction

Structural relations coexist with manual CausalEdges — they are
an evidence-backed layer on top of, not a replacement for, the
existing manual graph.

Translation flow:
    1. Receive DerivedEdges from derive_edges()
    2. Map each edge's predicate to a propagation sign (+1/-1)
    3. Split into current (live) and history (temporal)
    4. Return a StructuralSnapshot with metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.assertions.edges import DerivedEdge

# -- Predicate sign mapping ---------------------------------------------------

# Positive (+1): target benefits when source improves.
# Negative (-1): target suffers when source improves (competitive).
# Covers all predicates from concept_schemas.VALID_RELATIONSHIP_TYPES
# and graph/schemas.VALID_RELATION_TYPES.
PREDICATE_SIGNS: dict[str, int] = {
    "supplies_to": 1,
    "customer_of": 1,
    "competes_with": -1,
    "depends_on": 1,
    "drives": 1,
    "blocks": -1,
    "uses_technology": 1,
    "develops_technology": 1,
    "produces": 1,
    "consumes": 1,
    "component_of": 1,
    "contains_component": 1,
    "subsidiary_of": 1,
    "parent_of": 1,
    "operates_facility": 1,
    "located_at": 1,
}

DEFAULT_SIGN = 1


# -- StructuralRelation -------------------------------------------------------


@dataclass(frozen=True)
class StructuralRelation:
    """A typed structural relation derived from a resolved assertion.

    Carries assertion lineage, temporal validity, and evidence
    metadata so downstream consumers can trace any structural
    edge back to its supporting evidence.

    Attributes:
        source_concept_id: Subject of the relationship.
        target_concept_id: Object of the relationship.
        predicate: Relationship type (broader than manual CausalEdge).
        confidence: Assertion confidence (0-1).
        sign: Propagation sign (+1 positive, -1 competitive/blocking).
        assertion_id: Lineage to the source assertion.
        is_current: True for live graph edges, False for history.
        assertion_status: Status of the source assertion.
        valid_from: When the relationship became true.
        valid_to: When it ceased to be true (None = ongoing).
        support_count: Claims supporting this assertion.
        contradiction_count: Claims contradicting this assertion.
        source_diversity: Distinct source types in evidence.
        metadata: Extensible metadata.
    """

    source_concept_id: str
    target_concept_id: str
    predicate: str
    confidence: float
    sign: int
    assertion_id: str
    is_current: bool = True
    assertion_status: str = "active"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    support_count: int = 0
    contradiction_count: int = 0
    source_diversity: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for publication and graph consumers."""
        return {
            "source_concept_id": self.source_concept_id,
            "target_concept_id": self.target_concept_id,
            "predicate": self.predicate,
            "confidence": round(self.confidence, 4),
            "sign": self.sign,
            "assertion_id": self.assertion_id,
            "is_current": self.is_current,
            "assertion_status": self.assertion_status,
            "valid_from": (self.valid_from.isoformat() if self.valid_from else None),
            "valid_to": (self.valid_to.isoformat() if self.valid_to else None),
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "source_diversity": self.source_diversity,
        }


# -- StructuralSnapshot -------------------------------------------------------


@dataclass(frozen=True)
class StructuralSnapshot:
    """Complete assertion-derived structural state at a point in time.

    Separates current (live for traversal) from history (temporal
    audit trail). Metadata enables quick inspection without scanning
    all relations.

    Attributes:
        current: Live relations for traversal and propagation.
        history: All non-current relations (retracted, superseded, etc.).
        predicate_counts: Predicate → count of current relations.
        concept_count: Distinct concepts in the current graph.
        computed_at: When this snapshot was built.
    """

    current: list[StructuralRelation] = field(default_factory=list)
    history: list[StructuralRelation] = field(default_factory=list)
    predicate_counts: dict[str, int] = field(default_factory=dict)
    concept_count: int = 0
    computed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_current(self) -> int:
        """Number of live structural relations."""
        return len(self.current)

    @property
    def total_history(self) -> int:
        """Number of historical relations."""
        return len(self.history)

    def to_dict(self) -> dict[str, Any]:
        """Summary serialization (without full relation lists)."""
        return {
            "total_current": self.total_current,
            "total_history": self.total_history,
            "predicate_counts": self.predicate_counts,
            "concept_count": self.concept_count,
            "computed_at": self.computed_at.isoformat(),
        }


# -- Translation functions (stateless) ----------------------------------------


def get_predicate_sign(predicate: str) -> int:
    """Get the propagation sign for a predicate.

    Returns +1 (positive/cooperative) or -1 (negative/competitive).
    Unknown predicates default to +1.
    """
    return PREDICATE_SIGNS.get(predicate, DEFAULT_SIGN)


def translate_derived_edge(edge: DerivedEdge) -> StructuralRelation:
    """Translate a DerivedEdge into a StructuralRelation.

    Adds the propagation sign based on the predicate and carries
    through all assertion lineage and temporal metadata.
    """
    return StructuralRelation(
        source_concept_id=edge.source_concept_id,
        target_concept_id=edge.target_concept_id,
        predicate=edge.predicate,
        confidence=edge.confidence,
        sign=get_predicate_sign(edge.predicate),
        assertion_id=edge.assertion_id,
        is_current=edge.is_current,
        assertion_status=edge.assertion_status,
        valid_from=edge.valid_from,
        valid_to=edge.valid_to,
        support_count=edge.support_count,
        contradiction_count=edge.contradiction_count,
        source_diversity=edge.source_diversity,
        metadata=dict(edge.metadata),
    )


def build_structural_snapshot(
    edges: list[DerivedEdge],
    *,
    now: datetime | None = None,
) -> StructuralSnapshot:
    """Build a structural snapshot from assertion-derived edges.

    Translates all edges, splits into current/history, and computes
    metadata. The caller typically obtains edges by calling
    derive_edges() from src/assertions/edges.

    Args:
        edges: DerivedEdges (both current and history).
        now: Timestamp for the snapshot.

    Returns:
        StructuralSnapshot with current/history split and metadata.
    """
    if now is None:
        now = datetime.now(UTC)

    current: list[StructuralRelation] = []
    history: list[StructuralRelation] = []
    predicate_counts: dict[str, int] = {}
    concepts: set[str] = set()

    for edge in edges:
        relation = translate_derived_edge(edge)
        if relation.is_current:
            current.append(relation)
            predicate_counts[relation.predicate] = predicate_counts.get(relation.predicate, 0) + 1
            concepts.add(relation.source_concept_id)
            concepts.add(relation.target_concept_id)
        else:
            history.append(relation)

    return StructuralSnapshot(
        current=current,
        history=history,
        predicate_counts=predicate_counts,
        concept_count=len(concepts),
        computed_at=now,
    )
