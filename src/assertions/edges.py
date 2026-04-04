"""Derive graph-ready edges, exposures, and path-cache inputs from assertions.

Translates resolved assertions into structures that graph and structural
consumers read directly. Downstream logic never touches raw claims —
edges are derived entirely from assertion state.

Edge types:
    - Current: active assertions above confidence threshold
    - History: all assertions including retracted/superseded (temporal)

Exposure: per-concept aggregate of how many assertions touch it,
weighted by confidence.

Path-cache inputs: pre-computed adjacency for BFS/traversal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.assertions.schemas import ResolvedAssertion

# -- Configuration defaults -------------------------------------------------

DEFAULT_EDGE_CONFIDENCE_THRESHOLD = 0.3  # min confidence for current edges
DEFAULT_EXPOSURE_THRESHOLD = 0.1  # min confidence to count in exposure


# -- DerivedEdge -----------------------------------------------------------


@dataclass(frozen=True)
class DerivedEdge:
    """A graph-ready relationship edge derived from a resolved assertion.

    Carries the assertion_id as lineage so consumers can trace back
    to the supporting/contradicting evidence.

    Attributes:
        source_concept_id: Subject of the relationship.
        target_concept_id: Object of the relationship.
        predicate: Relationship type (e.g., "supplies_to").
        confidence: Assertion confidence (0-1).
        assertion_id: Source assertion for lineage.
        is_current: Whether this edge represents current belief.
        assertion_status: Status of the source assertion.
        valid_from: When the relationship became true.
        valid_to: When it ceased to be true (None = ongoing).
        support_count: Evidence support count.
        contradiction_count: Evidence contradiction count.
        source_diversity: Distinct source types.
        metadata: Extensible metadata.
    """

    source_concept_id: str
    target_concept_id: str
    predicate: str
    confidence: float
    assertion_id: str
    is_current: bool = True
    assertion_status: str = "active"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    support_count: int = 0
    contradiction_count: int = 0
    source_diversity: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# -- ConceptExposure -------------------------------------------------------


@dataclass(frozen=True)
class ConceptExposure:
    """Aggregate exposure of a concept through its assertion network.

    Gives downstream consumers a quick signal for "how much evidence
    do we have about this entity?" without re-querying assertions.

    Attributes:
        concept_id: The concept.
        total_assertions: Number of assertions touching this concept.
        active_assertions: Number of active assertions.
        disputed_assertions: Number of disputed assertions.
        weighted_confidence: Sum of confidences across active assertions.
        as_subject_count: Times appearing as subject.
        as_object_count: Times appearing as object.
        predicates: Distinct predicates involving this concept.
    """

    concept_id: str
    total_assertions: int = 0
    active_assertions: int = 0
    disputed_assertions: int = 0
    weighted_confidence: float = 0.0
    as_subject_count: int = 0
    as_object_count: int = 0
    predicates: frozenset[str] = field(default_factory=frozenset)


# -- PathCacheEntry --------------------------------------------------------


@dataclass(frozen=True)
class PathCacheEntry:
    """Pre-computed adjacency for a concept, ready for BFS/traversal.

    Attributes:
        concept_id: The source concept.
        outgoing: Edges where this concept is subject.
        incoming: Edges where this concept is object.
    """

    concept_id: str
    outgoing: tuple[DerivedEdge, ...] = ()
    incoming: tuple[DerivedEdge, ...] = ()

    @property
    def degree(self) -> int:
        """Total edge count (outgoing + incoming)."""
        return len(self.outgoing) + len(self.incoming)

    @property
    def neighbors(self) -> frozenset[str]:
        """All directly connected concept IDs."""
        out = {e.target_concept_id for e in self.outgoing}
        inc = {e.source_concept_id for e in self.incoming}
        return frozenset(out | inc)


# -- Derivation functions --------------------------------------------------


def derive_edge(
    assertion: ResolvedAssertion,
    *,
    confidence_threshold: float = DEFAULT_EDGE_CONFIDENCE_THRESHOLD,
) -> DerivedEdge | None:
    """Derive a graph edge from a single assertion.

    Returns None if the assertion has no object (unary) or falls
    below the confidence threshold for current edges.

    Retracted/superseded assertions produce history edges (is_current=False).
    """
    if assertion.object_concept_id is None:
        return None

    is_current = (
        assertion.status == "active"
        and assertion.confidence >= confidence_threshold
    )
    # History edges include everything with an object concept
    if not is_current and assertion.status not in ("retracted", "superseded", "disputed"):
        return None

    return DerivedEdge(
        source_concept_id=assertion.subject_concept_id,
        target_concept_id=assertion.object_concept_id,
        predicate=assertion.predicate,
        confidence=assertion.confidence,
        assertion_id=assertion.assertion_id,
        is_current=is_current,
        assertion_status=assertion.status,
        valid_from=assertion.valid_from,
        valid_to=assertion.valid_to,
        support_count=assertion.support_count,
        contradiction_count=assertion.contradiction_count,
        source_diversity=assertion.source_diversity,
        metadata=assertion.metadata,
    )


def derive_edges(
    assertions: list[ResolvedAssertion],
    *,
    confidence_threshold: float = DEFAULT_EDGE_CONFIDENCE_THRESHOLD,
) -> tuple[list[DerivedEdge], list[DerivedEdge]]:
    """Derive current and history edges from a list of assertions.

    Returns (current_edges, history_edges). Current edges are active
    assertions above the confidence threshold. History edges include
    all non-current edges (retracted, superseded, disputed, or below
    threshold).
    """
    current: list[DerivedEdge] = []
    history: list[DerivedEdge] = []

    for assertion in assertions:
        edge = derive_edge(assertion, confidence_threshold=confidence_threshold)
        if edge is None:
            continue
        if edge.is_current:
            current.append(edge)
        else:
            history.append(edge)

    return current, history


def compute_exposures(
    assertions: list[ResolvedAssertion],
    *,
    exposure_threshold: float = DEFAULT_EXPOSURE_THRESHOLD,
) -> dict[str, ConceptExposure]:
    """Compute per-concept exposure from assertions.

    Aggregates how many assertions touch each concept (as subject
    or object), counting active/disputed/total and summing weighted
    confidence for active assertions.
    """
    # Accumulate per-concept stats
    stats: dict[str, dict[str, Any]] = {}

    def _ensure(cid: str) -> dict[str, Any]:
        if cid not in stats:
            stats[cid] = {
                "total": 0,
                "active": 0,
                "disputed": 0,
                "weighted_conf": 0.0,
                "as_subject": 0,
                "as_object": 0,
                "predicates": set(),
            }
        return stats[cid]

    for a in assertions:
        if a.confidence < exposure_threshold:
            continue

        s = _ensure(a.subject_concept_id)
        s["total"] += 1
        s["as_subject"] += 1
        s["predicates"].add(a.predicate)
        if a.status == "active":
            s["active"] += 1
            s["weighted_conf"] += a.confidence
        elif a.status == "disputed":
            s["disputed"] += 1

        if a.object_concept_id:
            o = _ensure(a.object_concept_id)
            o["total"] += 1
            o["as_object"] += 1
            o["predicates"].add(a.predicate)
            if a.status == "active":
                o["active"] += 1
                o["weighted_conf"] += a.confidence
            elif a.status == "disputed":
                o["disputed"] += 1

    return {
        cid: ConceptExposure(
            concept_id=cid,
            total_assertions=s["total"],
            active_assertions=s["active"],
            disputed_assertions=s["disputed"],
            weighted_confidence=round(s["weighted_conf"], 4),
            as_subject_count=s["as_subject"],
            as_object_count=s["as_object"],
            predicates=frozenset(s["predicates"]),
        )
        for cid, s in stats.items()
    }


def build_path_cache(
    current_edges: list[DerivedEdge],
) -> dict[str, PathCacheEntry]:
    """Build per-concept adjacency cache from current edges.

    Only uses current (active, above-threshold) edges for path
    computation — history edges are excluded from traversal.
    """
    outgoing: dict[str, list[DerivedEdge]] = {}
    incoming: dict[str, list[DerivedEdge]] = {}

    for edge in current_edges:
        outgoing.setdefault(edge.source_concept_id, []).append(edge)
        incoming.setdefault(edge.target_concept_id, []).append(edge)

    all_concepts = set(outgoing) | set(incoming)
    return {
        cid: PathCacheEntry(
            concept_id=cid,
            outgoing=tuple(
                sorted(outgoing.get(cid, []), key=lambda e: -e.confidence)
            ),
            incoming=tuple(
                sorted(incoming.get(cid, []), key=lambda e: -e.confidence)
            ),
        )
        for cid in all_concepts
    }
