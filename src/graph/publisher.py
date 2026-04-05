"""Publish structural lane outputs keyed by manifest.

Orchestrates structural lane publication: transforms scored paths
and thematic baskets into publishable payloads with explanation-ready
path summaries, checks lane health, and produces a result for
manifest assembly.

This is the structural-lane analog of src/filing/publisher.py and
src/narrative/publisher.py.

Publication flow:
    1. Check lane health — abort if BLOCKED
    2. Build path explanation payloads (preserving score ingredients
       and assertion lineage so consumers can explain surfaced names)
    3. Build basket payloads (summary of each thematic basket)
    4. Return publishable result for manifest inclusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.graph.baskets import BasketMember, ThematicBasket
from src.graph.path_scoring import ScoredPath
from src.publish.lane_health import LaneHealthStatus, PublishReadiness


# -- Path explanation payload --------------------------------------------------


@dataclass(frozen=True)
class PathExplanation:
    """Publishable path summary with score ingredients and lineage.

    Answers "why did this name surface?" without live recomputation.
    Each explanation preserves the full score breakdown and assertion
    IDs for audit trails.

    Attributes:
        source_concept_id: Path start (theme or seed concept).
        target_concept_id: Path end (surfaced name).
        hops: Path length (1 or 2).
        path_score: Composite score.
        path_sign: Net sign (+1 cooperative, -1 competitive).
        intermediate_concept_id: Middle node for 2-hop paths.
        edge_predicates: Relationship types along the path.
        confidence_product: Product of edge confidences.
        freshness_product: Product of freshness factors.
        corroboration_product: Product of corroboration factors.
        hop_decay: Decay factor for path length.
        assertion_ids: Lineage to supporting assertions.
    """

    source_concept_id: str
    target_concept_id: str
    hops: int
    path_score: float
    path_sign: int
    intermediate_concept_id: str | None
    edge_predicates: list[str]
    confidence_product: float
    freshness_product: float
    corroboration_product: float
    hop_decay: float
    assertion_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_concept_id": self.source_concept_id,
            "target_concept_id": self.target_concept_id,
            "hops": self.hops,
            "path_score": round(self.path_score, 4),
            "path_sign": self.path_sign,
            "intermediate_concept_id": self.intermediate_concept_id,
            "edge_predicates": self.edge_predicates,
            "confidence_product": round(self.confidence_product, 4),
            "freshness_product": round(self.freshness_product, 4),
            "corroboration_product": round(self.corroboration_product, 4),
            "hop_decay": round(self.hop_decay, 4),
            "assertion_ids": self.assertion_ids,
        }


# -- Basket payload ------------------------------------------------------------


@dataclass(frozen=True)
class BasketPayload:
    """Publishable thematic basket summary.

    Carries top members and counts for consumer UIs. Full path
    details are in the companion PathExplanation objects.

    Attributes:
        source_concept_id: Theme or seed concept.
        beneficiary_count: Total positive-sign members.
        at_risk_count: Total negative-sign members.
        first_order_count: Members reachable in 1 hop.
        second_order_count: Members reachable only in 2 hops.
        top_beneficiaries: Top N beneficiaries serialized.
        top_at_risk: Top N at-risk members serialized.
    """

    source_concept_id: str
    beneficiary_count: int
    at_risk_count: int
    first_order_count: int
    second_order_count: int
    top_beneficiaries: list[dict[str, Any]] = field(default_factory=list)
    top_at_risk: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_concept_id": self.source_concept_id,
            "beneficiary_count": self.beneficiary_count,
            "at_risk_count": self.at_risk_count,
            "first_order_count": self.first_order_count,
            "second_order_count": self.second_order_count,
            "top_beneficiaries": self.top_beneficiaries,
            "top_at_risk": self.top_at_risk,
        }


# -- Publication result --------------------------------------------------------


@dataclass
class StructuralPublicationResult:
    """Result of a structural lane publication attempt.

    Attributes:
        published: Whether publication succeeded.
        lane_health: The health check result.
        path_explanations: Explanation-ready path summaries.
        basket_payloads: Thematic basket summaries.
        object_count: Total publishable objects produced.
        block_reason: Why publication was blocked (if applicable).
    """

    published: bool
    lane_health: LaneHealthStatus
    path_explanations: list[PathExplanation] = field(default_factory=list)
    basket_payloads: list[BasketPayload] = field(default_factory=list)
    object_count: int = 0
    block_reason: str | None = None


# -- Payload builders (stateless) ----------------------------------------------

DEFAULT_TOP_N = 10


def build_path_explanation(path: ScoredPath) -> PathExplanation:
    """Convert a ScoredPath into a publishable PathExplanation.

    Extracts score ingredients and assertion IDs from the scored
    edges, producing a self-contained explanation payload.
    """
    return PathExplanation(
        source_concept_id=path.source_concept_id,
        target_concept_id=path.target_concept_id,
        hops=path.hops,
        path_score=path.path_score,
        path_sign=path.path_sign,
        intermediate_concept_id=path.intermediate_concept_id,
        edge_predicates=[e.relation.predicate for e in path.edges],
        confidence_product=path.breakdown.confidence_product,
        freshness_product=path.breakdown.freshness_product,
        corroboration_product=path.breakdown.corroboration_product,
        hop_decay=path.breakdown.hop_decay,
        assertion_ids=[e.relation.assertion_id for e in path.edges],
    )


def build_basket_payload(
    basket: ThematicBasket,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> BasketPayload:
    """Convert a ThematicBasket into a publishable BasketPayload.

    Serializes the top N beneficiaries and at-risk members for
    consumer UIs. Full path details live in PathExplanation objects.
    """
    return BasketPayload(
        source_concept_id=basket.source_concept_id,
        beneficiary_count=len(basket.beneficiaries),
        at_risk_count=len(basket.at_risk),
        first_order_count=basket.first_order_count,
        second_order_count=basket.second_order_count,
        top_beneficiaries=[
            m.to_dict() for m in basket.beneficiaries[:top_n]
        ],
        top_at_risk=[m.to_dict() for m in basket.at_risk[:top_n]],
    )


# -- Publisher -----------------------------------------------------------------


def prepare_structural_publication(
    paths: list[ScoredPath],
    baskets: list[ThematicBasket],
    lane_health: LaneHealthStatus,
    *,
    top_n: int = DEFAULT_TOP_N,
    now: datetime | None = None,
) -> StructuralPublicationResult:
    """Prepare structural lane outputs for manifest publication.

    Checks lane health, builds explanation-ready path summaries and
    basket payloads. Returns a result the caller can use to create
    manifest objects.

    Does NOT persist anything — the caller handles manifest creation
    and object insertion.

    Args:
        paths: Scored structural paths to publish as explanations.
        baskets: Thematic baskets to publish as summaries.
        lane_health: Pre-computed lane health status.
        top_n: How many top members to include per basket.
        now: Current time (unused, kept for API symmetry).

    Returns:
        StructuralPublicationResult with payloads and counts.
    """
    if lane_health.readiness == PublishReadiness.BLOCKED:
        return StructuralPublicationResult(
            published=False,
            lane_health=lane_health,
            block_reason=(
                f"Lane {lane_health.lane} is blocked: "
                f"freshness={lane_health.freshness.value}, "
                f"quality={lane_health.quality.value}, "
                f"quarantine={lane_health.quarantine.value}"
            ),
        )

    path_explanations = [build_path_explanation(p) for p in paths]
    basket_payloads = [
        build_basket_payload(b, top_n=top_n) for b in baskets
    ]

    return StructuralPublicationResult(
        published=True,
        lane_health=lane_health,
        path_explanations=path_explanations,
        basket_payloads=basket_payloads,
        object_count=len(path_explanations) + len(basket_payloads),
    )
