"""Thematic baskets and second-order beneficiary outputs.

Assembles scored structural paths into decision-oriented baskets.
Each basket member is traceable to specific paths, preserving
signed relationships and hop distance so consumers can distinguish
first-order from second-order names and beneficiaries from
at-risk entities.

Basket construction:
    1. Group scored paths by target concept
    2. Pick the best path per target for primary classification
    3. Classify role by path sign: +1 → beneficiary, -1 → at_risk
    4. Sort each group by best score descending
    5. Return ThematicBasket with metadata

All functions are stateless — the caller provides pre-scored
paths from score_paths_from().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.graph.path_scoring import ScoredPath

# -- Basket roles -------------------------------------------------------------

ROLE_BENEFICIARY = "beneficiary"
ROLE_AT_RISK = "at_risk"


# -- BasketMember --------------------------------------------------------------


@dataclass(frozen=True)
class BasketMember:
    """A concept in a thematic basket with path provenance.

    Attributes:
        concept_id: The target concept reached via structural paths.
        role: "beneficiary" (positive sign) or "at_risk" (negative sign).
        best_score: Highest path score among all paths to this concept.
        best_sign: Sign of the best-scoring path.
        min_hops: Closest path distance (1 or 2).
        positive_paths: Count of positive-sign paths.
        negative_paths: Count of negative-sign paths.
        paths: All paths to this concept, best first.
    """

    concept_id: str
    role: str
    best_score: float
    best_sign: int
    min_hops: int
    positive_paths: int
    negative_paths: int
    paths: list[ScoredPath] = field(default_factory=list)

    @property
    def path_count(self) -> int:
        """Total paths reaching this concept."""
        return len(self.paths)

    @property
    def is_second_order(self) -> bool:
        """True if the closest path is 2-hop (second-order name)."""
        return self.min_hops >= 2

    @property
    def has_mixed_signals(self) -> bool:
        """True if both positive and negative paths exist."""
        return self.positive_paths > 0 and self.negative_paths > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for publication payloads."""
        return {
            "concept_id": self.concept_id,
            "role": self.role,
            "best_score": round(self.best_score, 4),
            "best_sign": self.best_sign,
            "min_hops": self.min_hops,
            "path_count": self.path_count,
            "positive_paths": self.positive_paths,
            "negative_paths": self.negative_paths,
            "is_second_order": self.is_second_order,
            "has_mixed_signals": self.has_mixed_signals,
        }


# -- ThematicBasket ------------------------------------------------------------


@dataclass(frozen=True)
class ThematicBasket:
    """Decision-oriented basket of concepts connected to a theme.

    Groups members by role (beneficiary vs at_risk) with
    first-order / second-order counts for quick inspection.

    Attributes:
        source_concept_id: The theme or seed concept.
        beneficiaries: Positive-sign members, best score first.
        at_risk: Negative-sign members, best score first.
        first_order_count: Members reachable in 1 hop.
        second_order_count: Members reachable only in 2 hops.
        computed_at: When this basket was assembled.
    """

    source_concept_id: str
    beneficiaries: list[BasketMember] = field(default_factory=list)
    at_risk: list[BasketMember] = field(default_factory=list)
    first_order_count: int = 0
    second_order_count: int = 0
    computed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def member_count(self) -> int:
        """Total unique concepts in the basket."""
        return len(self.beneficiaries) + len(self.at_risk)

    def to_dict(self) -> dict[str, Any]:
        """Summary serialization for publication."""
        return {
            "source_concept_id": self.source_concept_id,
            "beneficiary_count": len(self.beneficiaries),
            "at_risk_count": len(self.at_risk),
            "member_count": self.member_count,
            "first_order_count": self.first_order_count,
            "second_order_count": self.second_order_count,
            "computed_at": self.computed_at.isoformat(),
        }


# -- Basket builder (stateless) -----------------------------------------------


def _group_paths_by_target(
    paths: list[ScoredPath],
) -> dict[str, list[ScoredPath]]:
    """Group paths by target concept, sorted by score descending."""
    groups: dict[str, list[ScoredPath]] = {}
    for path in paths:
        groups.setdefault(path.target_concept_id, []).append(path)
    for target_paths in groups.values():
        target_paths.sort(key=lambda p: -p.path_score)
    return groups


def _build_member(
    concept_id: str,
    paths: list[ScoredPath],
) -> BasketMember:
    """Build a basket member from all paths to a concept.

    The best (highest-scoring) path determines the primary role.
    """
    best = paths[0]
    positive = sum(1 for p in paths if p.path_sign > 0)
    negative = sum(1 for p in paths if p.path_sign < 0)
    role = ROLE_BENEFICIARY if best.path_sign > 0 else ROLE_AT_RISK

    return BasketMember(
        concept_id=concept_id,
        role=role,
        best_score=best.path_score,
        best_sign=best.path_sign,
        min_hops=min(p.hops for p in paths),
        positive_paths=positive,
        negative_paths=negative,
        paths=list(paths),
    )


def build_thematic_basket(
    source_concept_id: str,
    paths: list[ScoredPath],
    *,
    now: datetime | None = None,
) -> ThematicBasket:
    """Assemble scored paths into a thematic basket.

    Groups paths by target concept, classifies each target as
    beneficiary or at_risk based on the best path's sign, and
    returns a structured basket with first/second-order counts.

    Args:
        source_concept_id: The theme or seed concept.
        paths: Pre-scored paths from score_paths_from().
        now: Timestamp for the basket.

    Returns:
        ThematicBasket with members sorted by best_score descending.
    """
    if now is None:
        now = datetime.now(UTC)

    groups = _group_paths_by_target(paths)

    beneficiaries: list[BasketMember] = []
    at_risk: list[BasketMember] = []

    for concept_id, target_paths in groups.items():
        member = _build_member(concept_id, target_paths)
        if member.role == ROLE_BENEFICIARY:
            beneficiaries.append(member)
        else:
            at_risk.append(member)

    beneficiaries.sort(key=lambda m: -m.best_score)
    at_risk.sort(key=lambda m: -m.best_score)

    all_members = beneficiaries + at_risk
    first_order = sum(1 for m in all_members if m.min_hops == 1)
    second_order = sum(1 for m in all_members if m.min_hops >= 2)

    return ThematicBasket(
        source_concept_id=source_concept_id,
        beneficiaries=beneficiaries,
        at_risk=at_risk,
        first_order_count=first_order,
        second_order_count=second_order,
        computed_at=now,
    )
