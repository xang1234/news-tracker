"""1-hop and 2-hop path scoring with decomposed factors.

Scores structural paths using four decomposed factors per edge:
confidence, freshness, corroboration, and sign. Path scores
compound across hops with configurable decay.

This is NOT sentiment propagation — it evaluates the strength
of structural connections, not the impact of a signal. The
outputs are explanation-ready: each scored path carries a full
breakdown of how the factors contributed.

Scoring formula per edge:
    edge_score = confidence * freshness_factor * corroboration_factor

Path composite:
    path_score = product(edge_scores) * hop_decay^(hops-1)
    path_sign = product(edge_signs)

All functions are stateless — the caller provides a
StructuralSnapshot and a source concept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.graph.structural import StructuralRelation, StructuralSnapshot


# -- Configuration defaults ---------------------------------------------------

DEFAULT_HOP_DECAY = 0.7
DEFAULT_FRESHNESS_HALF_LIFE_DAYS = 90.0
DEFAULT_FRESHNESS_UNKNOWN = 0.5
DEFAULT_DIVERSITY_CEILING = 3
DEFAULT_VOLUME_CEILING = 5
DEFAULT_MIN_PATH_SCORE = 0.01


# -- Scored edge ---------------------------------------------------------------


@dataclass(frozen=True)
class ScoredEdge:
    """A single edge scored with decomposed factors.

    Attributes:
        relation: The underlying structural relation.
        freshness_factor: 0-1 from temporal decay.
        corroboration_factor: 0-1 from evidence diversity/volume/agreement.
        edge_score: confidence * freshness * corroboration.
    """

    relation: StructuralRelation
    freshness_factor: float
    corroboration_factor: float
    edge_score: float


# -- Path score breakdown ------------------------------------------------------


@dataclass(frozen=True)
class PathScoreBreakdown:
    """Decomposed path score for explanation UIs.

    Attributes:
        confidence_product: Product of edge confidences.
        freshness_product: Product of freshness factors.
        corroboration_product: Product of corroboration factors.
        hop_decay: Decay factor applied for path length.
        composite: Final path score.
    """

    confidence_product: float
    freshness_product: float
    corroboration_product: float
    hop_decay: float
    composite: float


# -- Scored path ---------------------------------------------------------------


@dataclass(frozen=True)
class ScoredPath:
    """A scored 1-hop or 2-hop path through the structural graph.

    Attributes:
        source_concept_id: Path start.
        target_concept_id: Path end.
        hops: Number of edges (1 or 2).
        path_score: Composite score (0-1, higher = stronger connection).
        path_sign: Net sign (+1 cooperative, -1 competitive).
        breakdown: Decomposed score factors.
        edges: Scored edges in path order.
        intermediate_concept_id: The middle node (2-hop paths only).
    """

    source_concept_id: str
    target_concept_id: str
    hops: int
    path_score: float
    path_sign: int
    breakdown: PathScoreBreakdown
    edges: list[ScoredEdge] = field(default_factory=list)
    intermediate_concept_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for publication and explanation UIs."""
        return {
            "source_concept_id": self.source_concept_id,
            "target_concept_id": self.target_concept_id,
            "hops": self.hops,
            "path_score": round(self.path_score, 4),
            "path_sign": self.path_sign,
            "breakdown": {
                "confidence_product": round(self.breakdown.confidence_product, 4),
                "freshness_product": round(self.breakdown.freshness_product, 4),
                "corroboration_product": round(
                    self.breakdown.corroboration_product, 4
                ),
                "hop_decay": round(self.breakdown.hop_decay, 4),
            },
            "intermediate_concept_id": self.intermediate_concept_id,
            "edge_predicates": [e.relation.predicate for e in self.edges],
        }


# -- Factor computation (stateless) -------------------------------------------


def compute_freshness_factor(
    valid_from: datetime | None,
    now: datetime,
    *,
    half_life_days: float = DEFAULT_FRESHNESS_HALF_LIFE_DAYS,
) -> float:
    """Exponential decay from relationship start to now.

    More recent relationships score higher. Unknown validity
    gets a moderate default.
    """
    if valid_from is None:
        return DEFAULT_FRESHNESS_UNKNOWN
    days = max(0.0, (now - valid_from).total_seconds() / 86400)
    return math.exp(-0.693 * days / half_life_days)


def compute_corroboration_factor(
    support_count: int,
    contradiction_count: int,
    source_diversity: int,
    *,
    diversity_ceiling: int = DEFAULT_DIVERSITY_CEILING,
    volume_ceiling: int = DEFAULT_VOLUME_CEILING,
) -> float:
    """Evidence quality from diversity, volume, and agreement.

    Three sub-factors multiplied:
        - support_ratio: support / (support + contradiction)
        - diversity: min(1.0, source_diversity / ceiling)
        - volume: min(1.0, support / ceiling)
    """
    total = support_count + contradiction_count
    if total == 0:
        return 0.0

    support_ratio = support_count / total
    diversity = min(1.0, source_diversity / diversity_ceiling)
    volume = min(1.0, support_count / volume_ceiling)
    return support_ratio * diversity * volume


# -- Edge scoring --------------------------------------------------------------


def score_edge(
    relation: StructuralRelation,
    now: datetime,
    *,
    half_life_days: float = DEFAULT_FRESHNESS_HALF_LIFE_DAYS,
    diversity_ceiling: int = DEFAULT_DIVERSITY_CEILING,
    volume_ceiling: int = DEFAULT_VOLUME_CEILING,
) -> ScoredEdge:
    """Score a single structural relation with decomposed factors."""
    freshness = compute_freshness_factor(
        relation.valid_from, now, half_life_days=half_life_days,
    )
    corroboration = compute_corroboration_factor(
        relation.support_count,
        relation.contradiction_count,
        relation.source_diversity,
        diversity_ceiling=diversity_ceiling,
        volume_ceiling=volume_ceiling,
    )
    return ScoredEdge(
        relation=relation,
        freshness_factor=round(freshness, 4),
        corroboration_factor=round(corroboration, 4),
        edge_score=round(relation.confidence * freshness * corroboration, 4),
    )


# -- Path scoring --------------------------------------------------------------


def _build_adjacency(
    relations: list[StructuralRelation],
) -> dict[str, list[StructuralRelation]]:
    """Build outgoing adjacency index from current relations."""
    adj: dict[str, list[StructuralRelation]] = {}
    for rel in relations:
        adj.setdefault(rel.source_concept_id, []).append(rel)
    return adj


def _make_path(
    scored_edges: list[ScoredEdge],
    hop_decay: float,
) -> ScoredPath:
    """Construct a ScoredPath from a sequence of scored edges."""
    hops = len(scored_edges)
    decay = hop_decay ** max(0, hops - 1)

    conf_prod = math.prod(e.relation.confidence for e in scored_edges)
    fresh_prod = math.prod(e.freshness_factor for e in scored_edges)
    corr_prod = math.prod(e.corroboration_factor for e in scored_edges)
    sign = math.prod(e.relation.sign for e in scored_edges)

    composite = math.prod(e.edge_score for e in scored_edges) * decay

    return ScoredPath(
        source_concept_id=scored_edges[0].relation.source_concept_id,
        target_concept_id=scored_edges[-1].relation.target_concept_id,
        hops=hops,
        path_score=round(composite, 4),
        path_sign=sign,
        breakdown=PathScoreBreakdown(
            confidence_product=round(conf_prod, 4),
            freshness_product=round(fresh_prod, 4),
            corroboration_product=round(corr_prod, 4),
            hop_decay=round(decay, 4),
            composite=round(composite, 4),
        ),
        edges=list(scored_edges),
        intermediate_concept_id=(
            scored_edges[0].relation.target_concept_id
            if hops == 2
            else None
        ),
    )


def score_paths_from(
    snapshot: StructuralSnapshot,
    source_concept_id: str,
    *,
    max_hops: int = 2,
    hop_decay: float = DEFAULT_HOP_DECAY,
    half_life_days: float = DEFAULT_FRESHNESS_HALF_LIFE_DAYS,
    diversity_ceiling: int = DEFAULT_DIVERSITY_CEILING,
    volume_ceiling: int = DEFAULT_VOLUME_CEILING,
    min_path_score: float = DEFAULT_MIN_PATH_SCORE,
    now: datetime | None = None,
) -> list[ScoredPath]:
    """Find and score all 1-hop and 2-hop paths from a source concept.

    Uses the snapshot's current relations to build adjacency, scores
    each edge, and constructs paths with composite scores. Paths
    below min_path_score are excluded.

    Args:
        snapshot: Structural state with current relations.
        source_concept_id: Starting concept for path search.
        max_hops: Maximum path length (1 or 2).
        hop_decay: Score decay per additional hop.
        half_life_days: Half-life for freshness exponential decay.
        diversity_ceiling: Max source_diversity for full credit.
        volume_ceiling: Max support_count for full credit.
        min_path_score: Minimum composite score to include a path.
        now: Current time for freshness computation.

    Returns:
        List of ScoredPath sorted by path_score descending.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    adj = _build_adjacency(snapshot.current)
    score_kwargs = dict(
        half_life_days=half_life_days,
        diversity_ceiling=diversity_ceiling,
        volume_ceiling=volume_ceiling,
    )

    paths: list[ScoredPath] = []

    # 1-hop paths
    for rel in adj.get(source_concept_id, []):
        se = score_edge(rel, now, **score_kwargs)
        path = _make_path([se], hop_decay)
        if path.path_score >= min_path_score:
            paths.append(path)

    # 2-hop paths
    if max_hops >= 2:
        for rel1 in adj.get(source_concept_id, []):
            se1 = score_edge(rel1, now, **score_kwargs)
            mid = rel1.target_concept_id
            # Don't loop back to source
            if mid == source_concept_id:
                continue
            for rel2 in adj.get(mid, []):
                # Don't loop back to source or intermediate
                if rel2.target_concept_id in (source_concept_id, mid):
                    continue
                se2 = score_edge(rel2, now, **score_kwargs)
                path = _make_path([se1, se2], hop_decay)
                if path.path_score >= min_path_score:
                    paths.append(path)

    paths.sort(key=lambda p: -p.path_score)
    return paths
