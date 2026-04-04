"""Assertion aggregation from supporting and contradicting claims.

Computes stable current-belief state from claim evidence with
explainable confidence decomposition. Every factor (freshness,
diversity, support ratio, review state) is visible in the
breakdown so downstream consumers can explain assertion state.

The aggregation function is stateless — no I/O. It takes claims
and links as input and returns an updated assertion. The caller
handles persistence.

Confidence formula:
    base = weighted_mean(claim_confidences * contribution_weights)
    freshness = exp(-decay * days_since_last_evidence)
    diversity = min(1.0, distinct_sources / diversity_target)
    support_ratio = support / (support + contradiction)
    review_bonus = 0.1 if any claim is review-approved

    confidence = base * freshness * diversity * support_ratio + review_bonus
    clamped to [0.0, 1.0]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from src.assertions.schemas import (
    AssertionClaimLink,
    ResolvedAssertion,
    make_assertion_id,
)
from src.claims.schemas import EvidenceClaim


# -- Configuration defaults -------------------------------------------------

DEFAULT_FRESHNESS_DECAY = 0.01  # exp decay lambda (per day)
DEFAULT_DIVERSITY_TARGET = 3  # distinct source types for full diversity credit
DEFAULT_REVIEW_BONUS = 0.1  # confidence boost for review-approved claims
DEFAULT_CONTRADICTION_PENALTY = 0.5  # weight of contradictions vs support
DEFAULT_DISPUTED_THRESHOLD = 0.6  # support ratio below this → disputed


# -- Confidence breakdown ---------------------------------------------------


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Explainable decomposition of assertion confidence.

    Each factor is a multiplier (0-1) except review_bonus which
    is additive. The final confidence is:
        clamp(base * freshness * diversity * support_ratio + review_bonus, 0, 1)

    Attributes:
        base: Weighted mean of claim confidences.
        freshness: Time decay factor (1.0 = just seen, decays toward 0).
        diversity: Source diversity factor (1.0 = meets diversity target).
        support_ratio: Proportion of support vs total evidence.
        review_bonus: Additive bonus for review-approved evidence.
        final: The clamped result.
    """

    base: float
    freshness: float
    diversity: float
    support_ratio: float
    review_bonus: float
    final: float


# -- Aggregation function ---------------------------------------------------


def aggregate_assertion(
    subject_concept_id: str,
    predicate: str,
    object_concept_id: str | None,
    claims: list[EvidenceClaim],
    links: list[AssertionClaimLink],
    *,
    now: datetime | None = None,
    freshness_decay: float = DEFAULT_FRESHNESS_DECAY,
    diversity_target: int = DEFAULT_DIVERSITY_TARGET,
    review_bonus: float = DEFAULT_REVIEW_BONUS,
    disputed_threshold: float = DEFAULT_DISPUTED_THRESHOLD,
) -> tuple[ResolvedAssertion, ConfidenceBreakdown]:
    """Aggregate claims into a resolved assertion with confidence breakdown.

    Takes the raw claims and their link types (support/contradiction)
    and computes aggregate confidence with explainable factors.

    Args:
        subject_concept_id: The subject concept.
        predicate: The relationship predicate.
        object_concept_id: The object concept (None for unary).
        claims: Evidence claims to aggregate.
        links: Claim links with types and weights.
        now: Current time (for freshness calc; defaults to utcnow).
        freshness_decay: Exponential decay lambda per day.
        diversity_target: Distinct source types for full diversity credit.
        review_bonus: Additive bonus for review-approved evidence.

    Returns:
        (assertion, breakdown) tuple with the aggregated assertion
        and its confidence decomposition.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    assertion_id = make_assertion_id(
        subject_concept_id, predicate, object_concept_id
    )

    # Build claim lookup
    claim_map: dict[str, EvidenceClaim] = {c.claim_id: c for c in claims}

    # Partition into support and contradiction
    support_claims: list[tuple[EvidenceClaim, AssertionClaimLink]] = []
    contradiction_claims: list[tuple[EvidenceClaim, AssertionClaimLink]] = []

    for lnk in links:
        claim = claim_map.get(lnk.claim_id)
        if claim is None or claim.status == "retracted":
            continue
        if lnk.link_type == "support":
            support_claims.append((claim, lnk))
        else:
            contradiction_claims.append((claim, lnk))

    all_evidence = support_claims + contradiction_claims
    support_count = len(support_claims)
    contradiction_count = len(contradiction_claims)
    total_evidence = support_count + contradiction_count

    # -- Base confidence: weighted mean of supporting claim confidences --
    if support_claims:
        weighted_sum = sum(
            c.confidence * lnk.contribution_weight
            for c, lnk in support_claims
        )
        weight_sum = sum(lnk.contribution_weight for _, lnk in support_claims)
        base = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    else:
        base = 0.0

    # -- Freshness: exponential decay from most recent evidence --
    timestamps = [
        c.source_published_at
        for c, _ in all_evidence
        if c.source_published_at is not None
    ]
    if timestamps:
        latest = max(timestamps)
        days_since = max(0.0, (now - latest).total_seconds() / 86400)
        freshness = math.exp(-freshness_decay * days_since)
        first_seen = min(timestamps)
        last_evidence = latest
    else:
        freshness = 0.5  # unknown recency gets moderate weight
        first_seen = None
        last_evidence = None

    # -- Source diversity: distinct source types --
    source_types = {c.source_type for c, _ in all_evidence}
    diversity_count = len(source_types)
    diversity = min(1.0, diversity_count / diversity_target) if diversity_target > 0 else 1.0

    # -- Support ratio: proportion of support vs total --
    if total_evidence > 0:
        support_ratio = support_count / total_evidence
    else:
        support_ratio = 0.0

    # -- Review bonus: any review-approved claim adds a bonus --
    has_review_approval = any(
        lnk.metadata.get("review_approved", False)
        for _, lnk in support_claims
    )
    actual_review_bonus = review_bonus if has_review_approval else 0.0

    # -- Final confidence --
    raw = base * freshness * diversity * support_ratio + actual_review_bonus
    final = max(0.0, min(1.0, raw))

    breakdown = ConfidenceBreakdown(
        base=round(base, 4),
        freshness=round(freshness, 4),
        diversity=round(diversity, 4),
        support_ratio=round(support_ratio, 4),
        review_bonus=round(actual_review_bonus, 4),
        final=round(final, 4),
    )

    # -- Determine assertion status --
    if total_evidence == 0:
        status = "active"  # no evidence yet, default
    elif contradiction_count > 0 and support_ratio < disputed_threshold:
        status = "disputed"
    else:
        status = "active"

    # -- Compute validity window from claims --
    valid_froms = [
        c.claim_valid_from for c, _ in support_claims
        if c.claim_valid_from is not None
    ]
    valid_tos = [
        c.claim_valid_to for c, _ in support_claims
        if c.claim_valid_to is not None
    ]
    valid_from = min(valid_froms) if valid_froms else None
    valid_to = max(valid_tos) if valid_tos else None

    assertion = ResolvedAssertion(
        assertion_id=assertion_id,
        subject_concept_id=subject_concept_id,
        predicate=predicate,
        object_concept_id=object_concept_id,
        confidence=breakdown.final,
        status=status,
        valid_from=valid_from,
        valid_to=valid_to,
        support_count=support_count,
        contradiction_count=contradiction_count,
        first_seen_at=first_seen,
        last_evidence_at=last_evidence,
        source_diversity=diversity_count,
        metadata={
            "breakdown": {
                "base": breakdown.base,
                "freshness": breakdown.freshness,
                "diversity": breakdown.diversity,
                "support_ratio": breakdown.support_ratio,
                "review_bonus": breakdown.review_bonus,
            },
        },
    )

    return assertion, breakdown
