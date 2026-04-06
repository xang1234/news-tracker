"""Stateless trigger functions for the claim review queue.

Each function inspects a resolver result or claim and returns a
ReviewTask if review is warranted, or None if the result is clean.
Triggers are pure functions — no I/O, no side effects. The caller
is responsible for persisting any returned tasks.

Trigger inventory:
    - check_low_confidence: Fuzzy match with close alternatives
    - check_llm_proposed: Any LLM-gated resolution
    - check_competing_predicates: Different predicates on same entities
    - check_high_impact_predicate: Risky predicates that need review
"""

from __future__ import annotations

import logging

from src.claims.resolver import ResolverResult, ResolverTier
from src.claims.review import ReviewTask, make_review_task_id
from src.claims.schemas import EvidenceClaim

logger = logging.getLogger(__name__)

# Predicates that carry significant business risk when wrong
HIGH_IMPACT_PREDICATES = frozenset(
    {
        "supplies_to",
        "customer_of",
        "competes_with",
        "acquires",
        "subsidiary_of",
        "parent_of",
    }
)

# Confidence below this threshold triggers review for fuzzy matches
LOW_CONFIDENCE_THRESHOLD = 0.65


def check_low_confidence(
    result: ResolverResult,
    claim: EvidenceClaim,
) -> ReviewTask | None:
    """Trigger review when a fuzzy match has close alternatives.

    A fuzzy resolution with alternatives suggests ambiguity that
    could resolve to the wrong entity. The task payload captures
    the alternatives so the reviewer can pick the correct one.
    """
    if result.tier != ResolverTier.FUZZY:
        return None
    if result.confidence >= LOW_CONFIDENCE_THRESHOLD:
        return None
    if not result.alternatives:
        return None

    concept_ids = [result.concept_id] if result.concept_id else []
    concept_ids.extend(a.concept_id for a in result.alternatives)

    task_id = make_review_task_id(
        "entity_review",
        claim_ids=[claim.claim_id],
        concept_ids=concept_ids,
    )
    return ReviewTask(
        task_id=task_id,
        task_type="entity_review",
        trigger_reason="close_alternatives",
        claim_ids=[claim.claim_id],
        concept_ids=concept_ids,
        priority=2,
        payload={
            "mention": result.mention,
            "matched_concept_id": result.concept_id,
            "matched_confidence": result.confidence,
            "alternatives": [
                {"concept_id": a.concept_id, "name": a.canonical_name}
                for a in result.alternatives
            ],
        },
        lineage={
            "source_claim_id": claim.claim_id,
            "run_id": claim.run_id,
            "lane": claim.lane,
        },
    )


def check_llm_proposed(
    result: ResolverResult,
    claim: EvidenceClaim,
) -> ReviewTask | None:
    """Trigger review for any LLM-proposed resolution.

    LLM-proposed resolutions are never auto-authoritative. Every
    one gets a review task so a human can confirm or reject.
    """
    if result.tier != ResolverTier.LLM_PROPOSED:
        return None

    concept_ids = []
    proposed_id = result.metadata.get("proposed_concept_id")
    if proposed_id:
        concept_ids.append(proposed_id)

    task_id = make_review_task_id(
        "entity_review",
        claim_ids=[claim.claim_id],
        concept_ids=concept_ids,
    )
    return ReviewTask(
        task_id=task_id,
        task_type="entity_review",
        trigger_reason="llm_proposed",
        claim_ids=[claim.claim_id],
        concept_ids=concept_ids,
        priority=1,
        payload={
            "mention": result.mention,
            "gate_metadata": result.metadata,
        },
        lineage={
            "source_claim_id": claim.claim_id,
            "run_id": claim.run_id,
            "lane": claim.lane,
        },
    )


def check_competing_predicates(
    claim_a: EvidenceClaim,
    claim_b: EvidenceClaim,
) -> ReviewTask | None:
    """Trigger review when two claims assert different predicates about the same entities.

    Flags for human review — not all predicate differences are true
    contradictions (e.g., "supplies_to" and "competes_with" can
    coexist). The reviewer decides whether both claims are valid or
    one should be retracted.

    Both claims must have resolved concepts to detect the overlap.
    """
    if not claim_a.subject_concept_id or not claim_b.subject_concept_id:
        return None
    if claim_a.subject_concept_id != claim_b.subject_concept_id:
        return None
    # Object concepts must also match (or both be None)
    if claim_a.object_concept_id != claim_b.object_concept_id:
        return None
    if claim_a.predicate == claim_b.predicate:
        return None

    concept_ids = [claim_a.subject_concept_id]
    if claim_a.object_concept_id:
        concept_ids.append(claim_a.object_concept_id)

    claim_ids = sorted([claim_a.claim_id, claim_b.claim_id])
    task_id = make_review_task_id(
        "claim_review",
        claim_ids=claim_ids,
        concept_ids=concept_ids,
    )
    return ReviewTask(
        task_id=task_id,
        task_type="claim_review",
        trigger_reason="contradiction",
        claim_ids=claim_ids,
        concept_ids=concept_ids,
        priority=1,
        payload={
            "claim_a": {
                "claim_id": claim_a.claim_id,
                "predicate": claim_a.predicate,
                "confidence": claim_a.confidence,
            },
            "claim_b": {
                "claim_id": claim_b.claim_id,
                "predicate": claim_b.predicate,
                "confidence": claim_b.confidence,
            },
        },
        lineage={
            "source_claim_ids": claim_ids,
            "run_ids": [
                r for r in [claim_a.run_id, claim_b.run_id] if r
            ],
        },
    )


def check_high_impact_predicate(
    claim: EvidenceClaim,
    *,
    confidence_threshold: float = 0.7,
) -> ReviewTask | None:
    """Trigger review for low-confidence claims with high-impact predicates.

    Supply chain, competition, and ownership claims carry significant
    business risk when wrong. If the extraction confidence is below
    the threshold, flag for review.
    """
    if claim.predicate not in HIGH_IMPACT_PREDICATES:
        return None
    if claim.confidence >= confidence_threshold:
        return None

    concept_ids = []
    if claim.subject_concept_id:
        concept_ids.append(claim.subject_concept_id)
    if claim.object_concept_id:
        concept_ids.append(claim.object_concept_id)

    task_id = make_review_task_id(
        "claim_review",
        claim_ids=[claim.claim_id],
        concept_ids=concept_ids,
    )
    return ReviewTask(
        task_id=task_id,
        task_type="claim_review",
        trigger_reason="high_impact_predicate",
        claim_ids=[claim.claim_id],
        concept_ids=concept_ids,
        priority=1,
        payload={
            "predicate": claim.predicate,
            "confidence": claim.confidence,
            "threshold": confidence_threshold,
            "subject": claim.subject_text,
            "object": claim.object_text,
        },
        lineage={
            "source_claim_id": claim.claim_id,
            "run_id": claim.run_id,
            "lane": claim.lane,
        },
    )


def build_merge_proposal(
    source_concept_id: str,
    target_concept_id: str,
    *,
    claim_ids: list[str] | None = None,
    confidence: float = 0.0,
    evidence: str = "",
    run_id: str | None = None,
) -> ReviewTask:
    """Build a merge proposal review task.

    Proposes that source_concept_id should be merged into
    target_concept_id (i.e., they represent the same entity).
    """
    cids = sorted([source_concept_id, target_concept_id])
    task_id = make_review_task_id(
        "merge_proposal",
        claim_ids=claim_ids or [],
        concept_ids=cids,
    )
    return ReviewTask(
        task_id=task_id,
        task_type="merge_proposal",
        trigger_reason="manual",
        claim_ids=claim_ids or [],
        concept_ids=cids,
        priority=2,
        payload={
            "source_concept_id": source_concept_id,
            "target_concept_id": target_concept_id,
            "confidence": confidence,
            "evidence": evidence,
        },
        lineage={"run_id": run_id} if run_id else {},
    )


def build_split_proposal(
    concept_id: str,
    proposed_names: list[str],
    *,
    claim_ids: list[str] | None = None,
    evidence: str = "",
    run_id: str | None = None,
) -> ReviewTask:
    """Build a split proposal review task.

    Proposes that concept_id actually represents multiple distinct
    entities that should be separated.
    """
    task_id = make_review_task_id(
        "split_proposal",
        claim_ids=claim_ids or [],
        concept_ids=[concept_id],
    )
    return ReviewTask(
        task_id=task_id,
        task_type="split_proposal",
        trigger_reason="manual",
        claim_ids=claim_ids or [],
        concept_ids=[concept_id],
        priority=2,
        payload={
            "concept_id": concept_id,
            "proposed_names": proposed_names,
            "evidence": evidence,
        },
        lineage={"run_id": run_id} if run_id else {},
    )
