"""Recompute assertions and derived edges after review decisions.

Closes the feedback loop: review outcomes and claim status changes
trigger recomputation of affected assertions and their downstream
derived edges. Every recompute is auditable — the result captures
what changed, why, and the before/after state.

Recompute flow:
    1. Identify affected assertions (from changed claim IDs)
    2. Re-aggregate each assertion from its current claim links
    3. Re-derive edges from updated assertions
    4. Return RecomputeResult with full audit trail
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.assertions.aggregation import (
    ConfidenceBreakdown,
    aggregate_assertion,
)
from src.assertions.edges import DerivedEdge, derive_edge
from src.assertions.schemas import (
    AssertionClaimLink,
    ResolvedAssertion,
)
from src.claims.schemas import EvidenceClaim

logger = logging.getLogger(__name__)


# -- Recompute result ------------------------------------------------------


@dataclass
class AssertionDelta:
    """Before/after snapshot of a single assertion recomputation.

    Attributes:
        assertion_id: The assertion that was recomputed.
        previous_confidence: Confidence before recompute.
        new_confidence: Confidence after recompute.
        previous_status: Status before recompute.
        new_status: Status after recompute.
        confidence_changed: Whether confidence moved.
        status_changed: Whether status transitioned.
        edge_before: Derived edge before recompute (None if new).
        edge_after: Derived edge after recompute (None if removed).
        breakdown: The new confidence breakdown.
    """

    assertion_id: str
    previous_confidence: float
    new_confidence: float
    previous_status: str
    new_status: str
    confidence_changed: bool = False
    status_changed: bool = False
    edge_before: DerivedEdge | None = None
    edge_after: DerivedEdge | None = None
    breakdown: ConfidenceBreakdown | None = None


@dataclass
class RecomputeResult:
    """Full audit trail from a recompute operation.

    Attributes:
        trigger: What caused the recompute (review resolution, claim update).
        trigger_detail: Specific IDs and context.
        affected_assertion_ids: Which assertions were recomputed.
        deltas: Before/after for each assertion.
        assertions_updated: Number of assertions whose state changed.
        edges_added: Number of new current edges created.
        edges_removed: Number of current edges that became history/removed.
        recomputed_at: When the recompute ran.
    """

    trigger: str
    trigger_detail: dict[str, Any] = field(default_factory=dict)
    affected_assertion_ids: list[str] = field(default_factory=list)
    deltas: list[AssertionDelta] = field(default_factory=list)
    assertions_updated: int = 0
    edges_added: int = 0
    edges_removed: int = 0
    recomputed_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    @property
    def had_changes(self) -> bool:
        """Whether anything actually changed."""
        return self.assertions_updated > 0


# -- Recompute functions ---------------------------------------------------

# Confidence delta below this is considered unchanged (floating point noise)
CONFIDENCE_EPSILON = 1e-6


def recompute_assertion(
    existing: ResolvedAssertion | None,
    claims: list[EvidenceClaim],
    links: list[AssertionClaimLink],
    *,
    subject_concept_id: str,
    predicate: str,
    object_concept_id: str | None = None,
    now: datetime | None = None,
) -> tuple[ResolvedAssertion, AssertionDelta]:
    """Recompute a single assertion from its current claims and links.

    Compares the result against the existing assertion (if any) to
    produce an AssertionDelta with before/after state.

    Args:
        existing: Current assertion state (None if new).
        claims: All active claims for this triple.
        links: All claim links for this assertion.
        subject_concept_id: Subject concept.
        predicate: Relationship predicate.
        object_concept_id: Object concept (None for unary).
        now: Current time for freshness calculation.

    Returns:
        (updated_assertion, delta) with the new state and what changed.
    """
    new_assertion, breakdown = aggregate_assertion(
        subject_concept_id,
        predicate,
        object_concept_id,
        claims,
        links,
        now=now,
    )

    prev_confidence = existing.confidence if existing else 0.0
    prev_status = existing.status if existing else "active"

    confidence_changed = abs(new_assertion.confidence - prev_confidence) > CONFIDENCE_EPSILON
    status_changed = new_assertion.status != prev_status

    edge_before = derive_edge(existing) if existing else None
    edge_after = derive_edge(new_assertion)

    delta = AssertionDelta(
        assertion_id=new_assertion.assertion_id,
        previous_confidence=prev_confidence,
        new_confidence=new_assertion.confidence,
        previous_status=prev_status,
        new_status=new_assertion.status,
        confidence_changed=confidence_changed,
        status_changed=status_changed,
        edge_before=edge_before,
        edge_after=edge_after,
        breakdown=breakdown,
    )

    return new_assertion, delta


def build_recompute_result(
    trigger: str,
    trigger_detail: dict[str, Any],
    deltas: list[AssertionDelta],
) -> RecomputeResult:
    """Build a RecomputeResult from a list of assertion deltas.

    Counts how many assertions actually changed, and how many
    edges were added or removed from current state.
    """
    assertions_updated = 0
    edges_added = 0
    edges_removed = 0
    for d in deltas:
        if d.confidence_changed or d.status_changed:
            assertions_updated += 1
        had_current_before = d.edge_before is not None and d.edge_before.is_current
        has_current_after = d.edge_after is not None and d.edge_after.is_current
        if has_current_after and not had_current_before:
            edges_added += 1
        elif had_current_before and not has_current_after:
            edges_removed += 1

    return RecomputeResult(
        trigger=trigger,
        trigger_detail=trigger_detail,
        affected_assertion_ids=[d.assertion_id for d in deltas],
        deltas=deltas,
        assertions_updated=assertions_updated,
        edges_added=edges_added,
        edges_removed=edges_removed,
    )


def find_affected_assertion_ids(
    claim_ids: list[str],
    links: list[AssertionClaimLink],
) -> list[str]:
    """Find assertion IDs affected by changes to specific claims.

    Looks up which assertions reference the given claim IDs
    through their claim links.
    """
    claim_id_set = set(claim_ids)
    affected = {
        lnk.assertion_id
        for lnk in links
        if lnk.claim_id in claim_id_set
    }
    return sorted(affected)
