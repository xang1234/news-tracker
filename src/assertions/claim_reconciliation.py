"""Shared helpers for claim reconciliation tiers.

Reconciliation (numeric contradiction, predicate-polarity contradiction, and
future corroboration/semantic tiers) all need the claim's subject grounded to
a concept ID before they can group comparable facts. Subject resolution is
therefore tier-agnostic and lives here rather than inside any single tier.
"""

from __future__ import annotations

import structlog

from src.claims.schemas import EvidenceClaim

logger = structlog.get_logger(__name__)


async def resolve_claim_subject(claim: EvidenceClaim, resolver) -> EvidenceClaim:
    """Ground a claim's subject to a concept ID, in place.

    The narrative extractor sets ``subject_text`` but not
    ``subject_concept_id``; reconciliation tiers key on the concept ID. This
    resolves the subject via the entity resolver cascade and mutates the claim.

    No-op when the subject is already grounded; leaves the concept ID ``None``
    when the subject cannot be resolved.
    """
    if claim.subject_concept_id is not None:
        return claim
    # Best-effort: a transient resolver failure must not drop the claim —
    # leave the subject unresolved and let persistence continue.
    try:
        result = await resolver.resolve(claim.subject_text, concept_type="issuer")
    except Exception as e:
        logger.warning(
            "Subject resolution failed; leaving claim unresolved",
            claim_id=claim.claim_id,
            subject_text=claim.subject_text,
            error=str(e),
        )
        return claim
    if result.resolved:
        claim.subject_concept_id = result.concept_id
    return claim
