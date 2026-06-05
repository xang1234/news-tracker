"""Pure claim→citation mapping.

Projects an :class:`EvidenceClaim` down to the lineage the UI needs to make a
cited ``claim_id`` clickable: the resolved triple plus the source pointer
(type/id/character span) for jumping to the evidence document and span.
"""

from __future__ import annotations

from src.briefing.schemas import ClaimCitation
from src.claims.schemas import EvidenceClaim


def citation_from_claim(claim: EvidenceClaim) -> ClaimCitation:
    """Build a :class:`ClaimCitation` from a claim's lineage."""
    return ClaimCitation(
        claim_id=claim.claim_id,
        subject_text=claim.subject_text,
        predicate=claim.predicate,
        object_text=claim.object_text,
        source_type=claim.source_type,
        source_id=claim.source_id,
        source_span_start=claim.source_span_start,
        source_span_end=claim.source_span_end,
        snippet=claim.source_text,
    )
