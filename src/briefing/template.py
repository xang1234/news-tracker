"""Deterministic templated-briefing fallback.

When the LLM is unavailable (circuit breaker open, no API key, or it returns
nothing groundable), each retrieved claim becomes one self-cited clause. The
clause text reuses the canonical claim→text composer from the retrieval
substrate, so the fallback stays consistent with what was embedded/searched.
"""

from __future__ import annotations

from src.briefing.schemas import BriefingClause
from src.claims.schemas import EvidenceClaim
from src.retrieval.text import claim_embedding_text


def templated_clauses(claims: list[EvidenceClaim], *, limit: int) -> list[BriefingClause]:
    """One self-cited clause per claim, capped at ``limit``."""
    return [
        BriefingClause(text=claim_embedding_text(claim), claim_ids=[claim.claim_id])
        for claim in claims[:limit]
    ]
