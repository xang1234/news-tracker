"""Schemas for the claim retrieval substrate.

A retrieval result wraps the canonical :class:`EvidenceClaim` (so callers get
full lineage — source span, lane, confidence, concept resolutions, numeric
facts — for free) plus the similarity score that surfaced it. The filter is a
plain value object the repository translates into SQL predicates.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.claims.schemas import EvidenceClaim


@dataclass(frozen=True)
class ClaimRetrievalFilter:
    """Structured constraints applied alongside semantic ranking.

    ``status`` defaults to ``"active"`` (the only state worth grounding a
    briefing on); pass ``None`` explicitly to search across all states.
    ``theme_id`` filters to claims whose source *document* belongs to the
    theme.
    """

    lanes: list[str] | None = None
    status: str | None = "active"
    min_confidence: float | None = None
    subject_concept_id: str | None = None
    theme_id: str | None = None
    exclude_claim_ids: list[str] | None = None


@dataclass(frozen=True)
class RetrievedClaim:
    """A claim surfaced by retrieval, with its cosine similarity score."""

    claim: EvidenceClaim
    score: float
