"""Schemas for cited Q&A over the structured claim layer.

An answer is a list of segments, each grounded in (and citing) one or more
evidence claims — the same "no uncited assertion" discipline as briefings.
``confidence`` carries the grounding-sufficiency signal: ``insufficient``
means retrieval found too little/weak evidence to answer (an explicit
refusal), distinct from a low-confidence templated fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

#: Grounding-sufficiency levels for an answer.
CONFIDENCE_HIGH = "high"  # LLM-synthesized over sufficient grounding
CONFIDENCE_LOW = "low"  # sufficient grounding, templated fallback (LLM unavailable)
CONFIDENCE_INSUFFICIENT = "insufficient"  # grounding too thin/weak — refusal


@dataclass(frozen=True)
class AnswerSegment:
    """One sentence of an answer plus the claim ids it is grounded in."""

    text: str
    claim_ids: list[str]


@dataclass(frozen=True)
class CitedAnswer:
    """A cited answer to a free-text question, with a sufficiency signal."""

    question: str
    segments: list[AnswerSegment]
    confidence: str
    claim_count: int
    generated_by: str
    model: str | None = None
    generated_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_grounded(self) -> bool:
        """True when the answer rests on at least one cited segment."""
        return self.confidence != CONFIDENCE_INSUFFICIENT and bool(self.segments)
