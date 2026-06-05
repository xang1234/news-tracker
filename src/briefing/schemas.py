"""Schemas for theme briefings.

A briefing is a list of clauses, each grounded in (and citing) one or more
evidence claims. The clause is the unit that enforces the "no uncited
assertion" invariant: a clause with no valid ``claim_ids`` never reaches the
output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class BriefingClause:
    """One sentence of a briefing plus the claim ids it is grounded in."""

    text: str
    claim_ids: list[str]


@dataclass(frozen=True)
class ThemeBriefing:
    """A grounded, cited natural-language brief for a theme.

    ``generated_by`` is ``"llm"`` when the model produced the prose and
    ``"template"`` when the deterministic fallback did (breaker open, no API
    key, or the LLM returned nothing groundable).
    """

    theme_id: str
    clauses: list[BriefingClause]
    generated_by: str
    claim_count: int
    model: str | None = None
    generated_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def cited_claim_ids(self) -> list[str]:
        """Distinct claim ids cited across all clauses, in first-seen order."""
        seen: dict[str, None] = {}
        for clause in self.clauses:
            for cid in clause.claim_ids:
                seen.setdefault(cid, None)
        return list(seen)
