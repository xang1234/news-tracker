"""Theme briefing generator: grounded, cited natural-language briefs over the
structured claim layer (RAG epic q7g.2).

``ThemeBriefingService.generate`` retrieves the top grounding claims for a
theme, has the LLM write a brief citing them (citations validated against the
retrieved set), and falls back to a deterministic templated brief when the LLM
is unavailable — so every clause is always grounded and cited.
"""

from __future__ import annotations

from src.briefing.citations import citation_from_claim
from src.briefing.config import BriefingConfig
from src.briefing.generator import ThemeBriefingService
from src.briefing.schemas import BriefingClause, ClaimCitation, ThemeBriefing

__all__ = [
    "BriefingClause",
    "BriefingConfig",
    "ClaimCitation",
    "ThemeBriefing",
    "ThemeBriefingService",
    "citation_from_claim",
]
