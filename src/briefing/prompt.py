"""Pure prompt construction + grounded-citation parsing for theme briefings.

``parse_briefing_response`` is the grounding gate: it accepts the LLM's JSON,
strips any cited claim id the model invented (not in the retrieved set), and
drops any clause left without a real citation — so the output can never
contain an uncited or hallucinated-cited assertion.
"""

from __future__ import annotations

from typing import Any

from src.briefing.schemas import BriefingClause
from src.retrieval.citation_gate import parse_cited_entries

_BRIEFING_PROMPT = """\
You are writing a short factual brief about the theme "{theme_name}" for an \
analyst. Use ONLY the evidence claims listed below. Write 2-5 concise \
sentences. Every sentence MUST be grounded in one or more of these claims and \
MUST cite the claim id(s) it draws from. Do not state anything not supported \
by a listed claim. Do not invent claim ids.

Reply with ONLY a JSON object of this shape:
{{"clauses": [{{"text": "<sentence>", "claim_ids": ["<id>", ...]}}, ...]}}

Evidence claims:
{claims_block}
"""


def build_briefing_prompt(theme_name: str, claims: list[tuple[str, str]]) -> str:
    """Build the briefing prompt from a theme name and ``(claim_id, text)`` pairs."""
    claims_block = "\n".join(f"- [{claim_id}] {text}" for claim_id, text in claims)
    return _BRIEFING_PROMPT.format(theme_name=theme_name, claims_block=claims_block)


def parse_briefing_response(payload: Any, valid_claim_ids: set[str]) -> list[BriefingClause]:
    """Parse + ground an LLM briefing response.

    Returns only clauses with non-empty text and at least one citation that
    exists in ``valid_claim_ids``; invented ids are stripped and now-uncited
    clauses dropped. Any malformed input yields an empty list (caller falls
    back to the template).
    """
    return parse_cited_entries(
        payload,
        valid_claim_ids,
        key="clauses",
        factory=lambda text, claim_ids: BriefingClause(text=text, claim_ids=claim_ids),
    )
