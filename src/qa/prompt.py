"""Pure prompt construction + grounded-citation parsing for cited Q&A.

``parse_qa_response`` is the grounding gate (mirrors the briefing gate, Q&A
shape): it strips any cited claim id the model invented and drops any segment
left without a real citation, so an answer can never contain an uncited or
hallucinated-cited assertion.
"""

from __future__ import annotations

from typing import Any

from src.qa.schemas import AnswerSegment
from src.retrieval.citation_gate import parse_cited_entries

_QA_PROMPT = """\
You are answering an analyst's question using ONLY the evidence claims listed \
below. Answer in 1-4 concise sentences. Every sentence MUST be grounded in one \
or more of these claims and MUST cite the claim id(s) it draws from. If the \
claims do not answer the question, say so in a single sentence citing the \
closest claim. Do not state anything not supported by a listed claim, and do \
not invent claim ids.

Reply with ONLY a JSON object of this shape:
{{"segments": [{{"text": "<sentence>", "claim_ids": ["<id>", ...]}}, ...]}}

Question: {question}

Evidence claims:
{claims_block}
"""


def build_qa_prompt(question: str, claims: list[tuple[str, str]]) -> str:
    """Build the Q&A prompt from a question and ``(claim_id, text)`` pairs."""
    claims_block = "\n".join(f"- [{claim_id}] {text}" for claim_id, text in claims)
    return _QA_PROMPT.format(question=question, claims_block=claims_block)


def parse_qa_response(payload: Any, valid_claim_ids: set[str]) -> list[AnswerSegment]:
    """Parse + ground an LLM answer response.

    Returns only segments with non-empty text and at least one citation that
    exists in ``valid_claim_ids``; invented ids are stripped and now-uncited
    segments dropped. Malformed input yields an empty list.
    """
    return parse_cited_entries(
        payload,
        valid_claim_ids,
        key="segments",
        factory=lambda text, claim_ids: AnswerSegment(text=text, claim_ids=claim_ids),
    )
