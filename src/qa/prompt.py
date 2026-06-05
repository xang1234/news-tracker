"""Pure prompt construction + grounded-citation parsing for cited Q&A.

``parse_qa_response`` is the grounding gate (mirrors the briefing gate, Q&A
shape): it strips any cited claim id the model invented and drops any segment
left without a real citation, so an answer can never contain an uncited or
hallucinated-cited assertion.
"""

from __future__ import annotations

import json
from typing import Any

from src.qa.schemas import AnswerSegment

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


def _dedupe(ids: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for i in ids:
        seen.setdefault(i, None)
    return list(seen)


def parse_qa_response(payload: Any, valid_claim_ids: set[str]) -> list[AnswerSegment]:
    """Parse + ground an LLM answer response.

    Returns only segments with non-empty text and at least one citation that
    exists in ``valid_claim_ids``; invented ids are stripped and now-uncited
    segments dropped. Malformed input yields an empty list.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(payload, dict):
        return []
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return []

    segments: list[AnswerSegment] = []
    for entry in raw_segments:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        ids = entry.get("claim_ids")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(ids, list):
            continue
        grounded = _dedupe([i for i in ids if isinstance(i, str) and i in valid_claim_ids])
        if not grounded:
            continue
        segments.append(AnswerSegment(text=text.strip(), claim_ids=grounded))
    return segments
