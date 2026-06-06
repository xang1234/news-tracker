"""Semantic contradiction judge.

Decides whether two claims semantically contradict, for the residual pairs the
deterministic tiers (numeric, predicate-polarity) can't settle — e.g. "TSMC
will raise prices" vs "TSMC will cut prices" (no antonym predicate, no numeric
fact). Wraps a single LLM call behind a circuit breaker.

Prompt construction and verdict parsing are pure functions; the LLM round-trip
lives in ``SemanticContradictionJudge.judge`` and is exercised via injected
fakes in tests, never a live API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only: keeps the scoring service package off the `assertions` import
    # path (the judge only needs scoring when it is actually constructed).
    from src.scoring.config import ScoringConfig
    from src.scoring.json_llm import JsonLLMClient

VALID_RELATIONS = frozenset({"contradicts", "agrees", "unrelated"})

_JUDGE_PROMPT = """\
You are comparing two factual claims about the same entity. Decide their \
relationship and reply with ONLY a JSON object: \
{{"relation": "contradicts"|"agrees"|"unrelated", "confidence": 0.0-1.0}}.

- "contradicts": both cannot be true at the same time.
- "agrees": they assert the same thing.
- "unrelated": they are about different facts.

Claim A: {text_a}
Claim B: {text_b}
"""


@dataclass(frozen=True)
class ContradictionVerdict:
    """An LLM verdict on whether two claims contradict.

    Attributes:
        relation: one of ``contradicts`` / ``agrees`` / ``unrelated``.
        confidence: model confidence in the relation, 0-1.
    """

    relation: str
    confidence: float


def build_judge_prompt(text_a: str, text_b: str) -> str:
    """Build the contradiction-judgement prompt for two claim texts."""
    return _JUDGE_PROMPT.format(text_a=text_a, text_b=text_b)


def parse_verdict(payload: Any) -> ContradictionVerdict | None:
    """Parse an LLM response into a verdict, or None if malformed.

    Accepts a dict (already-parsed JSON) or a JSON string. Returns None for an
    unknown relation, missing/ non-numeric confidence, or unparseable input.
    Confidence is clamped to [0, 1].
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None

    relation = payload.get("relation")
    confidence = payload.get("confidence")
    if relation not in VALID_RELATIONS:
        return None
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return None

    return ContradictionVerdict(
        relation=relation,
        confidence=max(0.0, min(1.0, float(confidence))),
    )


class SemanticContradictionJudge:
    """LLM-backed contradiction judge over the shared JSON-LLM client.

    The contradiction prompt and verdict parsing are domain-specific and live
    here; the breaker-guarded round-trip is delegated to the scoring layer's
    ``JsonLLMClient``. A no-key client short-circuits to ``None`` without
    tripping its breaker, so the tier no-ops cleanly when unconfigured.
    """

    def __init__(self, config: ScoringConfig, *, llm: JsonLLMClient | None = None) -> None:
        # Lazy import so importing `assertions` doesn't pull in the scoring
        # service package; the judge is only built when the tier is enabled.
        from src.scoring.json_llm import JsonLLMClient

        self._llm = llm or JsonLLMClient(config, name="semantic_judge")

    async def judge(self, text_a: str, text_b: str) -> ContradictionVerdict | None:
        """Judge two claim texts; returns a verdict or None on failure/open circuit."""
        # parse_verdict is None-safe (a None payload parses to None), so no guard.
        return parse_verdict(await self._llm.complete_json(build_judge_prompt(text_a, text_b)))
