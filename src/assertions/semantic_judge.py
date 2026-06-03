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
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

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
    """LLM-backed contradiction judge, gated by a circuit breaker.

    Reuses the generic circuit breaker from the scoring layer (the only
    genuinely-shared piece); the prompt and parsing are contradiction-specific
    and live here, decoupled from compellingness scoring.
    """

    def __init__(self, config: Any, *, breaker: Any = None) -> None:
        from src.scoring.circuit_breaker import GenericCircuitBreaker

        self._config = config
        self._client: Any = None
        self._breaker = breaker or GenericCircuitBreaker(
            failure_threshold=getattr(config, "circuit_failure_threshold", 5),
            recovery_timeout=getattr(config, "circuit_recovery_timeout", 60.0),
            name="semantic_judge",
        )

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            api_key = self._config.openai_api_key
            self._client = openai.AsyncOpenAI(
                api_key=api_key.get_secret_value() if api_key else None,
                timeout=self._config.llm_timeout,
            )
        return self._client

    async def judge(self, text_a: str, text_b: str) -> ContradictionVerdict | None:
        """Judge two claim texts; returns a verdict or None on failure/open circuit."""
        prompt = build_judge_prompt(text_a, text_b)

        async def _call() -> ContradictionVerdict | None:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return parse_verdict(response.choices[0].message.content)

        try:
            return await self._breaker.call(_call)
        except Exception as e:  # breaker open, API error, etc. — degrade gracefully
            logger.warning("Semantic judge call failed", error=str(e))
            return None
