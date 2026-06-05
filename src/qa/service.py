"""Cited Q&A service: grounded answer synthesis with a sufficiency signal.

Retrieves grounding claims for a free-text question (corpus-wide — no theme
filter), refuses when the grounding is too thin, otherwise asks the LLM to
synthesize a cited answer, validates the citations against the retrieved set,
and falls back to a templated extractive answer when the LLM is unavailable.

LLM wiring mirrors ``briefing.generator`` / ``assertions.semantic_judge`` (lazy
OpenAI client behind the scoring layer's ``GenericCircuitBreaker``, reusing
``ScoringConfig``). The ``_call_llm`` / ``_has_api_key`` seams are overridden
in tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.qa.config import QAConfig
from src.qa.prompt import build_qa_prompt, parse_qa_response
from src.qa.schemas import (
    CONFIDENCE_HIGH,
    CONFIDENCE_INSUFFICIENT,
    CONFIDENCE_LOW,
    AnswerSegment,
    CitedAnswer,
)
from src.retrieval.schemas import ClaimRetrievalFilter
from src.retrieval.service import ClaimRetrievalService
from src.retrieval.text import claim_embedding_text

if TYPE_CHECKING:
    from src.scoring.circuit_breaker import GenericCircuitBreaker
    from src.scoring.config import ScoringConfig

logger = structlog.get_logger(__name__)


class CitedQAService:
    """Answer free-text questions with citations into the structured claim layer."""

    def __init__(
        self,
        *,
        retrieval_service: ClaimRetrievalService,
        scoring_config: ScoringConfig,
        config: QAConfig | None = None,
        breaker: GenericCircuitBreaker | None = None,
    ) -> None:
        from src.scoring.circuit_breaker import GenericCircuitBreaker

        self._retrieval = retrieval_service
        self._scoring_config = scoring_config
        self._config = config or QAConfig()
        self._client: Any = None
        self._breaker = breaker or GenericCircuitBreaker(
            failure_threshold=scoring_config.circuit_failure_threshold,
            recovery_timeout=scoring_config.circuit_recovery_timeout,
            name="cited_qa",
        )

    async def answer(self, question: str) -> CitedAnswer:
        """Answer ``question`` with cited segments, or refuse if grounding is thin."""
        retrieved = await self._retrieval.retrieve(
            question,
            limit=self._config.max_claims,
            filters=ClaimRetrievalFilter(min_confidence=self._config.min_confidence),
        )
        claims = [r.claim for r in retrieved]
        top_score = retrieved[0].score if retrieved else 0.0

        # Sufficiency gate: thin or weak grounding → explicit refusal.
        if not claims or top_score < self._config.min_grounding_score:
            return self._answer(question, [], CONFIDENCE_INSUFFICIENT, len(claims), "template")

        segments = None
        model = None
        if self._has_api_key():
            valid_ids = {c.claim_id for c in claims}
            prompt = build_qa_prompt(
                question, [(c.claim_id, claim_embedding_text(c)) for c in claims]
            )
            raw = await self._call_llm(prompt)
            if raw is not None:
                parsed = parse_qa_response(raw, valid_ids)
                if parsed:
                    segments = parsed[: self._config.max_segments]
                    model = self._scoring_config.openai_model

        if segments is None:
            # Grounding was sufficient, but the LLM was unavailable → templated
            # extractive answer at low confidence (still fully cited).
            segments = [
                AnswerSegment(text=claim_embedding_text(c), claim_ids=[c.claim_id])
                for c in claims[: self._config.max_segments]
            ]
            return self._answer(question, segments, CONFIDENCE_LOW, len(claims), "template")

        return self._answer(question, segments, CONFIDENCE_HIGH, len(claims), "llm", model)

    @staticmethod
    def _answer(
        question: str,
        segments: list[AnswerSegment],
        confidence: str,
        claim_count: int,
        generated_by: str,
        model: str | None = None,
    ) -> CitedAnswer:
        return CitedAnswer(
            question=question,
            segments=segments,
            confidence=confidence,
            claim_count=claim_count,
            generated_by=generated_by,
            model=model,
            generated_at=datetime.now(UTC),
        )

    def _has_api_key(self) -> bool:
        return bool(self._scoring_config.openai_api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            api_key = self._scoring_config.openai_api_key
            self._client = openai.AsyncOpenAI(
                api_key=api_key.get_secret_value()
                if hasattr(api_key, "get_secret_value")
                else api_key,
                timeout=self._scoring_config.llm_timeout,
            )
        return self._client

    async def _call_llm(self, prompt: str) -> Any:
        """Run the Q&A completion behind the breaker; None on failure/open."""

        async def _call() -> Any:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._scoring_config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content) if content else None

        try:
            return await self._breaker.call(_call)
        except Exception as e:  # breaker open, API error, bad JSON — degrade
            logger.warning("Cited Q&A LLM call failed", error=str(e))
            return None
