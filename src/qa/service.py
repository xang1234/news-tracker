"""Cited Q&A service: grounded answer synthesis with a sufficiency signal.

Retrieves grounding claims for a free-text question (corpus-wide — no theme
filter), refuses when the grounding is too thin, otherwise asks the LLM to
synthesize a cited answer, validates the citations against the retrieved set,
and falls back to a templated extractive answer when the LLM is unavailable.

The breaker-guarded LLM round-trip is delegated to the scoring layer's shared
``JsonLLMClient`` (built from ``ScoringConfig``); a fake client is injected in
tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

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
    from src.scoring.config import ScoringConfig
    from src.scoring.json_llm import JsonLLMClient


class CitedQAService:
    """Answer free-text questions with citations into the structured claim layer."""

    def __init__(
        self,
        *,
        retrieval_service: ClaimRetrievalService,
        scoring_config: ScoringConfig,
        config: QAConfig | None = None,
        llm: JsonLLMClient | None = None,
    ) -> None:
        from src.scoring.json_llm import JsonLLMClient

        self._retrieval = retrieval_service
        self._config = config or QAConfig()
        self._llm = llm or JsonLLMClient(scoring_config, name="cited_qa")

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
        if self._llm.has_api_key:
            valid_ids = {c.claim_id for c in claims}
            prompt = build_qa_prompt(
                question, [(c.claim_id, claim_embedding_text(c)) for c in claims]
            )
            raw = await self._llm.complete_json(prompt)
            if raw is not None:
                parsed = parse_qa_response(raw, valid_ids)
                if parsed:
                    segments = parsed[: self._config.max_segments]
                    model = self._llm.model

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
