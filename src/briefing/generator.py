"""Theme briefing generator.

Retrieves the top grounding claims for a theme, asks the LLM to write a brief
that cites them, validates the citations against the retrieved set, and wraps
the result. Falls back to a deterministic templated brief whenever the LLM is
unavailable or produces nothing groundable — so the endpoint always returns a
grounded, fully-cited briefing.

LLM wiring mirrors ``assertions.semantic_judge``: a lazily-built OpenAI client
guarded by the scoring layer's ``GenericCircuitBreaker``, reusing
``ScoringConfig`` for credentials/model/timeout/breaker thresholds. The
``_call_llm`` and ``_has_api_key`` seams are overridden in tests to exercise
orchestration without a live model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from src.briefing.citations import citation_from_claim
from src.briefing.config import BriefingConfig
from src.briefing.prompt import build_briefing_prompt, parse_briefing_response
from src.briefing.schemas import ThemeBriefing
from src.briefing.template import templated_clauses
from src.retrieval.schemas import ClaimRetrievalFilter
from src.retrieval.service import ClaimRetrievalService
from src.retrieval.text import claim_embedding_text

if TYPE_CHECKING:
    # Type-only: keeps the scoring service package off the briefing import path
    # (scoring is only needed when a briefing is actually generated with a key).
    from src.scoring.circuit_breaker import GenericCircuitBreaker
    from src.scoring.config import ScoringConfig
    from src.themes.repository import ThemeRepository

logger = structlog.get_logger(__name__)


class ThemeBriefingService:
    """Generate grounded, cited theme briefings with templated fallback."""

    def __init__(
        self,
        *,
        theme_repository: ThemeRepository,
        retrieval_service: ClaimRetrievalService,
        scoring_config: ScoringConfig,
        config: BriefingConfig | None = None,
        breaker: GenericCircuitBreaker | None = None,
    ) -> None:
        from src.scoring.circuit_breaker import GenericCircuitBreaker

        self._themes = theme_repository
        self._retrieval = retrieval_service
        self._scoring_config = scoring_config
        self._config = config or BriefingConfig()
        self._client: Any = None
        self._breaker = breaker or GenericCircuitBreaker(
            failure_threshold=scoring_config.circuit_failure_threshold,
            recovery_timeout=scoring_config.circuit_recovery_timeout,
            name="theme_briefing",
        )

    async def generate(self, theme_id: str) -> ThemeBriefing | None:
        """Generate a briefing for ``theme_id``; None if the theme doesn't exist."""
        theme = await self._themes.get_by_id(theme_id)
        if theme is None:
            return None

        retrieved = await self._retrieval.retrieve(
            self._theme_query(theme),
            limit=self._config.max_claims,
            filters=ClaimRetrievalFilter(
                theme_id=theme_id, min_confidence=self._config.min_confidence
            ),
        )
        claims = [r.claim for r in retrieved]

        clauses = None
        model = None
        if claims and self._has_api_key():
            valid_ids = {c.claim_id for c in claims}
            prompt = build_briefing_prompt(
                theme.name, [(c.claim_id, claim_embedding_text(c)) for c in claims]
            )
            raw = await self._call_llm(prompt)
            if raw is not None:
                parsed = parse_briefing_response(raw, valid_ids)
                if parsed:
                    clauses = parsed[: self._config.max_clauses]
                    model = self._scoring_config.openai_model

        if clauses is None:
            clauses = templated_clauses(claims, limit=self._config.max_clauses)
            generated_by = "template"
        else:
            generated_by = "llm"

        return ThemeBriefing(
            theme_id=theme_id,
            clauses=clauses,
            generated_by=generated_by,
            claim_count=len(claims),
            citations=[citation_from_claim(c) for c in claims],
            model=model,
            generated_at=datetime.now(UTC),
        )

    @staticmethod
    def _theme_query(theme: Any) -> str:
        keywords = " ".join(getattr(theme, "top_keywords", []) or [])
        return f"{theme.name}: {keywords}".strip().rstrip(":").strip()

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
        """Run the briefing completion behind the breaker; None on failure/open."""

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
            logger.warning("Theme briefing LLM call failed", error=str(e))
            return None
