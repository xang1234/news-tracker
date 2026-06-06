"""Theme briefing generator.

Retrieves the top grounding claims for a theme, asks the LLM to write a brief
that cites them, validates the citations against the retrieved set, and wraps
the result. Falls back to a deterministic templated brief whenever the LLM is
unavailable or produces nothing groundable — so the endpoint always returns a
grounded, fully-cited briefing.

The breaker-guarded LLM round-trip is delegated to the scoring layer's shared
``JsonLLMClient`` (built from ``ScoringConfig``); a fake client is injected in
tests to exercise orchestration without a live model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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
    from src.scoring.config import ScoringConfig
    from src.scoring.json_llm import JsonLLMClient
    from src.themes.repository import ThemeRepository


class ThemeBriefingService:
    """Generate grounded, cited theme briefings with templated fallback."""

    def __init__(
        self,
        *,
        theme_repository: ThemeRepository,
        retrieval_service: ClaimRetrievalService,
        scoring_config: ScoringConfig,
        config: BriefingConfig | None = None,
        llm: JsonLLMClient | None = None,
    ) -> None:
        from src.scoring.json_llm import JsonLLMClient

        self._themes = theme_repository
        self._retrieval = retrieval_service
        self._config = config or BriefingConfig()
        self._llm = llm or JsonLLMClient(scoring_config, name="theme_briefing")

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
        if claims and self._llm.has_api_key:
            valid_ids = {c.claim_id for c in claims}
            prompt = build_briefing_prompt(
                theme.name, [(c.claim_id, claim_embedding_text(c)) for c in claims]
            )
            # parse_briefing_response is None-safe, so the raw payload feeds it directly.
            parsed = parse_briefing_response(await self._llm.complete_json(prompt), valid_ids)
            if parsed:
                clauses = parsed[: self._config.max_clauses]
                model = self._llm.model

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
