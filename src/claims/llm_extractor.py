"""LLM claim extractor — the second extraction pass (epic ``7th``).

Reads a document's raw text and emits ``EvidenceClaim`` objects with
``extraction_method="llm"``, capturing implicit/multi-clause relationships the
regex ``narrative_extractor`` misses. The breaker-guarded round-trip is the
shared ``JsonLLMClient`` (see scoring's Grounded-LLM Substrate); prompt and
parsing are the pure functions in ``llm_prompt``. Degrades to ``[]`` whenever
the model is unavailable (no key / open breaker / unparseable), so a caller can
always fall back to the rule pass.

``7th.2`` is the pass itself; ``7th.3`` adds the high-value ``is_high_value``
gate (to bound cost), the ``claims.merge.merge_claims`` dedup-merge with the
rule pass, and the ``llm_claim_extraction_enabled`` wiring in
``processing_service``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.claims.llm_prompt import build_extraction_prompt, parse_extraction_response
from src.claims.schemas import EvidenceClaim

if TYPE_CHECKING:
    from src.scoring.config import ScoringConfig
    from src.scoring.json_llm import JsonLLMClient


class LLMExtractionConfig(BaseSettings):
    """Tuning for the LLM extraction pass (overridable via ``LLM_EXTRACTION_*``)."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_EXTRACTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    max_claims: int = Field(default=12, ge=1, description="Max claims requested per document")
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Drop LLM claims below this confidence"
    )
    # High-value gate — bounds cost by only running the paid pass on docs that
    # clear an authority bar (primary) or an opt-in engagement bar (secondary).
    min_authority: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Authority score required to run the LLM pass"
    )
    min_engagement: float = Field(
        default=0.0,
        ge=0.0,
        description="Engagement score that also qualifies a doc (0 disables this lever)",
    )


def is_high_value(doc: Any, config: LLMExtractionConfig) -> bool:
    """Whether a document clears the bar for the paid LLM extraction pass.

    Authority (normalized 0-1) is the primary signal; engagement is an opt-in
    secondary lever, active only when ``min_engagement > 0``. ``doc`` is a
    ``NormalizedDocument`` (typed ``Any`` only to avoid an ingestion import).
    """
    if (doc.authority_score or 0.0) >= config.min_authority:
        return True
    if config.min_engagement > 0:
        return doc.engagement.engagement_score >= config.min_engagement
    return False


class LLMClaimExtractor:
    """Extract evidence claims from raw document text via a single LLM call."""

    def __init__(
        self,
        *,
        scoring_config: ScoringConfig,
        config: LLMExtractionConfig | None = None,
        llm: JsonLLMClient | None = None,
    ) -> None:
        from src.scoring.json_llm import JsonLLMClient

        self._config = config or LLMExtractionConfig()
        self._llm = llm or JsonLLMClient(scoring_config, name="claim_extraction")

    @property
    def config(self) -> LLMExtractionConfig:
        """The extraction tuning (read by the high-value gate before invoking)."""
        return self._config

    async def extract(
        self,
        doc_id: str,
        content: str,
        *,
        run_id: str | None = None,
        published_at: Any | None = None,
    ) -> list[EvidenceClaim]:
        """Extract LLM claims for one document; ``[]`` if the model is unavailable."""
        if not content.strip() or not self._llm.has_api_key:
            return []
        prompt = build_extraction_prompt(content, max_claims=self._config.max_claims)
        # parse_extraction_response is None-safe, so the raw payload feeds it directly.
        return parse_extraction_response(
            await self._llm.complete_json(prompt),
            doc_id=doc_id,
            run_id=run_id,
            published_at=published_at,
            min_confidence=self._config.min_confidence,
        )
