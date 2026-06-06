"""LLM claim extractor — the second extraction pass (epic ``7th``).

Reads a document's raw text and emits ``EvidenceClaim`` objects with
``extraction_method="llm"``, capturing implicit/multi-clause relationships the
regex ``narrative_extractor`` misses. The breaker-guarded round-trip is the
shared ``JsonLLMClient`` (see scoring's Grounded-LLM Substrate); prompt and
parsing are the pure functions in ``llm_prompt``. Degrades to ``[]`` whenever
the model is unavailable (no key / open breaker / unparseable), so a caller can
always fall back to the rule pass.

This bead (``7th.2``) is the pass itself; merging its output with the rule
pass and gating it behind a feature flag in ``processing_service`` is ``7th.3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.claims.llm_prompt import build_extraction_prompt, parse_extraction_response
from src.claims.schemas import EvidenceClaim

if TYPE_CHECKING:
    from src.scoring.config import ScoringConfig
    from src.scoring.json_llm import JsonLLMClient


@dataclass(frozen=True)
class LLMExtractionConfig:
    """Tuning for the LLM extraction pass (injectable; no env surface yet)."""

    max_claims: int = 12
    min_confidence: float = 0.5


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
