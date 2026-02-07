"""LLM API abstraction for compellingness scoring.

Provides a unified interface to OpenAI (GPT-4o-mini, Tier 2) and Anthropic
(Claude, Tier 3) with lazy SDK initialization, per-provider circuit breakers,
and response validation against Pydantic schemas.

SDK imports are deferred to method calls (lazy loading) to avoid import-time
failures when API keys are not configured.
"""

import json
import logging
from typing import Any

from src.scoring.circuit_breaker import GenericCircuitBreaker
from src.scoring.config import ScoringConfig
from src.scoring.prompts import SCORING_PROMPT, SYSTEM_PROMPT, VALIDATION_PROMPT
from src.scoring.schemas import (
    CompellingnessScore,
    DimensionScores,
    EvidenceQuote,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client for scoring with OpenAI and Anthropic.

    Features:
    - Lazy SDK initialization (import on first use)
    - Per-provider circuit breakers
    - Response validation against Pydantic schema
    - Graceful fallback to None on parse failure

    Args:
        config: Scoring configuration with API keys and model names.
    """

    def __init__(self, config: ScoringConfig) -> None:
        self._config = config
        self._openai_client: Any = None
        self._anthropic_client: Any = None
        self._openai_breaker = GenericCircuitBreaker(
            failure_threshold=config.circuit_failure_threshold,
            recovery_timeout=config.circuit_recovery_timeout,
            name="openai",
        )
        self._anthropic_breaker = GenericCircuitBreaker(
            failure_threshold=config.circuit_failure_threshold,
            recovery_timeout=config.circuit_recovery_timeout,
            name="anthropic",
        )

    def _get_openai_client(self) -> Any:
        """Lazy-initialize OpenAI async client."""
        if self._openai_client is None:
            import openai

            api_key = self._config.openai_api_key
            key_str = api_key.get_secret_value() if api_key else None
            self._openai_client = openai.AsyncOpenAI(
                api_key=key_str,
                timeout=self._config.llm_timeout,
            )
        return self._openai_client

    def _get_anthropic_client(self) -> Any:
        """Lazy-initialize Anthropic async client."""
        if self._anthropic_client is None:
            import anthropic

            api_key = self._config.anthropic_api_key
            key_str = api_key.get_secret_value() if api_key else None
            self._anthropic_client = anthropic.AsyncAnthropic(
                api_key=key_str,
                timeout=self._config.llm_timeout,
            )
        return self._anthropic_client

    @property
    def openai_breaker(self) -> GenericCircuitBreaker:
        """Access OpenAI circuit breaker state."""
        return self._openai_breaker

    @property
    def anthropic_breaker(self) -> GenericCircuitBreaker:
        """Access Anthropic circuit breaker state."""
        return self._anthropic_breaker

    async def score_with_openai(
        self,
        thesis_text: str,
        context: dict[str, Any],
    ) -> CompellingnessScore | None:
        """Score a theme using GPT-4o-mini (Tier 2).

        Uses JSON mode for structured output. Validates response
        against the CompellingnessScore schema.

        Args:
            thesis_text: Aggregated theme content.
            context: Dict with tickers, keywords, lifecycle_stage, document_count.

        Returns:
            CompellingnessScore or None if parsing fails.
        """
        prompt = SCORING_PROMPT.format(
            thesis_text=thesis_text,
            tickers=", ".join(context.get("tickers", [])),
            keywords=", ".join(context.get("keywords", [])),
            lifecycle_stage=context.get("lifecycle_stage", "unknown"),
            document_count=context.get("document_count", 0),
        )

        async def _call() -> CompellingnessScore | None:
            client = self._get_openai_client()
            response = await client.chat.completions.create(
                model=self._config.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return self._parse_score_response(raw, tier="gpt", model=self._config.openai_model)

        return await self._openai_breaker.call(_call)

    async def score_with_anthropic(
        self,
        thesis_text: str,
        context: dict[str, Any],
        previous_scores: dict[str, Any] | None = None,
    ) -> CompellingnessScore | None:
        """Score/validate a theme using Claude (Tier 3).

        Uses tool_use for structured output. When previous_scores is provided,
        acts as a validation (consensus) pass.

        Args:
            thesis_text: Aggregated theme content.
            context: Dict with tickers, keywords, lifecycle_stage, document_count.
            previous_scores: Optional previous tier's scores for validation.

        Returns:
            CompellingnessScore or None if parsing fails.
        """
        if previous_scores:
            prompt = VALIDATION_PROMPT.format(
                thesis_text=thesis_text,
                previous_scores=json.dumps(previous_scores, indent=2),
            )
        else:
            prompt = SCORING_PROMPT.format(
                thesis_text=thesis_text,
                tickers=", ".join(context.get("tickers", [])),
                keywords=", ".join(context.get("keywords", [])),
                lifecycle_stage=context.get("lifecycle_stage", "unknown"),
                document_count=context.get("document_count", 0),
            )

        tool_schema = {
            "name": "submit_score",
            "description": "Submit the compellingness score assessment",
            "input_schema": {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number", "minimum": 0, "maximum": 10},
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "authority": {"type": "number"},
                            "evidence": {"type": "number"},
                            "reasoning": {"type": "number"},
                            "risk": {"type": "number"},
                            "actionability": {"type": "number"},
                            "technical": {"type": "number"},
                        },
                        "required": [
                            "authority", "evidence", "reasoning",
                            "risk", "actionability", "technical",
                        ],
                    },
                    "summary": {"type": "string"},
                    "evidence_quotes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "relevance": {"type": "string"},
                            },
                        },
                    },
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "time_horizon": {"type": "string"},
                    "key_risks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["overall_score", "dimensions", "summary"],
            },
        }

        async def _call() -> CompellingnessScore | None:
            client = self._get_anthropic_client()
            response = await client.messages.create(
                model=self._config.anthropic_model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": "submit_score"},
            )
            # Extract tool use input from response
            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_score":
                    raw_json = json.dumps(block.input)
                    return self._parse_score_response(
                        raw_json, tier="claude", model=self._config.anthropic_model,
                    )
            logger.warning("Anthropic response contained no tool_use block")
            return None

        return await self._anthropic_breaker.call(_call)

    def _parse_score_response(
        self,
        raw: str | None,
        tier: str,
        model: str,
    ) -> CompellingnessScore | None:
        """Parse raw JSON response into a CompellingnessScore.

        Returns None on any parse/validation failure (graceful degradation).
        """
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s response as JSON", tier)
            return None

        try:
            dims_data = data.get("dimensions", {})
            dimensions = DimensionScores(**dims_data)

            evidence_quotes = [
                EvidenceQuote(**eq) for eq in data.get("evidence_quotes", [])
            ]

            return CompellingnessScore(
                overall_score=float(data["overall_score"]),
                dimensions=dimensions,
                summary=data.get("summary", ""),
                evidence_quotes=evidence_quotes,
                tickers=data.get("tickers", []),
                time_horizon=data.get("time_horizon", "unknown"),
                key_risks=data.get("key_risks", []),
                tier_used=tier,  # type: ignore[arg-type]
                model_version=model,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to validate %s response: %s", tier, e)
            return None

    async def close(self) -> None:
        """Clean up SDK clients."""
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None
        if self._anthropic_client is not None:
            await self._anthropic_client.close()
            self._anthropic_client = None
