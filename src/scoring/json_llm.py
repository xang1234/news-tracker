"""Shared breaker-guarded JSON-completion LLM client.

Several grounded-LLM features — semantic contradiction judging, theme
briefings, cited Q&A — make the exact same call: a single OpenAI chat
completion forced to JSON, lazily built from ``ScoringConfig`` and guarded by a
``GenericCircuitBreaker``, degrading to ``None`` on any failure (no API key,
open breaker, API error, or unparseable JSON). This is that one call, extracted
so each feature owns only its prompt and its parsing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    # Type-only: a consumer importing this module shouldn't have to pull in the
    # scoring service package until a client is actually constructed.
    from src.scoring.circuit_breaker import GenericCircuitBreaker
    from src.scoring.config import ScoringConfig

logger = structlog.get_logger(__name__)


class JsonLLMClient:
    """Lazily-built OpenAI client returning parsed JSON dicts behind a breaker."""

    def __init__(
        self,
        config: ScoringConfig,
        *,
        name: str,
        breaker: GenericCircuitBreaker | None = None,
    ) -> None:
        from src.scoring.circuit_breaker import GenericCircuitBreaker

        self._config = config
        self._name = name
        self._client: Any = None
        self._breaker = breaker or GenericCircuitBreaker(
            failure_threshold=config.circuit_failure_threshold,
            recovery_timeout=config.circuit_recovery_timeout,
            name=name,
        )

    @property
    def has_api_key(self) -> bool:
        """Whether an OpenAI key is configured (cheap gate before building a prompt)."""
        return bool(self._config.openai_api_key)

    @property
    def model(self) -> str:
        """The model these completions run against (for response attribution)."""
        return self._config.openai_model

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            api_key = self._config.openai_api_key
            self._client = openai.AsyncOpenAI(
                api_key=api_key.get_secret_value() if api_key else None,
                timeout=self._config.llm_timeout,
            )
        return self._client

    async def complete_json(self, prompt: str) -> dict[str, Any] | None:
        """Return the model's JSON object for ``prompt``, or None on any failure.

        Short-circuits to None when no API key is configured *without* touching
        the breaker or creating a client — so a feature that is enabled but
        misconfigured no-ops cleanly instead of tripping the breaker every call.
        """
        if not self.has_api_key:
            return None

        async def _call() -> dict[str, Any] | None:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content) if content else None

        try:
            return await self._breaker.call(_call)
        except Exception as e:  # breaker open, API error, bad JSON — degrade
            logger.warning("JSON LLM call failed", name=self._name, error=str(e))
            return None
