"""Tests for LLMClaimExtractor orchestration.

Fakes the shared JsonLLMClient to exercise: gate on content/key → call →
parse → schema-valid llm claims; and degradation (no key / empty content /
model unavailable) to an empty result.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.claims.llm_extractor import LLMClaimExtractor, LLMExtractionConfig


class _FakeScoringConfig:
    openai_model = "gpt-4o-mini"
    openai_api_key = "sk-test"
    llm_timeout = 30.0
    circuit_failure_threshold = 5
    circuit_recovery_timeout = 60.0


class _FakeLLM:
    """Stand-in for JsonLLMClient: a sync responder behind the async seam."""

    def __init__(self, responder, *, api_key: bool, model: str = "gpt-4o-mini") -> None:
        self._responder = responder
        self._api_key = api_key
        self.model = model
        self.calls = 0

    @property
    def has_api_key(self) -> bool:
        return self._api_key

    async def complete_json(self, prompt: str) -> Any:
        self.calls += 1
        return self._responder(prompt)


def _extractor(responder=lambda _p: None, *, api_key=True, config=None) -> LLMClaimExtractor:
    return LLMClaimExtractor(
        scoring_config=_FakeScoringConfig(),
        config=config,
        llm=_FakeLLM(responder, api_key=api_key),
    )


def _claim(subject, predicate, obj, confidence):
    return {"subject": subject, "predicate": predicate, "object": obj, "confidence": confidence}


@pytest.mark.asyncio
async def test_extracts_llm_claims() -> None:
    responder = lambda _p: {"claims": [_claim("TSMC", "supplies_to", "NVIDIA", 0.9)]}  # noqa: E731
    claims = await _extractor(responder).extract("d1", "TSMC supplies NVIDIA with wafers.")
    assert len(claims) == 1
    assert claims[0].extraction_method == "llm"
    assert (claims[0].subject_text, claims[0].predicate) == ("TSMC", "supplies_to")


@pytest.mark.asyncio
async def test_no_api_key_returns_empty_without_calling() -> None:
    ext = _extractor(api_key=False)
    assert await ext.extract("d1", "some content") == []
    assert ext._llm.calls == 0  # gated before the call


@pytest.mark.asyncio
async def test_blank_content_returns_empty_without_calling() -> None:
    ext = _extractor(lambda _p: {"claims": []})
    assert await ext.extract("d1", "   ") == []
    assert ext._llm.calls == 0


@pytest.mark.asyncio
async def test_model_unavailable_degrades_to_empty() -> None:
    # complete_json returns None (open breaker / API error / bad JSON) → [].
    assert await _extractor(lambda _p: None).extract("d1", "content") == []


@pytest.mark.asyncio
async def test_min_confidence_from_config_filters() -> None:
    responder = lambda _p: {  # noqa: E731
        "claims": [
            _claim("TSMC", "supplies_to", "A", 0.4),
            _claim("TSMC", "supplies_to", "B", 0.9),
        ]
    }
    ext = _extractor(responder, config=LLMExtractionConfig(min_confidence=0.5))
    claims = await ext.extract("d1", "content")
    assert [c.object_text for c in claims] == ["B"]
