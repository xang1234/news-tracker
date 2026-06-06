"""Tests for the semantic contradiction judge's pure helpers.

The judge wraps an LLM call to decide whether two claims semantically
contradict. The prompt construction and verdict parsing are pure functions
(tested here); the LLM round-trip itself is exercised via injected fakes in
the tier tests, never a live API.
"""

from __future__ import annotations

import pytest

from src.assertions.semantic_judge import (
    ContradictionVerdict,
    build_judge_prompt,
    parse_verdict,
)


class TestBuildJudgePrompt:
    def test_includes_both_claim_texts(self):
        prompt = build_judge_prompt("TSMC will raise wafer prices", "TSMC will cut wafer prices")
        assert "TSMC will raise wafer prices" in prompt
        assert "TSMC will cut wafer prices" in prompt

    def test_asks_for_the_three_relations(self):
        prompt = build_judge_prompt("a", "b").lower()
        assert "contradicts" in prompt
        assert "agrees" in prompt
        assert "unrelated" in prompt


class TestParseVerdict:
    def test_parses_valid_payload(self):
        v = parse_verdict({"relation": "contradicts", "confidence": 0.82})
        assert v == ContradictionVerdict(relation="contradicts", confidence=0.82)

    def test_clamps_confidence(self):
        assert parse_verdict({"relation": "agrees", "confidence": 1.5}).confidence == 1.0
        assert parse_verdict({"relation": "agrees", "confidence": -0.2}).confidence == 0.0

    def test_unknown_relation_is_none(self):
        assert parse_verdict({"relation": "maybe", "confidence": 0.9}) is None

    def test_missing_fields_is_none(self):
        assert parse_verdict({"relation": "contradicts"}) is None
        assert parse_verdict({"confidence": 0.9}) is None

    def test_non_numeric_confidence_is_none(self):
        assert parse_verdict({"relation": "agrees", "confidence": "high"}) is None

    def test_none_payload_is_none(self):
        assert parse_verdict(None) is None


class _NoKeyConfig:
    openai_api_key = None
    openai_model = "gpt-4o-mini"
    llm_timeout = 30.0
    circuit_failure_threshold = 5
    circuit_recovery_timeout = 60.0


@pytest.mark.asyncio
async def test_judge_short_circuits_without_api_key():
    # Misconfigured (enabled but no key): return None WITHOUT tripping the
    # breaker or creating a client (which would silently no-op every call).
    from src.assertions.semantic_judge import SemanticContradictionJudge

    judge = SemanticContradictionJudge(_NoKeyConfig())
    verdict = await judge.judge("a", "b")

    assert verdict is None
    assert judge._llm._breaker.consecutive_failures == 0  # breaker untouched
