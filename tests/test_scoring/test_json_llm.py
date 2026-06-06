"""Tests for the shared breaker-guarded JSON-LLM client.

Exercises the degrade-to-None contract every grounded-LLM feature relies on:
no-key short-circuit (breaker untouched, no client built), parsed-dict on
success, and graceful None on empty content / malformed JSON / API error /
open breaker. The OpenAI client is faked by setting ``_client`` directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.scoring.json_llm import JsonLLMClient


class _Config:
    openai_api_key = "sk-test"
    openai_model = "gpt-4o-mini"
    llm_timeout = 30.0
    circuit_failure_threshold = 2
    circuit_recovery_timeout = 60.0


class _NoKeyConfig(_Config):
    openai_api_key = None


class _FakeCompletions:
    def __init__(self, content=None, *, error=None) -> None:
        self._content = content
        self._error = error
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_client(completions: _FakeCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _client_with(content=None, *, error=None) -> JsonLLMClient:
    llm = JsonLLMClient(_Config(), name="test")
    llm._client = _fake_client(_FakeCompletions(content, error=error))
    return llm


def test_has_api_key_and_model() -> None:
    llm = JsonLLMClient(_Config(), name="test")
    assert llm.has_api_key is True
    assert llm.model == "gpt-4o-mini"
    assert JsonLLMClient(_NoKeyConfig(), name="test").has_api_key is False


@pytest.mark.asyncio
async def test_no_key_short_circuits_without_touching_breaker() -> None:
    llm = JsonLLMClient(_NoKeyConfig(), name="test")
    assert await llm.complete_json("prompt") is None
    assert llm._breaker.consecutive_failures == 0
    assert llm._client is None  # client never built


@pytest.mark.asyncio
async def test_returns_parsed_dict_on_success() -> None:
    llm = _client_with('{"clauses": [{"text": "x"}]}')
    assert await llm.complete_json("p") == {"clauses": [{"text": "x"}]}


@pytest.mark.asyncio
async def test_empty_content_returns_none() -> None:
    assert await _client_with(None).complete_json("p") is None


@pytest.mark.asyncio
async def test_malformed_json_degrades_and_trips_breaker() -> None:
    llm = _client_with("not json at all")
    assert await llm.complete_json("p") is None
    assert llm._breaker.consecutive_failures == 1


@pytest.mark.asyncio
async def test_api_error_degrades_to_none() -> None:
    llm = _client_with(error=RuntimeError("boom"))
    assert await llm.complete_json("p") is None
    assert llm._breaker.consecutive_failures == 1


@pytest.mark.asyncio
async def test_open_breaker_returns_none_without_calling_api() -> None:
    completions = _FakeCompletions(error=RuntimeError("boom"))
    llm = JsonLLMClient(_Config(), name="test")
    llm._client = _fake_client(completions)
    # threshold=2: two failures open the circuit.
    await llm.complete_json("p")
    await llm.complete_json("p")
    calls_before = completions.calls
    assert await llm.complete_json("p") is None
    assert completions.calls == calls_before  # open breaker rejected before the API call
