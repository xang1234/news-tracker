"""Tests for the POST /qa endpoint.

Overrides the qa-service dependency with a fake so we test the route contract
(feature gate, response shape, refusal signal) without an LLM or DB.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.auth import verify_api_key
from src.api.dependencies import get_qa_service
from src.api.routes.qa import router
from src.config.settings import get_settings
from src.qa.schemas import CONFIDENCE_HIGH, CONFIDENCE_INSUFFICIENT, AnswerSegment, CitedAnswer


@pytest.fixture(autouse=True)
def _restore_cited_qa_enabled():
    """Restore the global feature flag so these tests can't leak across order."""
    settings = get_settings()
    original = settings.cited_qa_enabled
    try:
        yield
    finally:
        settings.cited_qa_enabled = original


class _FakeQAService:
    def __init__(self, result: CitedAnswer) -> None:
        self._result = result

    async def answer(self, question: str) -> CitedAnswer:
        return self._result


def _client(result: CitedAnswer, *, enabled: bool = True) -> TestClient:
    settings = get_settings()
    settings.cited_qa_enabled = enabled
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_qa_service] = lambda: _FakeQAService(result)
    return TestClient(app)


def _answer(**overrides) -> CitedAnswer:
    base = {
        "question": "who supplies HBM to NVIDIA?",
        "segments": [AnswerSegment(text="SK Hynix supplies NVIDIA.", claim_ids=["claim_a"])],
        "confidence": CONFIDENCE_HIGH,
        "claim_count": 3,
        "generated_by": "llm",
        "model": "gpt-4o-mini",
        "generated_at": datetime(2026, 6, 5, tzinfo=UTC),
    }
    base.update(overrides)
    return CitedAnswer(**base)


def test_answers_with_cited_segments() -> None:
    resp = _client(_answer()).post("/qa", json={"question": "who supplies HBM to NVIDIA?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == "high"
    assert body["segments"][0]["claim_ids"] == ["claim_a"]
    assert body["claim_count"] == 3


def test_insufficient_grounding_signal() -> None:
    ans = _answer(segments=[], confidence=CONFIDENCE_INSUFFICIENT, generated_by="template")
    body = _client(ans).post("/qa", json={"question": "obscure question"}).json()
    assert body["confidence"] == "insufficient"
    assert body["segments"] == []


def test_404_when_feature_disabled() -> None:
    resp = _client(_answer(), enabled=False).post("/qa", json={"question": "q"})
    assert resp.status_code == 404


def test_disabled_route_does_not_construct_service() -> None:
    settings = get_settings()
    settings.cited_qa_enabled = False
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_api_key] = lambda: "test-key"

    def _must_not_run():
        raise AssertionError("qa service constructed for a disabled route")

    app.dependency_overrides[get_qa_service] = _must_not_run
    resp = TestClient(app).post("/qa", json={"question": "q"})
    assert resp.status_code == 404


def test_empty_question_rejected() -> None:
    resp = _client(_answer()).post("/qa", json={"question": ""})
    assert resp.status_code == 422
