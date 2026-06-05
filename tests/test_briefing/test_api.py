"""Tests for the GET /themes/{theme_id}/briefing endpoint.

Overrides the briefing-service dependency with a fake so we test the route
contract (feature gate, 404, response shape) without an LLM or DB.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.auth import verify_api_key
from src.api.dependencies import get_briefing_service
from src.api.routes.themes import router
from src.briefing.schemas import BriefingClause, ThemeBriefing
from src.config.settings import get_settings


class _FakeBriefingService:
    def __init__(self, result: ThemeBriefing | None) -> None:
        self._result = result

    async def generate(self, theme_id: str) -> ThemeBriefing | None:
        return self._result


def _client(result: ThemeBriefing | None, *, enabled: bool = True) -> TestClient:
    settings = get_settings()
    settings.theme_briefing_enabled = enabled
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_briefing_service] = lambda: _FakeBriefingService(result)
    return TestClient(app)


def _briefing() -> ThemeBriefing:
    return ThemeBriefing(
        theme_id="t1",
        clauses=[BriefingClause(text="TSMC supplies NVIDIA.", claim_ids=["claim_a"])],
        generated_by="llm",
        claim_count=3,
        model="gpt-4o-mini",
        generated_at=datetime(2026, 6, 5, tzinfo=UTC),
    )


def test_returns_briefing_with_cited_clauses() -> None:
    resp = _client(_briefing()).get("/themes/t1/briefing")
    assert resp.status_code == 200
    body = resp.json()
    assert body["theme_id"] == "t1"
    assert body["generated_by"] == "llm"
    assert body["claim_count"] == 3
    assert body["clauses"][0]["text"] == "TSMC supplies NVIDIA."
    assert body["clauses"][0]["claim_ids"] == ["claim_a"]


def test_404_when_feature_disabled() -> None:
    resp = _client(_briefing(), enabled=False).get("/themes/t1/briefing")
    assert resp.status_code == 404


def test_404_when_theme_missing() -> None:
    resp = _client(None).get("/themes/ghost/briefing")
    assert resp.status_code == 404


def test_every_clause_is_cited() -> None:
    resp = _client(_briefing()).get("/themes/t1/briefing")
    for clause in resp.json()["clauses"]:
        assert clause["claim_ids"], "no uncited clauses allowed"
