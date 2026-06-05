"""Tests for ThemeBriefingService orchestration.

Fakes the retrieval service, theme repo, and the LLM call so we exercise the
wiring — theme lookup → retrieve grounding claims → LLM (guarded) → validate
citations → wrap; plus every fallback path (theme missing, no claims, breaker
open, no API key, LLM returns nothing groundable). No model, no DB.
"""

from __future__ import annotations

from typing import Any

from src.briefing.config import BriefingConfig
from src.briefing.generator import ThemeBriefingService
from src.briefing.schemas import ThemeBriefing
from src.claims.schemas import EvidenceClaim, make_claim_key
from src.retrieval.schemas import ClaimRetrievalFilter, RetrievedClaim


def _claim(claim_id: str) -> EvidenceClaim:
    key = make_claim_key("narrative", claim_id, "TSMC", "supplies_to", "NVIDIA")
    return EvidenceClaim(
        claim_id=claim_id,
        claim_key=key,
        lane="narrative",
        source_id=claim_id,
        predicate="supplies_to",
        subject_text="TSMC",
        object_text="NVIDIA",
        contract_version="v1",
    )


class _FakeTheme:
    def __init__(self, theme_id: str) -> None:
        self.theme_id = theme_id
        self.name = "AI accelerators"
        self.top_keywords = ["gpu", "hbm"]


class _FakeThemeRepo:
    def __init__(self, themes: dict[str, Any]) -> None:
        self._themes = themes

    async def get_by_id(self, theme_id: str):
        return self._themes.get(theme_id)


class _FakeRetrieval:
    def __init__(self, claims: list[EvidenceClaim]) -> None:
        self._claims = claims
        self.last_filter: ClaimRetrievalFilter | None = None
        self.last_limit: int | None = None

    async def retrieve(self, query, *, limit=None, filters=None):
        self.last_filter = filters
        self.last_limit = limit
        return [RetrievedClaim(claim=c, score=0.9) for c in self._claims]


class _FakeScoringConfig:
    openai_model = "gpt-4o-mini"
    openai_api_key = "sk-test"
    llm_timeout = 30.0
    circuit_failure_threshold = 5
    circuit_recovery_timeout = 60.0


def _service(
    *, theme_repo, retrieval, config=None, llm=None, api_key="sk-test"
) -> ThemeBriefingService:
    svc = ThemeBriefingService(
        theme_repository=theme_repo,
        retrieval_service=retrieval,
        scoring_config=_FakeScoringConfig(),
        config=config or BriefingConfig(),
    )
    # Inject the LLM call seam directly (bypasses OpenAI client + breaker).
    # The real seam is async, so wrap the test's sync lambda.
    sync_llm = llm if llm is not None else (lambda prompt: None)

    async def _async_llm(prompt: str, _fn=sync_llm) -> Any:
        return _fn(prompt)

    svc._call_llm = _async_llm  # type: ignore[assignment]
    svc._has_api_key = lambda: api_key is not None  # type: ignore[assignment]
    return svc


class TestGenerate:
    async def test_returns_none_when_theme_missing(self) -> None:
        svc = _service(theme_repo=_FakeThemeRepo({}), retrieval=_FakeRetrieval([]))
        assert await svc.generate("ghost") is None

    async def test_llm_path_produces_cited_clauses(self) -> None:
        claims = [_claim("claim_a"), _claim("claim_b")]
        llm = lambda prompt: {  # noqa: E731
            "clauses": [{"text": "TSMC supplies NVIDIA.", "claim_ids": ["claim_a"]}]
        }
        svc = _service(
            theme_repo=_FakeThemeRepo({"t1": _FakeTheme("t1")}),
            retrieval=_FakeRetrieval(claims),
            llm=llm,
        )

        brief = await svc.generate("t1")

        assert isinstance(brief, ThemeBriefing)
        assert brief.generated_by == "llm"
        assert brief.theme_id == "t1"
        assert brief.claim_count == 2
        assert [c.text for c in brief.clauses] == ["TSMC supplies NVIDIA."]
        assert brief.clauses[0].claim_ids == ["claim_a"]

    async def test_retrieval_scoped_to_theme(self) -> None:
        retrieval = _FakeRetrieval([_claim("claim_a")])
        svc = _service(
            theme_repo=_FakeThemeRepo({"t1": _FakeTheme("t1")}),
            retrieval=retrieval,
            config=BriefingConfig(max_claims=9, min_confidence=0.5),
            llm=lambda p: {"clauses": [{"text": "X.", "claim_ids": ["claim_a"]}]},
        )

        await svc.generate("t1")

        assert retrieval.last_filter.theme_id == "t1"
        assert retrieval.last_filter.min_confidence == 0.5
        assert retrieval.last_limit == 9

    async def test_falls_back_to_template_when_llm_returns_none(self) -> None:
        claims = [_claim("claim_a"), _claim("claim_b")]
        svc = _service(
            theme_repo=_FakeThemeRepo({"t1": _FakeTheme("t1")}),
            retrieval=_FakeRetrieval(claims),
            llm=lambda prompt: None,  # breaker open / API error degrade to None
        )

        brief = await svc.generate("t1")

        assert brief.generated_by == "template"
        assert len(brief.clauses) == 2
        assert brief.clauses[0].claim_ids == ["claim_a"]

    async def test_falls_back_when_no_api_key(self) -> None:
        called = False

        def llm(prompt):
            nonlocal called
            called = True
            return {"clauses": [{"text": "X.", "claim_ids": ["claim_a"]}]}

        svc = _service(
            theme_repo=_FakeThemeRepo({"t1": _FakeTheme("t1")}),
            retrieval=_FakeRetrieval([_claim("claim_a")]),
            llm=llm,
            api_key=None,
        )

        brief = await svc.generate("t1")

        assert brief.generated_by == "template"
        assert called is False  # must not attempt the LLM without a key

    async def test_falls_back_when_llm_output_all_hallucinated(self) -> None:
        # LLM cites only ids not in the retrieved set → nothing groundable →
        # template, not an empty/uncited brief.
        svc = _service(
            theme_repo=_FakeThemeRepo({"t1": _FakeTheme("t1")}),
            retrieval=_FakeRetrieval([_claim("claim_a")]),
            llm=lambda p: {"clauses": [{"text": "Bogus.", "claim_ids": ["claim_ghost"]}]},
        )

        brief = await svc.generate("t1")

        assert brief.generated_by == "template"
        assert brief.clauses[0].claim_ids == ["claim_a"]

    async def test_no_claims_yields_empty_template_brief(self) -> None:
        svc = _service(
            theme_repo=_FakeThemeRepo({"t1": _FakeTheme("t1")}),
            retrieval=_FakeRetrieval([]),
        )

        brief = await svc.generate("t1")

        assert brief.generated_by == "template"
        assert brief.clauses == []
        assert brief.claim_count == 0

    async def test_no_uncited_clauses_ever(self) -> None:
        # Invariant: whatever path, every clause has >=1 citation.
        svc = _service(
            theme_repo=_FakeThemeRepo({"t1": _FakeTheme("t1")}),
            retrieval=_FakeRetrieval([_claim("claim_a"), _claim("claim_b")]),
            llm=lambda p: {
                "clauses": [
                    {"text": "Cited.", "claim_ids": ["claim_a"]},
                    {"text": "Uncited.", "claim_ids": []},
                ]
            },
        )

        brief = await svc.generate("t1")

        assert all(c.claim_ids for c in brief.clauses)
