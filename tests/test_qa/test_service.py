"""Tests for CitedQAService orchestration.

Fakes retrieval + the LLM seam to exercise: retrieve -> sufficiency gate ->
LLM (guarded) -> validate citations -> wrap; and every fallback (insufficient
grounding refusal, breaker-open/no-key templated low-confidence). No model, no DB.
"""

from __future__ import annotations

from typing import Any

from src.claims.schemas import EvidenceClaim, make_claim_key
from src.qa.config import QAConfig
from src.qa.schemas import (
    CONFIDENCE_HIGH,
    CONFIDENCE_INSUFFICIENT,
    CONFIDENCE_LOW,
    CitedAnswer,
)
from src.qa.service import CitedQAService
from src.retrieval.schemas import RetrievedClaim


def _claim(claim_id: str) -> EvidenceClaim:
    key = make_claim_key("narrative", claim_id, "SK Hynix", "supplies_to", "NVIDIA")
    return EvidenceClaim(
        claim_id=claim_id,
        claim_key=key,
        lane="narrative",
        source_id=claim_id,
        predicate="supplies_to",
        subject_text="SK Hynix",
        object_text="NVIDIA",
        contract_version="v1",
    )


class _FakeRetrieval:
    def __init__(self, scored: list[tuple[str, float]]) -> None:
        self._scored = scored
        self.last_filter: Any = None
        self.last_limit: int | None = None

    async def retrieve(self, query, *, limit=None, filters=None):
        self.last_filter = filters
        self.last_limit = limit
        return [RetrievedClaim(claim=_claim(cid), score=score) for cid, score in self._scored]


class _FakeScoringConfig:
    openai_model = "gpt-4o-mini"
    openai_api_key = "sk-test"
    llm_timeout = 30.0
    circuit_failure_threshold = 5
    circuit_recovery_timeout = 60.0


def _service(*, retrieval, config=None, llm=None, api_key="sk-test") -> CitedQAService:
    svc = CitedQAService(
        retrieval_service=retrieval,
        scoring_config=_FakeScoringConfig(),
        config=config or QAConfig(),
    )
    sync_llm = llm if llm is not None else (lambda prompt: None)

    async def _async_llm(prompt: str, _fn=sync_llm) -> Any:
        return _fn(prompt)

    svc._call_llm = _async_llm  # type: ignore[assignment]
    svc._has_api_key = lambda: api_key is not None  # type: ignore[assignment]
    return svc


class TestSufficiencyGate:
    async def test_no_claims_is_insufficient(self) -> None:
        svc = _service(retrieval=_FakeRetrieval([]))
        ans = await svc.answer("what is the latest on HBM?")
        assert isinstance(ans, CitedAnswer)
        assert ans.confidence == CONFIDENCE_INSUFFICIENT
        assert ans.segments == []
        assert ans.is_grounded is False

    async def test_weak_top_score_is_insufficient(self) -> None:
        # Top similarity below min_grounding_score → refuse even though claims exist.
        svc = _service(
            retrieval=_FakeRetrieval([("claim_a", 0.20), ("claim_b", 0.10)]),
            config=QAConfig(min_grounding_score=0.35),
            llm=lambda p: {"segments": [{"text": "x", "claim_ids": ["claim_a"]}]},
        )
        ans = await svc.answer("q")
        assert ans.confidence == CONFIDENCE_INSUFFICIENT
        assert ans.segments == []


class TestAnswer:
    async def test_llm_path_high_confidence(self) -> None:
        svc = _service(
            retrieval=_FakeRetrieval([("claim_a", 0.9), ("claim_b", 0.8)]),
            llm=lambda p: {
                "segments": [{"text": "SK Hynix supplies NVIDIA.", "claim_ids": ["claim_a"]}]
            },
        )
        ans = await svc.answer("who supplies HBM to NVIDIA?")
        assert ans.confidence == CONFIDENCE_HIGH
        assert ans.generated_by == "llm"
        assert ans.claim_count == 2
        assert ans.segments[0].claim_ids == ["claim_a"]

    async def test_no_theme_filter_corpus_wide(self) -> None:
        retrieval = _FakeRetrieval([("claim_a", 0.9)])
        svc = _service(
            retrieval=retrieval,
            config=QAConfig(max_claims=8, min_confidence=0.4),
            llm=lambda p: {"segments": [{"text": "x.", "claim_ids": ["claim_a"]}]},
        )
        await svc.answer("q")
        assert retrieval.last_limit == 8
        assert retrieval.last_filter.theme_id is None  # ask-anything: corpus-wide
        assert retrieval.last_filter.min_confidence == 0.4

    async def test_templated_low_confidence_when_llm_unavailable(self) -> None:
        svc = _service(
            retrieval=_FakeRetrieval([("claim_a", 0.9), ("claim_b", 0.8)]),
            llm=lambda p: None,  # breaker open / API error
        )
        ans = await svc.answer("q")
        assert ans.confidence == CONFIDENCE_LOW
        assert ans.generated_by == "template"
        assert ans.segments[0].claim_ids == ["claim_a"]

    async def test_templated_low_confidence_when_no_api_key(self) -> None:
        called = False

        def llm(p):
            nonlocal called
            called = True
            return {"segments": [{"text": "x", "claim_ids": ["claim_a"]}]}

        svc = _service(retrieval=_FakeRetrieval([("claim_a", 0.9)]), llm=llm, api_key=None)
        ans = await svc.answer("q")
        assert ans.confidence == CONFIDENCE_LOW
        assert called is False

    async def test_falls_back_when_llm_all_hallucinated(self) -> None:
        svc = _service(
            retrieval=_FakeRetrieval([("claim_a", 0.9)]),
            llm=lambda p: {"segments": [{"text": "bogus", "claim_ids": ["ghost"]}]},
        )
        ans = await svc.answer("q")
        assert ans.confidence == CONFIDENCE_LOW
        assert ans.generated_by == "template"
        assert ans.segments[0].claim_ids == ["claim_a"]

    async def test_every_segment_cited_on_all_paths(self) -> None:
        svc = _service(
            retrieval=_FakeRetrieval([("claim_a", 0.9), ("claim_b", 0.8)]),
            llm=lambda p: {
                "segments": [
                    {"text": "cited", "claim_ids": ["claim_a"]},
                    {"text": "uncited", "claim_ids": []},
                ]
            },
        )
        ans = await svc.answer("q")
        assert all(s.claim_ids for s in ans.segments)
