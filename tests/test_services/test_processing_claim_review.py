"""Tests for routing low-confidence LLM claims through review (7th.4).

Exercises ProcessingService._hold_for_review and the _persist_claim hold branch
in isolation (built via __new__ to skip queue/DB construction): a low-confidence
LLM claim is upserted, held in the review queue, and NOT reconciled; trusted
claims (rule, hybrid, high-confidence LLM) reconcile normally.
"""

from __future__ import annotations

import pytest

from src.claims.llm_extractor import LLMExtractionConfig
from src.claims.schemas import EvidenceClaim, make_claim_id, make_claim_key
from src.contracts.intelligence.lanes import LANE_NARRATIVE
from src.services.processing_service import ProcessingService


def _claim(*, method="llm", confidence=0.5, subject="TSMC"):
    key = make_claim_key(LANE_NARRATIVE, "d1", subject, "supplies_to", "NVIDIA")
    return EvidenceClaim(
        claim_id=make_claim_id(key),
        claim_key=key,
        lane=LANE_NARRATIVE,
        source_id="d1",
        subject_text=subject,
        predicate="supplies_to",
        object_text="NVIDIA",
        confidence=confidence,
        extraction_method=method,
    )


class _FakeExtractor:
    def __init__(self, config=None) -> None:
        self.config = config or LLMExtractionConfig()


class _FakeReviewRepo:
    def __init__(self) -> None:
        self.tasks: list = []

    async def upsert_task(self, task):
        self.tasks.append(task)
        return task


class _FakeReconcileEngine:
    def __init__(self) -> None:
        self.reconciled: list = []

    async def reconcile_claim(self, claim):
        self.reconciled.append(claim)


class _FakeClaimRepo:
    def __init__(self) -> None:
        self.upserted: list = []

    async def upsert_claim(self, claim):
        self.upserted.append(claim)


def _service(*, review_repo=None, extractor=None, engine=None) -> ProcessingService:
    svc = ProcessingService.__new__(ProcessingService)
    svc._review_repo = review_repo
    svc._llm_extractor = extractor
    svc._reconciliation_engine = engine
    svc._claim_reconciliation_enabled = False
    svc._entity_resolver = None
    return svc


class TestHoldForReview:
    @pytest.mark.asyncio
    async def test_low_confidence_llm_is_held(self) -> None:
        repo = _FakeReviewRepo()
        svc = _service(review_repo=repo, extractor=_FakeExtractor())
        held = await svc._hold_for_review(_claim(method="llm", confidence=0.5))
        assert held is True
        assert len(repo.tasks) == 1
        assert repo.tasks[0].trigger_reason == "low_confidence"

    @pytest.mark.asyncio
    async def test_high_confidence_llm_not_held(self) -> None:
        repo = _FakeReviewRepo()
        svc = _service(review_repo=repo, extractor=_FakeExtractor())
        assert await svc._hold_for_review(_claim(method="llm", confidence=0.95)) is False
        assert repo.tasks == []

    @pytest.mark.asyncio
    async def test_hybrid_and_rule_not_held(self) -> None:
        repo = _FakeReviewRepo()
        svc = _service(review_repo=repo, extractor=_FakeExtractor())
        assert await svc._hold_for_review(_claim(method="hybrid", confidence=0.3)) is False
        assert await svc._hold_for_review(_claim(method="rule", confidence=0.3)) is False
        assert repo.tasks == []

    @pytest.mark.asyncio
    async def test_no_review_repo_means_never_held(self) -> None:
        svc = _service(review_repo=None, extractor=None)
        assert await svc._hold_for_review(_claim(method="llm", confidence=0.1)) is False


class TestPersistClaimHonorsHold:
    @pytest.mark.asyncio
    async def test_held_claim_is_persisted_but_not_reconciled(self) -> None:
        repo, engine, claims = _FakeReviewRepo(), _FakeReconcileEngine(), _FakeClaimRepo()
        svc = _service(review_repo=repo, extractor=_FakeExtractor(), engine=engine)

        await svc._persist_claim(_claim(method="llm", confidence=0.5), claims)

        assert len(claims.upserted) == 1  # still persisted as evidence
        assert repo.tasks and engine.reconciled == []  # held, not reconciled

    @pytest.mark.asyncio
    async def test_trusted_claim_reconciles(self) -> None:
        repo, engine, claims = _FakeReviewRepo(), _FakeReconcileEngine(), _FakeClaimRepo()
        svc = _service(review_repo=repo, extractor=_FakeExtractor(), engine=engine)

        await svc._persist_claim(_claim(method="hybrid", confidence=0.5), claims)

        assert len(claims.upserted) == 1
        assert repo.tasks == [] and len(engine.reconciled) == 1
