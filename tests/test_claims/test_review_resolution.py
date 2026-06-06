"""Tests for review-task resolution + reconcile-on-approve (lrb).

Approving a held (claim_review/low_confidence) task must re-run reconciliation
for its claims — releasing the 7th.4 hold; any other resolution, or a non-held
task, must NOT reconcile. Fakes stand in for the repos + reconciliation engine.
"""

from __future__ import annotations

import pytest

from src.claims.review import ReviewTask
from src.claims.review_resolution import ReviewResolutionService, was_held_for_review
from src.claims.schemas import EvidenceClaim, make_claim_key
from src.contracts.intelligence.lanes import LANE_NARRATIVE


def _claim(claim_id: str) -> EvidenceClaim:
    key = make_claim_key(LANE_NARRATIVE, claim_id, "TSMC", "supplies_to", "NVIDIA")
    return EvidenceClaim(
        claim_id=claim_id,
        claim_key=key,
        lane=LANE_NARRATIVE,
        source_id="d1",
        subject_text="TSMC",
        predicate="supplies_to",
        object_text="NVIDIA",
        extraction_method="llm",
    )


def _task(task_type="claim_review", trigger_reason="low_confidence", claim_ids=("c1",)):
    return ReviewTask(
        task_id="review_x",
        task_type=task_type,
        trigger_reason=trigger_reason,
        claim_ids=list(claim_ids),
    )


class _FakeReviewRepo:
    """transition_task returns (prev_status, resolved task) and records the call."""

    def __init__(self, task: ReviewTask) -> None:
        self._task = task
        self.calls: list[tuple] = []

    async def transition_task(self, task_id, target_status, *, resolution=None, **kw):
        self.calls.append((task_id, target_status, resolution))
        self._task.status = target_status
        self._task.resolution = resolution
        return "pending", self._task


class _FakeClaimRepo:
    def __init__(self, claims: dict[str, EvidenceClaim]) -> None:
        self._claims = claims

    async def get_claim(self, claim_id: str) -> EvidenceClaim | None:
        return self._claims.get(claim_id)


class _FakeEngine:
    def __init__(self) -> None:
        self.reconciled: list[str] = []

    async def reconcile_claim(self, claim: EvidenceClaim):
        self.reconciled.append(claim.claim_id)
        return None


def _service(task, claims):
    repo = _FakeReviewRepo(task)
    engine = _FakeEngine()
    svc = ReviewResolutionService(repo, _FakeClaimRepo(claims), engine)
    return svc, repo, engine


class TestWasHeldForReview:
    def test_low_confidence_claim_review_is_held(self) -> None:
        assert was_held_for_review(_task("claim_review", "low_confidence")) is True

    def test_high_impact_claim_review_is_not_held(self) -> None:
        assert was_held_for_review(_task("claim_review", "high_impact_predicate")) is False

    def test_entity_review_is_not_held(self) -> None:
        assert was_held_for_review(_task("entity_review", "low_confidence")) is False


class TestResolve:
    @pytest.mark.asyncio
    async def test_approving_held_task_reconciles_its_claims(self) -> None:
        claims = {"c1": _claim("c1"), "c2": _claim("c2")}
        svc, repo, engine = _service(_task(claim_ids=("c1", "c2")), claims)
        task = await svc.resolve("review_x", "approved")
        assert task.status == "resolved"
        assert repo.calls == [("review_x", "resolved", "approved")]
        assert engine.reconciled == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_rejecting_held_task_does_not_reconcile(self) -> None:
        svc, _repo, engine = _service(_task(), {"c1": _claim("c1")})
        await svc.resolve("review_x", "rejected")
        assert engine.reconciled == []

    @pytest.mark.asyncio
    async def test_approving_non_held_task_does_not_reconcile(self) -> None:
        # A high-impact claim_review was never held → approval must not reconcile.
        task = _task(trigger_reason="high_impact_predicate")
        svc, _repo, engine = _service(task, {"c1": _claim("c1")})
        await svc.resolve("review_x", "approved")
        assert engine.reconciled == []

    @pytest.mark.asyncio
    async def test_missing_claim_is_skipped_others_reconcile(self) -> None:
        svc, _repo, engine = _service(_task(claim_ids=("c1", "gone")), {"c1": _claim("c1")})
        await svc.resolve("review_x", "approved")
        assert engine.reconciled == ["c1"]  # 'gone' skipped, no crash
