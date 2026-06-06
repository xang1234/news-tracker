"""Resolve review-queue tasks and propagate the decision.

The `7th.4` hold (``processing_service._hold_for_review``) parks low-confidence
LLM claims in the review queue and *skips reconciliation*, so they never feed
assertions. This is the other half of that loop: when such a held claim is
**approved**, its claims are re-reconciled so they can finally feed assertions;
any other resolution (rejected/deferred/…) leaves them held.

Only tasks created by the hold (``claim_review`` + ``low_confidence``) trigger
re-reconciliation — other review tasks' claims were reconciled at persist time
and must not be needlessly recomputed on approval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.claims.review import ReviewTask

if TYPE_CHECKING:
    from src.assertions.reconciliation_engine import ClaimReconciliationEngine
    from src.claims.repository import ClaimRepository
    from src.claims.review_repository import ReviewRepository

logger = structlog.get_logger(__name__)

RESOLUTION_APPROVED = "approved"


def was_held_for_review(task: ReviewTask) -> bool:
    """True for tasks created by the low-confidence-LLM hold (7th.4)."""
    return task.task_type == "claim_review" and task.trigger_reason == "low_confidence"


class ReviewResolutionService:
    """Resolve a review task and reconcile its claims when a hold is approved."""

    def __init__(
        self,
        review_repo: ReviewRepository,
        claim_repo: ClaimRepository,
        reconciliation_engine: ClaimReconciliationEngine,
    ) -> None:
        self._review_repo = review_repo
        self._claim_repo = claim_repo
        self._reconciliation_engine = reconciliation_engine

    async def resolve(
        self,
        task_id: str,
        resolution: str,
        *,
        resolution_notes: str | None = None,
        assigned_to: str | None = None,
    ) -> ReviewTask:
        """Resolve ``task_id`` with ``resolution``; reconcile held claims on approval.

        Approving a held (claim_review/low_confidence) task re-runs reconciliation
        for its claims — releasing them from the hold into the assertion layer.

        Reconciliation runs *before* the task is transitioned, so a reconciliation
        failure leaves the task in its pre-approval state rather than recording an
        "approved" task whose claims never reached the assertion layer. (A revert
        after transition isn't possible: ``resolved`` is a terminal state.)
        """
        if resolution == RESOLUTION_APPROVED:
            held = await self._review_repo.get_task(task_id)
            if held is None:
                raise ValueError(f"Review task not found: {task_id}")
            if was_held_for_review(held):
                await self._reconcile_claims(held)

        _previous, task = await self._review_repo.transition_task(
            task_id,
            "resolved",
            resolution=resolution,
            resolution_notes=resolution_notes,
            assigned_to=assigned_to,
        )
        return task

    async def _reconcile_claims(self, task: ReviewTask) -> None:
        """Reconcile every claim a held task references; missing claims are skipped."""
        for claim_id in task.claim_ids:
            claim = await self._claim_repo.get_claim(claim_id)
            if claim is None:
                logger.warning(
                    "Approved review task references a missing claim",
                    task_id=task.task_id,
                    claim_id=claim_id,
                )
                continue
            await self._reconciliation_engine.reconcile_claim(claim)
