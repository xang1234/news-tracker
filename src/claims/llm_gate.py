"""LLM fallback gate for the entity resolver cascade.

Enforces disciplined escalation: LLM calls are a controlled
exception path, not the default. Every invocation is gated by
policy checks and produces an auditable provenance record.

Gate decision flow:
    1. Master switch (llm_fallback_enabled)
    2. Passage qualifies as high-value (length + predicate check)
    3. Budget not exhausted (per-run and daily caps)
    4. → Approve with reason, or Deny with reason

LLM-proposed resolutions:
    - Get ResolverTier.LLM_PROPOSED (not EXACT/ALIAS/FUZZY)
    - Carry lower confidence than deterministic tiers
    - Include full provenance metadata for audit
    - Require review before becoming authoritative
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.claims.config import ClaimsConfig

logger = logging.getLogger(__name__)


class GateVerdict(str, Enum):
    """Whether the LLM fallback gate approved or denied."""

    APPROVED = "approved"
    DENIED = "denied"


class DenyReason(str, Enum):
    """Why the gate denied an LLM fallback request."""

    MASTER_DISABLED = "master_disabled"
    PASSAGE_TOO_SHORT = "passage_too_short"
    PREDICATE_NOT_HIGH_VALUE = "predicate_not_high_value"
    DAILY_BUDGET_EXHAUSTED = "daily_budget_exhausted"
    RUN_BUDGET_EXHAUSTED = "run_budget_exhausted"


@dataclass(frozen=True)
class GateDecision:
    """Result of evaluating the LLM fallback gate.

    Attributes:
        verdict: Whether the gate approved or denied.
        deny_reason: Why the gate denied (None if approved).
        mention: The raw mention that was evaluated.
        passage_length: Length of the source passage.
        predicate: The claim predicate (if known at gate time).
        run_invocations: How many LLM calls this run has used.
        daily_invocations: How many LLM calls today has used.
    """

    verdict: GateVerdict
    mention: str
    deny_reason: DenyReason | None = None
    passage_length: int = 0
    predicate: str | None = None
    run_invocations: int = 0
    daily_invocations: int = 0

    @property
    def approved(self) -> bool:
        return self.verdict == GateVerdict.APPROVED


@dataclass
class FallbackProvenance:
    """Audit record for an LLM fallback invocation.

    Captures everything needed to understand why the LLM was called,
    what it returned, and whether the result needs review.

    Attributes:
        gate_decision: The gate decision that authorized this call.
        model_id: Identifier of the LLM model used.
        invoked_at: When the LLM was called.
        proposed_concept_id: What the LLM proposed as a resolution.
        proposed_name: Human-readable name the LLM returned.
        llm_confidence: Confidence the LLM reported (if any).
        effective_confidence: Confidence applied after gate policy.
        needs_review: Whether this resolution requires human review.
        review_reason: Why review is needed (if applicable).
        metadata: Extra audit data (prompt hash, token count, etc.).
    """

    gate_decision: GateDecision
    model_id: str = ""
    invoked_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    proposed_concept_id: str | None = None
    proposed_name: str | None = None
    llm_confidence: float | None = None
    effective_confidence: float = 0.0
    needs_review: bool = True
    review_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Serialize to a dict suitable for claim metadata storage."""
        return {
            "llm_fallback": True,
            "model_id": self.model_id,
            "invoked_at": self.invoked_at.isoformat(),
            "proposed_concept_id": self.proposed_concept_id,
            "proposed_name": self.proposed_name,
            "llm_confidence": self.llm_confidence,
            "effective_confidence": self.effective_confidence,
            "needs_review": self.needs_review,
            "review_reason": self.review_reason,
            "gate_verdict": self.gate_decision.verdict.value,
            "gate_mention": self.gate_decision.mention,
            "passage_length": self.gate_decision.passage_length,
            "predicate": self.gate_decision.predicate,
            **self.metadata,
        }


class FallbackBudget:
    """Tracks LLM invocation counts for budget enforcement.

    Thread-safe within a single asyncio event loop (no locks needed).
    Resets daily count on date change; per-run counts are scoped by
    run_id.
    """

    def __init__(self) -> None:
        self._daily_count: int = 0
        self._daily_date: str = ""
        self._run_counts: dict[str, int] = {}

    @property
    def daily_count(self) -> int:
        self._maybe_reset_daily()
        return self._daily_count

    def run_count(self, run_id: str) -> int:
        return self._run_counts.get(run_id, 0)

    def record_invocation(self, run_id: str | None = None) -> None:
        """Record that an LLM fallback invocation occurred."""
        self._maybe_reset_daily()
        self._daily_count += 1
        if run_id:
            self._run_counts[run_id] = self._run_counts.get(run_id, 0) + 1

    def _maybe_reset_daily(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_count = 0
            self._daily_date = today
            # Day changed — per-run counts from yesterday are stale
            self._run_counts.clear()


class FallbackGate:
    """Policy engine that decides whether LLM fallback is warranted.

    Pure policy evaluation — no I/O, no LLM calls. The gate evaluates
    a set of criteria and returns a decision. The caller is responsible
    for acting on the decision.

    Usage:
        gate = FallbackGate(config)
        decision = gate.evaluate(
            mention="Taiwan Semi",
            passage_length=250,
            predicate="supplies_to",
            run_id="run_123",
        )
        if decision.approved:
            # proceed with LLM call
            ...
    """

    def __init__(self, config: ClaimsConfig) -> None:
        self._config = config
        self._budget = FallbackBudget()

    @property
    def budget(self) -> FallbackBudget:
        """Access the budget tracker (for inspection/testing)."""
        return self._budget

    def evaluate(
        self,
        mention: str,
        *,
        passage_length: int = 0,
        predicate: str | None = None,
        run_id: str | None = None,
    ) -> GateDecision:
        """Evaluate whether LLM fallback should be invoked.

        Checks are ordered from cheapest to most expensive:
        1. Master switch
        2. Passage length
        3. Predicate high-value check
        4. Budget caps

        Args:
            mention: The unresolved text mention.
            passage_length: Character length of the source passage.
            predicate: The claim predicate (if known).
            run_id: Current lane run ID (for per-run budget).

        Returns:
            GateDecision with verdict and reason.
        """
        # Snapshot budget counts once to avoid redundant lookups
        daily = self._budget.daily_count
        run_used = self._budget.run_count(run_id) if run_id else 0
        base = {
            "mention": mention,
            "passage_length": passage_length,
            "predicate": predicate,
            "run_invocations": run_used,
            "daily_invocations": daily,
        }

        # 1. Master switch
        if not self._config.llm_fallback_enabled:
            return GateDecision(
                verdict=GateVerdict.DENIED,
                deny_reason=DenyReason.MASTER_DISABLED,
                **base,
            )

        # 2. Passage length
        if passage_length < self._config.min_passage_length:
            logger.debug(
                "LLM gate denied: passage too short (%d < %d) for %r",
                passage_length,
                self._config.min_passage_length,
                mention,
            )
            return GateDecision(
                verdict=GateVerdict.DENIED,
                deny_reason=DenyReason.PASSAGE_TOO_SHORT,
                **base,
            )

        # 3. Predicate high-value check
        hvp = self._config.high_value_predicates
        if predicate is not None and hvp and predicate not in hvp:
            logger.debug(
                "LLM gate denied: predicate %r not in high-value set for %r",
                predicate,
                mention,
            )
            return GateDecision(
                verdict=GateVerdict.DENIED,
                deny_reason=DenyReason.PREDICATE_NOT_HIGH_VALUE,
                **base,
            )

        # 4a. Daily budget
        if daily >= self._config.daily_llm_budget:
            logger.info(
                "LLM gate denied: daily budget exhausted (%d/%d)",
                daily,
                self._config.daily_llm_budget,
            )
            return GateDecision(
                verdict=GateVerdict.DENIED,
                deny_reason=DenyReason.DAILY_BUDGET_EXHAUSTED,
                **base,
            )

        # 4b. Per-run budget
        if run_id and run_used >= self._config.per_run_llm_budget:
            logger.info(
                "LLM gate denied: per-run budget exhausted (%d/%d) for %s",
                run_used,
                self._config.per_run_llm_budget,
                run_id,
            )
            return GateDecision(
                verdict=GateVerdict.DENIED,
                deny_reason=DenyReason.RUN_BUDGET_EXHAUSTED,
                **base,
            )

        logger.debug(
            "LLM gate approved for %r (passage=%d, predicate=%s)",
            mention,
            passage_length,
            predicate,
        )
        return GateDecision(
            verdict=GateVerdict.APPROVED,
            **base,
        )

    def record_invocation(self, run_id: str | None = None) -> None:
        """Record that an approved invocation was actually executed.

        Call this AFTER the LLM call completes (whether it succeeded
        or failed) to keep budget counts accurate.
        """
        self._budget.record_invocation(run_id)

    def make_provenance(
        self,
        decision: GateDecision,
        *,
        model_id: str = "",
        proposed_concept_id: str | None = None,
        proposed_name: str | None = None,
        llm_confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FallbackProvenance:
        """Build a provenance record for an LLM fallback invocation.

        Automatically determines effective_confidence and review
        requirement based on gate policy.
        """
        effective = self._config.llm_proposed_confidence
        if llm_confidence is not None:
            # Use the lower of policy default and LLM-reported confidence
            effective = min(llm_confidence, effective)

        needs_review = effective < self._config.auto_approve_threshold
        review_reason = ""
        if needs_review:
            review_reason = (
                f"LLM-proposed resolution below auto-approve threshold "
                f"({effective:.2f} < {self._config.auto_approve_threshold:.2f})"
            )

        return FallbackProvenance(
            gate_decision=decision,
            model_id=model_id,
            proposed_concept_id=proposed_concept_id,
            proposed_name=proposed_name,
            llm_confidence=llm_confidence,
            effective_confidence=effective,
            needs_review=needs_review,
            review_reason=review_reason,
            metadata=metadata or {},
        )
