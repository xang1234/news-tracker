"""Tests for the LLM fallback gate and provenance system.

Verifies that LLM invocation is a controlled exception path:
gated by master switch, passage criteria, budget caps, and
predicate filtering. Tests are pure/in-memory — no LLM calls.
"""

from __future__ import annotations

import pytest

from src.claims.config import ClaimsConfig
from src.claims.llm_gate import (
    DenyReason,
    FallbackBudget,
    FallbackGate,
    GateDecision,
    GateVerdict,
)

# -- Helpers ---------------------------------------------------------------


def _make_config(**overrides) -> ClaimsConfig:
    """Build a ClaimsConfig with sensible test defaults."""
    defaults = {
        "llm_fallback_enabled": True,
        "min_passage_length": 80,
        "high_value_predicates": ["supplies_to", "competes_with"],
        "daily_llm_budget": 50,
        "per_run_llm_budget": 10,
        "llm_proposed_confidence": 0.45,
        "auto_approve_threshold": 0.85,
    }
    defaults.update(overrides)
    return ClaimsConfig(**defaults)


def _make_gate(**config_overrides) -> FallbackGate:
    return FallbackGate(_make_config(**config_overrides))


# -- Master switch tests ---------------------------------------------------


class TestMasterSwitch:
    """Gate denies everything when master switch is off."""

    def test_disabled_denies(self) -> None:
        gate = _make_gate(llm_fallback_enabled=False)
        decision = gate.evaluate("TSMC", passage_length=200)
        assert not decision.approved
        assert decision.deny_reason == DenyReason.MASTER_DISABLED

    def test_enabled_can_approve(self) -> None:
        gate = _make_gate()
        decision = gate.evaluate(
            "TSMC",
            passage_length=200,
            predicate="supplies_to",
        )
        assert decision.approved
        assert decision.deny_reason is None


# -- Passage length tests --------------------------------------------------


class TestPassageLength:
    """Gate denies short passages that don't justify LLM cost."""

    def test_too_short_denied(self) -> None:
        gate = _make_gate(min_passage_length=100)
        decision = gate.evaluate("TSMC", passage_length=50, predicate="supplies_to")
        assert not decision.approved
        assert decision.deny_reason == DenyReason.PASSAGE_TOO_SHORT

    def test_exactly_at_threshold(self) -> None:
        gate = _make_gate(min_passage_length=80)
        decision = gate.evaluate("TSMC", passage_length=80, predicate="supplies_to")
        assert decision.approved

    def test_above_threshold(self) -> None:
        gate = _make_gate(min_passage_length=80)
        decision = gate.evaluate("TSMC", passage_length=200, predicate="supplies_to")
        assert decision.approved

    def test_zero_length_denied(self) -> None:
        gate = _make_gate(min_passage_length=1)
        decision = gate.evaluate("TSMC", passage_length=0)
        assert not decision.approved
        assert decision.deny_reason == DenyReason.PASSAGE_TOO_SHORT


# -- Predicate filter tests ------------------------------------------------


class TestPredicateFilter:
    """Gate only approves high-value predicates."""

    def test_high_value_predicate_approved(self) -> None:
        gate = _make_gate(high_value_predicates=["supplies_to", "competes_with"])
        decision = gate.evaluate("TSMC", passage_length=200, predicate="supplies_to")
        assert decision.approved

    def test_low_value_predicate_denied(self) -> None:
        gate = _make_gate(high_value_predicates=["supplies_to", "competes_with"])
        decision = gate.evaluate("TSMC", passage_length=200, predicate="mentions")
        assert not decision.approved
        assert decision.deny_reason == DenyReason.PREDICATE_NOT_HIGH_VALUE

    def test_unknown_predicate_allowed_when_list_empty(self) -> None:
        """Empty high_value_predicates means all predicates qualify."""
        gate = _make_gate(high_value_predicates=[])
        decision = gate.evaluate("TSMC", passage_length=200, predicate="random_pred")
        assert decision.approved

    def test_no_predicate_skips_check(self) -> None:
        """When predicate is None, the check is skipped."""
        gate = _make_gate(high_value_predicates=["supplies_to"])
        decision = gate.evaluate("TSMC", passage_length=200)
        assert decision.approved


# -- Budget cap tests ------------------------------------------------------


class TestBudgetCaps:
    """Gate enforces daily and per-run budget limits."""

    def test_daily_budget_exhausted(self) -> None:
        gate = _make_gate(daily_llm_budget=2)
        gate.record_invocation()
        gate.record_invocation()
        decision = gate.evaluate("TSMC", passage_length=200, predicate="supplies_to")
        assert not decision.approved
        assert decision.deny_reason == DenyReason.DAILY_BUDGET_EXHAUSTED

    def test_daily_budget_not_exhausted(self) -> None:
        gate = _make_gate(daily_llm_budget=5)
        gate.record_invocation()
        decision = gate.evaluate("TSMC", passage_length=200, predicate="supplies_to")
        assert decision.approved

    def test_per_run_budget_exhausted(self) -> None:
        gate = _make_gate(per_run_llm_budget=2)
        gate.record_invocation(run_id="run_1")
        gate.record_invocation(run_id="run_1")
        decision = gate.evaluate(
            "TSMC",
            passage_length=200,
            predicate="supplies_to",
            run_id="run_1",
        )
        assert not decision.approved
        assert decision.deny_reason == DenyReason.RUN_BUDGET_EXHAUSTED

    def test_per_run_budget_independent(self) -> None:
        """Different runs have independent budgets."""
        gate = _make_gate(per_run_llm_budget=1)
        gate.record_invocation(run_id="run_1")
        decision = gate.evaluate(
            "TSMC",
            passage_length=200,
            predicate="supplies_to",
            run_id="run_2",
        )
        assert decision.approved

    def test_zero_daily_budget_blocks_all(self) -> None:
        gate = _make_gate(daily_llm_budget=0)
        decision = gate.evaluate("TSMC", passage_length=200, predicate="supplies_to")
        assert not decision.approved
        assert decision.deny_reason == DenyReason.DAILY_BUDGET_EXHAUSTED


# -- Budget tracker tests --------------------------------------------------


class TestFallbackBudget:
    """Budget tracker accounting and date reset."""

    def test_initial_counts_zero(self) -> None:
        budget = FallbackBudget()
        assert budget.daily_count == 0
        assert budget.run_count("any") == 0

    def test_record_increments(self) -> None:
        budget = FallbackBudget()
        budget.record_invocation("run_1")
        assert budget.daily_count == 1
        assert budget.run_count("run_1") == 1

    def test_run_scoping(self) -> None:
        budget = FallbackBudget()
        budget.record_invocation("run_a")
        budget.record_invocation("run_a")
        budget.record_invocation("run_b")
        assert budget.run_count("run_a") == 2
        assert budget.run_count("run_b") == 1
        assert budget.daily_count == 3

    def test_none_run_id(self) -> None:
        """Recording with no run_id increments daily but not per-run."""
        budget = FallbackBudget()
        budget.record_invocation(None)
        assert budget.daily_count == 1
        # No per-run count should be tracked for None


# -- GateDecision tests ----------------------------------------------------


class TestGateDecision:
    """GateDecision dataclass properties."""

    def test_approved_property(self) -> None:
        d = GateDecision(verdict=GateVerdict.APPROVED, mention="TSMC")
        assert d.approved is True

    def test_denied_property(self) -> None:
        d = GateDecision(
            verdict=GateVerdict.DENIED,
            mention="TSMC",
            deny_reason=DenyReason.MASTER_DISABLED,
        )
        assert d.approved is False

    def test_metadata_fields(self) -> None:
        d = GateDecision(
            verdict=GateVerdict.APPROVED,
            mention="Samsung",
            passage_length=300,
            predicate="supplies_to",
            run_invocations=2,
            daily_invocations=10,
        )
        assert d.passage_length == 300
        assert d.predicate == "supplies_to"


# -- Provenance tests ------------------------------------------------------


class TestFallbackProvenance:
    """Provenance record construction and serialization."""

    def test_make_provenance_needs_review(self) -> None:
        gate = _make_gate(
            llm_proposed_confidence=0.45,
            auto_approve_threshold=0.85,
        )
        decision = gate.evaluate("TSMC", passage_length=200, predicate="supplies_to")
        prov = gate.make_provenance(
            decision,
            model_id="gpt-4o-mini",
            proposed_concept_id="concept_issuer_tsmc",
            proposed_name="Taiwan Semiconductor",
        )
        assert prov.needs_review is True
        assert prov.effective_confidence == 0.45
        assert "below auto-approve" in prov.review_reason

    def test_make_provenance_auto_approved(self) -> None:
        gate = _make_gate(
            llm_proposed_confidence=0.90,
            auto_approve_threshold=0.85,
        )
        decision = gate.evaluate("TSMC", passage_length=200, predicate="supplies_to")
        prov = gate.make_provenance(
            decision,
            model_id="gpt-4o-mini",
            llm_confidence=0.95,
        )
        assert prov.needs_review is False
        assert prov.effective_confidence == 0.90

    def test_llm_confidence_caps_effective(self) -> None:
        """Effective confidence is the min of policy and LLM-reported."""
        gate = _make_gate(llm_proposed_confidence=0.45)
        decision = gate.evaluate("TSMC", passage_length=200, predicate="supplies_to")
        prov = gate.make_provenance(decision, llm_confidence=0.30)
        assert prov.effective_confidence == 0.30

    def test_to_metadata_serialization(self) -> None:
        gate = _make_gate()
        decision = gate.evaluate("Samsung", passage_length=500, predicate="competes_with")
        prov = gate.make_provenance(
            decision,
            model_id="claude-sonnet",
            proposed_concept_id="concept_issuer_samsung",
            proposed_name="Samsung Electronics",
            metadata={"token_count": 150},
        )
        meta = prov.to_metadata()
        assert meta["llm_fallback"] is True
        assert meta["model_id"] == "claude-sonnet"
        assert meta["proposed_concept_id"] == "concept_issuer_samsung"
        assert meta["needs_review"] is True
        assert meta["gate_verdict"] == "approved"
        assert meta["passage_length"] == 500
        assert meta["token_count"] == 150
        assert "invoked_at" in meta


# -- Resolver integration tests -------------------------------------------


class TestResolverGateIntegration:
    """EntityResolver integration with FallbackGate."""

    @pytest.fixture()
    def repo(self):
        """Minimal mock concept repo (reuse from test_resolver)."""
        from tests.test_claims.test_resolver import _MockConceptRepo

        r = _MockConceptRepo()
        r.add_concept(
            "concept_issuer_nvda",
            "issuer",
            "NVIDIA Corporation",
            ticker="NVDA",
        )
        return r

    async def test_gate_disabled_returns_unresolved(self, repo) -> None:
        """Without a gate, unresolved mentions stay unresolved."""
        from src.claims.resolver import EntityResolver, ResolverTier

        resolver = EntityResolver(repo)
        result = await resolver.resolve("Unknown Entity XYZ")
        assert not result.resolved
        assert result.tier == ResolverTier.UNRESOLVED

    async def test_gate_denied_returns_unresolved_with_metadata(self, repo) -> None:
        """Gate denial produces unresolved result with denial info."""
        from src.claims.resolver import EntityResolver, ResolverTier

        gate = FallbackGate(_make_config(llm_fallback_enabled=False))
        resolver = EntityResolver(repo, fallback_gate=gate)
        result = await resolver.resolve("Unknown Entity XYZ")
        assert not result.resolved
        assert result.tier == ResolverTier.UNRESOLVED
        assert result.metadata.get("gate_verdict") == "denied"
        assert result.metadata.get("gate_deny_reason") == "master_disabled"

    async def test_gate_approved_returns_llm_proposed(self, repo) -> None:
        """Gate approval produces LLM_PROPOSED tier result."""
        from src.claims.resolver import EntityResolver, ResolverTier

        gate = FallbackGate(_make_config())
        resolver = EntityResolver(repo, fallback_gate=gate)
        result = await resolver.resolve(
            "Unknown Entity XYZ",
            passage_length=200,
            predicate="supplies_to",
        )
        assert not result.resolved  # concept is None until LLM fills it
        assert result.tier == ResolverTier.LLM_PROPOSED
        assert result.metadata["match_type"] == "llm_gate_approved"
        assert result.metadata["gate_verdict"] == "approved"

    async def test_deterministic_tiers_bypass_gate(self, repo) -> None:
        """Exact/alias/fuzzy matches skip the gate entirely."""
        from src.claims.resolver import EntityResolver, ResolverTier

        gate = FallbackGate(_make_config())
        resolver = EntityResolver(repo, fallback_gate=gate)
        result = await resolver.resolve("NVDA", passage_length=200, predicate="supplies_to")
        assert result.resolved
        assert result.tier == ResolverTier.EXACT
        assert result.concept_id == "concept_issuer_nvda"

    async def test_gate_passage_too_short_returns_unresolved(self, repo) -> None:
        """Short passage denied by gate produces unresolved."""
        from src.claims.resolver import EntityResolver

        gate = FallbackGate(_make_config(min_passage_length=100))
        resolver = EntityResolver(repo, fallback_gate=gate)
        result = await resolver.resolve(
            "Unknown Entity XYZ",
            passage_length=50,
            predicate="supplies_to",
        )
        assert not result.resolved
        assert result.metadata.get("gate_deny_reason") == "passage_too_short"
