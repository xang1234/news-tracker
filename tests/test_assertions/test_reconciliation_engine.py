"""Tests for the claim reconciliation engine.

A single engine reconciles the incoming claim's assertion by gathering
candidate claims from every applicable tier, collecting each tier's
support/contradiction *opinion* per claim, merging them contradiction-dominant,
and recomputing + persisting the assertion exactly once. This replaces the old
per-tier reconcilers that each independently persisted (and clobbered) the same
assertion.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.assertions.reconciliation_engine import (
    ClaimReconciliationEngine,
    CorroborationTier,
    NumericTier,
    PredicateContradictionTier,
    merge_link_types,
)
from src.assertions.schemas import ResolvedAssertion
from src.claims.schemas import EvidenceClaim, make_claim_key


def _claim(
    claim_id: str,
    predicate: str,
    *,
    subject_concept_id: str | None = "concept_tsmc",
    object_concept_id: str | None = None,
    metric: str | None = None,
    numeric_value: float | None = None,
    unit: str | None = None,
    period: str | None = None,
    source_type: str = "document",
    confidence: float = 0.7,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> EvidenceClaim:
    key = make_claim_key("narrative", claim_id, "TSMC", predicate, object_concept_id or "")
    return EvidenceClaim(
        claim_id=claim_id,
        claim_key=key,
        lane="narrative",
        source_id=claim_id,
        source_type=source_type,
        predicate=predicate,
        subject_text="TSMC",
        subject_concept_id=subject_concept_id,
        object_concept_id=object_concept_id,
        metric=metric,
        numeric_value=numeric_value,
        unit=unit,
        period=period,
        confidence=confidence,
        claim_valid_from=valid_from,
        claim_valid_to=valid_to,
    )


class FakeClaimRepo:
    """Realistic in-memory repo filtering a fixed claim set."""

    def __init__(self, claims: list[EvidenceClaim]) -> None:
        self._claims = claims

    async def list_comparable_numeric_claims(self, *, subject_concept_id, metric, period):
        return [
            c
            for c in self._claims
            if c.subject_concept_id == subject_concept_id
            and c.metric == metric
            and c.period == period
            and c.numeric_value is not None
            and c.status == "active"
        ]

    async def list_claims_by_subject_predicates(self, *, subject_concept_id, predicates):
        return [
            c
            for c in self._claims
            if c.subject_concept_id == subject_concept_id
            and c.predicate in predicates
            and c.status == "active"
        ]


class FakeAssertionRepo:
    def __init__(self) -> None:
        self.assertions: dict[str, ResolvedAssertion] = {}
        self.links: list = []

    async def get_assertion(self, assertion_id):
        return self.assertions.get(assertion_id)

    async def upsert_assertion(self, assertion):
        self.assertions[assertion.assertion_id] = assertion
        return assertion

    async def upsert_link(self, link):
        self.links.append(link)
        return link


def _engine(claims: list[EvidenceClaim]) -> tuple[ClaimReconciliationEngine, FakeAssertionRepo]:
    repo = FakeClaimRepo(claims)
    arepo = FakeAssertionRepo()
    engine = ClaimReconciliationEngine(
        repo,
        arepo,
        tiers=[NumericTier(), PredicateContradictionTier(), CorroborationTier()],
    )
    return engine, arepo


class TestMergeLinkTypes:
    def test_contradiction_dominates_support(self):
        merged = merge_link_types([{"a": "support"}, {"a": "contradiction"}, {"a": "support"}])
        assert merged["a"] == "contradiction"

    def test_support_when_no_contradiction(self):
        merged = merge_link_types([{"a": "support"}, {"b": "support"}])
        assert merged == {"a": "support", "b": "support"}

    def test_union_of_claim_ids(self):
        merged = merge_link_types([{"a": "support"}, {"b": "contradiction"}])
        assert set(merged) == {"a", "b"}


@pytest.mark.asyncio
async def test_numeric_divergence_disputes():
    c1 = _claim(
        "c1", "expands_capacity", metric="capacity", numeric_value=30.0, unit="%", period="2026-Q3"
    )
    c2 = _claim(
        "c2", "expands_capacity", metric="capacity", numeric_value=50.0, unit="%", period="2026-Q3"
    )
    engine, arepo = _engine([c1, c2])

    result = await engine.reconcile_claim(c1)

    assert result is not None
    assert result.status == "disputed"


@pytest.mark.asyncio
async def test_numeric_contradiction_not_clobbered_by_predicate_tier():
    # REGRESSION: a claim matching BOTH numeric and predicate tiers must stay
    # disputed — the predicate tier's support view must not overwrite it.
    c1 = _claim(
        "c1", "expands_capacity", metric="capacity", numeric_value=30.0, unit="%", period="2026-Q3"
    )
    c2 = _claim(
        "c2", "expands_capacity", metric="capacity", numeric_value=50.0, unit="%", period="2026-Q3"
    )
    engine, arepo = _engine([c1, c2])

    result = await engine.reconcile_claim(c1)

    assert result.status == "disputed"
    assert result.contradiction_count >= 1


@pytest.mark.asyncio
async def test_antonym_predicates_dispute():
    c1 = _claim("c1", "expands_capacity")
    c2 = _claim("c2", "constrains_capacity")
    engine, arepo = _engine([c1, c2])

    result = await engine.reconcile_claim(c1)

    assert result.status == "disputed"
    assert result.contradiction_count >= 1


@pytest.mark.asyncio
async def test_corroboration_aggregates_support_across_sources():
    # Same relationship triple from two different source documents/lanes.
    c1 = _claim("c1", "supplies_to", object_concept_id="concept_nvda", source_type="document")
    c2 = _claim("c2", "supplies_to", object_concept_id="concept_nvda", source_type="filing_section")
    engine, arepo = _engine([c1, c2])

    result = await engine.reconcile_claim(c1)

    assert result is not None
    assert result.status == "active"
    assert result.support_count == 2
    assert result.source_diversity == 2  # two distinct source types corroborate


@pytest.mark.asyncio
async def test_different_object_does_not_corroborate():
    c1 = _claim("c1", "supplies_to", object_concept_id="concept_nvda")
    c2 = _claim("c2", "supplies_to", object_concept_id="concept_amd")
    engine, arepo = _engine([c1, c2])

    result = await engine.reconcile_claim(c1)

    assert result.support_count == 1  # only the incoming claim's triple


@pytest.mark.asyncio
async def test_unresolved_subject_is_skipped():
    c1 = _claim("c1", "expands_capacity", subject_concept_id=None)
    engine, arepo = _engine([c1])

    result = await engine.reconcile_claim(c1)

    assert result is None
    assert arepo.assertions == {}


@pytest.mark.asyncio
async def test_persists_exactly_one_assertion():
    c1 = _claim(
        "c1", "expands_capacity", metric="capacity", numeric_value=30.0, unit="%", period="2026-Q3"
    )
    c2 = _claim(
        "c2",
        "constrains_capacity",
        metric="capacity",
        numeric_value=50.0,
        unit="%",
        period="2026-Q3",
    )
    engine, arepo = _engine([c1, c2])

    await engine.reconcile_claim(c1)

    assert len(arepo.assertions) == 1


@pytest.mark.asyncio
async def test_semantic_tier_disputes_via_engine():
    # Two non-numeric, no-antonym claims the deterministic tiers can't judge;
    # the semantic tier (fake judge) flags them contradictory end-to-end.
    from src.assertions.reconciliation_engine import SemanticTier
    from src.assertions.semantic_judge import ContradictionVerdict

    class _FakeJudge:
        async def judge(self, a, b):
            return ContradictionVerdict("contradicts", 0.9)

    c1 = _claim("c1", "changes_pricing")
    c2 = _claim("c2", "changes_pricing")
    repo = FakeClaimRepo([c1, c2])
    arepo = FakeAssertionRepo()
    engine = ClaimReconciliationEngine(
        repo, arepo, tiers=[CorroborationTier(), SemanticTier(_FakeJudge())]
    )

    result = await engine.reconcile_claim(c1)

    assert result is not None
    assert result.status == "disputed"
    assert result.contradiction_count >= 1
