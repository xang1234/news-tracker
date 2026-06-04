"""Tests for shared claim-reconciliation helpers.

Subject resolution is tier-agnostic: every reconciliation tier (numeric,
predicate-polarity, future corroboration/semantic) needs the claim's subject
grounded to a concept ID, so resolution is hoisted out of any single tier.
"""

from __future__ import annotations

import pytest

from src.assertions.claim_reconciliation import resolve_claim_subject
from src.claims.schemas import EvidenceClaim, make_claim_key


def _claim(subject_concept_id: str | None, *, predicate: str = "expands_capacity") -> EvidenceClaim:
    key = make_claim_key("narrative", "c1", "TSMC", predicate, "")
    return EvidenceClaim(
        claim_id="c1",
        claim_key=key,
        lane="narrative",
        source_id="c1",
        predicate=predicate,
        subject_text="TSMC",
        subject_concept_id=subject_concept_id,
    )


class _FakeResolverResult:
    def __init__(self, concept_id: str | None) -> None:
        self.concept_id = concept_id

    @property
    def resolved(self) -> bool:
        return self.concept_id is not None


class _FakeResolver:
    def __init__(self, concept_id: str | None) -> None:
        self._concept_id = concept_id
        self.calls: list[str] = []

    async def resolve(self, mention: str, **kwargs):
        self.calls.append(mention)
        return _FakeResolverResult(self._concept_id)


@pytest.mark.asyncio
async def test_resolves_when_subject_missing():
    claim = _claim(None)
    resolver = _FakeResolver("concept_tsmc")
    await resolve_claim_subject(claim, resolver)
    assert claim.subject_concept_id == "concept_tsmc"
    assert resolver.calls == ["TSMC"]


@pytest.mark.asyncio
async def test_skips_when_already_resolved():
    claim = _claim("concept_existing")
    resolver = _FakeResolver("concept_other")
    await resolve_claim_subject(claim, resolver)
    assert claim.subject_concept_id == "concept_existing"
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_leaves_none_when_unresolved():
    claim = _claim(None)
    resolver = _FakeResolver(None)
    await resolve_claim_subject(claim, resolver)
    assert claim.subject_concept_id is None


@pytest.mark.asyncio
async def test_resolves_non_numeric_claims_too():
    # A relationship/polarity claim with no numeric value still gets grounded.
    claim = _claim(None, predicate="constrains_capacity")
    resolver = _FakeResolver("concept_tsmc")
    await resolve_claim_subject(claim, resolver)
    assert claim.subject_concept_id == "concept_tsmc"


class _ThrowingResolver:
    async def resolve(self, mention: str, **kwargs):
        raise RuntimeError("resolver unavailable")


@pytest.mark.asyncio
async def test_resolver_failure_is_best_effort():
    # A transient resolver error must NOT propagate (which would drop the
    # claim before persistence); leave the subject unresolved and continue.
    claim = _claim(None)
    result = await resolve_claim_subject(claim, _ThrowingResolver())
    assert result.subject_concept_id is None
