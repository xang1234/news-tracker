"""Tests for the numeric reconciliation orchestrator.

The orchestrator is the live-pipeline glue that was missing: given a
resolved numeric claim it finds comparable facts, classifies
support/contradiction links, recomputes the assertion, and persists both —
so two contradicting numeric claims about the same (subject, metric, period)
yield a persisted ``disputed`` assertion.

Tested against in-memory fake repositories so the real classification +
aggregation + recompute logic runs end to end without a database.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.assertions.numeric_reconciler import NumericReconciler
from src.assertions.schemas import ResolvedAssertion
from src.claims.schemas import EvidenceClaim, make_claim_key


def _claim(
    claim_id: str,
    *,
    numeric_value: float | None,
    subject_concept_id: str | None = "concept_tsmc",
    metric: str | None = "capex",
    unit: str | None = "USD",
    period: str | None = "2026-Q3",
    predicate: str = "expands_capacity",
    confidence: float = 0.7,
    published: datetime | None = None,
) -> EvidenceClaim:
    key = make_claim_key("narrative", claim_id, "TSMC", predicate, "")
    return EvidenceClaim(
        claim_id=claim_id,
        claim_key=key,
        lane="narrative",
        source_id=claim_id,
        predicate=predicate,
        subject_text="TSMC",
        subject_concept_id=subject_concept_id,
        metric=metric,
        numeric_value=numeric_value,
        unit=unit,
        period=period,
        confidence=confidence,
        source_published_at=published,
    )


class FakeClaimRepo:
    """Returns a fixed comparable set, recording the lookup arguments."""

    def __init__(self, comparable: list[EvidenceClaim]) -> None:
        self._comparable = comparable
        self.calls: list[dict] = []

    async def list_comparable_numeric_claims(
        self, *, subject_concept_id: str, metric: str, period: str | None
    ) -> list[EvidenceClaim]:
        self.calls.append(
            {"subject_concept_id": subject_concept_id, "metric": metric, "period": period}
        )
        return list(self._comparable)


class FakeAssertionRepo:
    def __init__(self) -> None:
        self.assertions: dict[str, ResolvedAssertion] = {}
        self.links: list = []

    async def get_assertion(self, assertion_id: str) -> ResolvedAssertion | None:
        return self.assertions.get(assertion_id)

    async def upsert_assertion(self, assertion: ResolvedAssertion) -> ResolvedAssertion:
        self.assertions[assertion.assertion_id] = assertion
        return assertion

    async def upsert_link(self, link):
        self.links.append(link)
        return link


@pytest.mark.asyncio
async def test_contradicting_numeric_claims_persist_disputed_assertion():
    incoming = _claim("c1", numeric_value=42e9, confidence=0.9)
    other = _claim("c2", numeric_value=30e9, confidence=0.6)
    claim_repo = FakeClaimRepo([incoming, other])
    assertion_repo = FakeAssertionRepo()
    reconciler = NumericReconciler(claim_repo, assertion_repo)

    result = await reconciler.reconcile_claim(incoming)

    assert result is not None
    assert result.status == "disputed"
    assert result.contradiction_count >= 1
    # Assertion + both links persisted.
    assert result.assertion_id in assertion_repo.assertions
    assert len(assertion_repo.links) == 2
    # Lookup used the canonical (subject_concept_id, metric, period) key.
    assert claim_repo.calls == [
        {"subject_concept_id": "concept_tsmc", "metric": "capex", "period": "2026-Q3"}
    ]


@pytest.mark.asyncio
async def test_agreeing_numeric_claims_stay_active():
    incoming = _claim("c1", numeric_value=42e9)
    other = _claim("c2", numeric_value=43e9)
    assertion_repo = FakeAssertionRepo()
    reconciler = NumericReconciler(FakeClaimRepo([incoming, other]), assertion_repo)

    result = await reconciler.reconcile_claim(incoming)

    assert result is not None
    assert result.status == "active"
    assert result.contradiction_count == 0


@pytest.mark.asyncio
async def test_non_numeric_claim_is_skipped():
    relationship = _claim("c1", numeric_value=None, metric=None)
    assertion_repo = FakeAssertionRepo()
    reconciler = NumericReconciler(FakeClaimRepo([]), assertion_repo)

    result = await reconciler.reconcile_claim(relationship)

    assert result is None
    assert assertion_repo.assertions == {}
    assert assertion_repo.links == []


@pytest.mark.asyncio
async def test_unresolved_subject_is_skipped():
    unresolved = _claim("c1", numeric_value=42e9, subject_concept_id=None)
    assertion_repo = FakeAssertionRepo()
    reconciler = NumericReconciler(FakeClaimRepo([]), assertion_repo)

    result = await reconciler.reconcile_claim(unresolved)

    assert result is None
    assert assertion_repo.assertions == {}


@pytest.mark.asyncio
async def test_incoming_claim_included_even_if_repo_omits_it():
    # Repo returns only the older claim; reconciler must still fold in the
    # incoming one so the contradiction is detected.
    incoming = _claim("c1", numeric_value=42e9, confidence=0.9)
    older = _claim("c2", numeric_value=30e9, confidence=0.6)
    assertion_repo = FakeAssertionRepo()
    reconciler = NumericReconciler(FakeClaimRepo([older]), assertion_repo)

    result = await reconciler.reconcile_claim(incoming)

    assert result is not None
    assert result.status == "disputed"
    assert {link.claim_id for link in assertion_repo.links} == {"c1", "c2"}


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
async def test_resolve_numeric_subject_sets_concept_id():
    from src.assertions.numeric_reconciler import resolve_numeric_subject

    claim = _claim("c1", numeric_value=42e9, subject_concept_id=None)
    resolver = _FakeResolver("concept_tsmc")

    await resolve_numeric_subject(claim, resolver)

    assert claim.subject_concept_id == "concept_tsmc"
    assert resolver.calls == ["TSMC"]


@pytest.mark.asyncio
async def test_resolve_numeric_subject_skips_already_resolved():
    from src.assertions.numeric_reconciler import resolve_numeric_subject

    claim = _claim("c1", numeric_value=42e9, subject_concept_id="concept_existing")
    resolver = _FakeResolver("concept_other")

    await resolve_numeric_subject(claim, resolver)

    assert claim.subject_concept_id == "concept_existing"
    assert resolver.calls == []  # no resolve call when already grounded


@pytest.mark.asyncio
async def test_resolve_numeric_subject_unresolved_leaves_none():
    from src.assertions.numeric_reconciler import resolve_numeric_subject

    claim = _claim("c1", numeric_value=42e9, subject_concept_id=None)
    resolver = _FakeResolver(None)

    await resolve_numeric_subject(claim, resolver)

    assert claim.subject_concept_id is None


@pytest.mark.asyncio
async def test_resolve_numeric_subject_ignores_non_numeric():
    from src.assertions.numeric_reconciler import resolve_numeric_subject

    claim = _claim("c1", numeric_value=None, metric=None, subject_concept_id=None)
    resolver = _FakeResolver("concept_tsmc")

    await resolve_numeric_subject(claim, resolver)

    assert claim.subject_concept_id is None
    assert resolver.calls == []
