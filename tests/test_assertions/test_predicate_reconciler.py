"""Tests for the predicate-polarity reconciliation orchestrator.

Given a resolved claim whose predicate has an antonym, finds same-subject
antonym claims with overlapping validity, classifies support/contradiction,
recomputes the assertion, and persists — so an ``expands_capacity`` claim
and a ``constrains_capacity`` claim on the same subject yield a persisted
``disputed`` assertion.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.assertions.predicate_reconciler import PredicateContradictionReconciler
from src.assertions.schemas import ResolvedAssertion
from src.claims.schemas import EvidenceClaim, make_claim_key


def _claim(
    claim_id: str,
    predicate: str,
    *,
    subject_concept_id: str | None = "concept_tsmc",
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
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
        claim_valid_from=valid_from,
        claim_valid_to=valid_to,
        confidence=0.7,
    )


class FakeClaimRepo:
    def __init__(self, comparable: list[EvidenceClaim]) -> None:
        self._comparable = comparable
        self.calls: list[dict] = []

    async def list_claims_by_subject_predicates(
        self, *, subject_concept_id: str, predicates: list[str]
    ) -> list[EvidenceClaim]:
        self.calls.append({"subject_concept_id": subject_concept_id, "predicates": predicates})
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
async def test_antonym_claims_persist_disputed_assertion():
    incoming = _claim("c1", "expands_capacity")
    opposite = _claim("c2", "constrains_capacity")
    claim_repo = FakeClaimRepo([incoming, opposite])
    assertion_repo = FakeAssertionRepo()
    reconciler = PredicateContradictionReconciler(claim_repo, assertion_repo)

    result = await reconciler.reconcile_claim(incoming)

    assert result is not None
    assert result.status == "disputed"
    assert result.contradiction_count >= 1
    assert len(assertion_repo.links) == 2
    # Lookup asked for both polarity sides.
    assert set(claim_repo.calls[0]["predicates"]) == {
        "expands_capacity",
        "constrains_capacity",
    }


@pytest.mark.asyncio
async def test_same_predicate_only_stays_active():
    incoming = _claim("c1", "expands_capacity")
    agreeing = _claim("c2", "expands_capacity")
    assertion_repo = FakeAssertionRepo()
    reconciler = PredicateContradictionReconciler(
        FakeClaimRepo([incoming, agreeing]), assertion_repo
    )

    result = await reconciler.reconcile_claim(incoming)

    assert result is not None
    assert result.status == "active"
    assert result.contradiction_count == 0


@pytest.mark.asyncio
async def test_non_overlapping_antonym_does_not_contradict():
    incoming = _claim(
        "c1",
        "expands_capacity",
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_to=datetime(2026, 3, 1, tzinfo=UTC),
    )
    later = _claim(
        "c2",
        "constrains_capacity",
        valid_from=datetime(2026, 6, 1, tzinfo=UTC),
        valid_to=datetime(2026, 9, 1, tzinfo=UTC),
    )
    assertion_repo = FakeAssertionRepo()
    reconciler = PredicateContradictionReconciler(FakeClaimRepo([incoming, later]), assertion_repo)

    result = await reconciler.reconcile_claim(incoming)

    assert result is not None
    assert result.status == "active"
    assert result.contradiction_count == 0


@pytest.mark.asyncio
async def test_predicate_without_antonym_is_skipped():
    neutral = _claim("c1", "changes_pricing")
    assertion_repo = FakeAssertionRepo()
    reconciler = PredicateContradictionReconciler(FakeClaimRepo([]), assertion_repo)

    result = await reconciler.reconcile_claim(neutral)

    assert result is None
    assert assertion_repo.assertions == {}


@pytest.mark.asyncio
async def test_unresolved_subject_is_skipped():
    unresolved = _claim("c1", "expands_capacity", subject_concept_id=None)
    assertion_repo = FakeAssertionRepo()
    reconciler = PredicateContradictionReconciler(FakeClaimRepo([]), assertion_repo)

    result = await reconciler.reconcile_claim(unresolved)

    assert result is None
    assert assertion_repo.assertions == {}
