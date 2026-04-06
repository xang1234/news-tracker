"""Tests for assertion recomputation after review decisions.

Verifies that review outcomes and claim updates trigger coherent
recomputation of assertions and derived edges, with full audit trail.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.assertions.recompute import (
    AssertionDelta,
    build_recompute_result,
    find_affected_assertion_ids,
    recompute_assertion,
)
from src.assertions.schemas import (
    AssertionClaimLink,
    ResolvedAssertion,
    make_assertion_id,
)
from src.claims.schemas import EvidenceClaim

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Helpers ---------------------------------------------------------------


def _make_claim(
    claim_id: str = "claim_1",
    *,
    confidence: float = 0.8,
    status: str = "active",
    source_published_at: datetime | None = None,
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        claim_key=f"clk_{claim_id}",
        lane="narrative",
        source_id=f"src_{claim_id}",
        predicate="supplies_to",
        subject_text="TSMC",
        confidence=confidence,
        status=status,
        source_published_at=source_published_at or NOW,
        contract_version="0.1.0",
    )


def _make_link(
    claim_id: str = "claim_1",
    assertion_id: str = "asrt_test",
    link_type: str = "support",
    **metadata,
) -> AssertionClaimLink:
    return AssertionClaimLink(
        assertion_id=assertion_id,
        claim_id=claim_id,
        link_type=link_type,
        metadata=metadata,
    )


def _make_existing(
    confidence: float = 0.7,
    status: str = "active",
    **kwargs,
) -> ResolvedAssertion:
    return ResolvedAssertion(
        assertion_id=make_assertion_id("concept_tsmc", "supplies_to", "concept_nvda"),
        subject_concept_id="concept_tsmc",
        predicate="supplies_to",
        object_concept_id="concept_nvda",
        confidence=confidence,
        status=status,
        **kwargs,
    )


# -- recompute_assertion tests ---------------------------------------------


class TestRecomputeAssertion:
    """Single assertion recomputation."""

    def test_new_assertion(self) -> None:
        """No existing assertion — creates from scratch."""
        claim = _make_claim(confidence=0.8)
        link = _make_link()
        assertion, delta = recompute_assertion(
            None, [claim], [link],
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )
        assert assertion.confidence > 0
        assert delta.previous_confidence == 0.0
        assert delta.new_confidence == assertion.confidence
        assert delta.confidence_changed is True

    def test_unchanged_assertion(self) -> None:
        """Same claims → no change in confidence."""
        claim = _make_claim(confidence=0.8)
        link = _make_link()
        _make_existing()

        # First compute to get the actual confidence
        first, _ = recompute_assertion(
            None, [claim], [link],
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )

        # Recompute with same inputs against the first result
        _, delta = recompute_assertion(
            first, [claim], [link],
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )
        assert delta.confidence_changed is False
        assert delta.status_changed is False

    def test_confidence_increases_after_new_support(self) -> None:
        """Adding a supporting claim increases confidence."""
        claim1 = _make_claim("c1", confidence=0.6)
        link1 = _make_link("c1")
        existing = _make_existing(confidence=0.1, support_count=1)

        claim2 = _make_claim("c2", confidence=0.9)
        link2 = _make_link("c2")

        assertion, delta = recompute_assertion(
            existing, [claim1, claim2], [link1, link2],
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )
        assert delta.confidence_changed is True
        assert delta.new_confidence > delta.previous_confidence

    def test_retracted_claim_reduces_confidence(self) -> None:
        """Retracting a claim should reduce assertion confidence."""
        claim1 = _make_claim("c1", confidence=0.8)
        claim2 = _make_claim("c2", confidence=0.9, status="retracted")
        links = [_make_link("c1"), _make_link("c2")]

        existing = _make_existing(confidence=0.85, support_count=2)
        assertion, delta = recompute_assertion(
            existing, [claim1, claim2], links,
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )
        assert delta.confidence_changed is True
        assert delta.new_confidence < delta.previous_confidence

    def test_contradiction_triggers_disputed(self) -> None:
        """Adding contradiction should change status to disputed."""
        support = _make_claim("c1", confidence=0.6)
        contra1 = _make_claim("c2", confidence=0.7)
        contra2 = _make_claim("c3", confidence=0.5)
        links = [
            _make_link("c1", link_type="support"),
            _make_link("c2", link_type="contradiction"),
            _make_link("c3", link_type="contradiction"),
        ]
        existing = _make_existing(status="active")

        assertion, delta = recompute_assertion(
            existing, [support, contra1, contra2], links,
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )
        assert delta.status_changed is True
        assert delta.new_status == "disputed"

    def test_edge_transitions(self) -> None:
        """Edge state changes tracked in delta."""
        # Use high confidence + low edge threshold to ensure edge is created
        # (diversity=0.333 with 1 source type reduces final confidence)
        claim = _make_claim(confidence=0.95)
        link = _make_link()

        assertion, delta = recompute_assertion(
            None, [claim], [link],
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )
        assert delta.edge_before is None  # no existing assertion
        # Final ≈ 0.95 * 1.0 * 0.333 * 1.0 ≈ 0.317 > default threshold 0.3
        assert delta.edge_after is not None
        assert delta.edge_after.is_current is True

    def test_review_approved_boosts_confidence(self) -> None:
        """Review-approved link metadata increases confidence."""
        claim = _make_claim(confidence=0.7)
        link_approved = _make_link(review_approved=True)
        link_plain = _make_link()

        _, with_review = recompute_assertion(
            None, [claim], [link_approved],
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )
        _, without_review = recompute_assertion(
            None, [claim], [link_plain],
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )
        assert with_review.new_confidence > without_review.new_confidence

    def test_breakdown_included(self) -> None:
        claim = _make_claim(confidence=0.8)
        link = _make_link()
        _, delta = recompute_assertion(
            None, [claim], [link],
            subject_concept_id="concept_tsmc",
            predicate="supplies_to",
            object_concept_id="concept_nvda",
            now=NOW,
        )
        assert delta.breakdown is not None
        assert delta.breakdown.base > 0


# -- build_recompute_result tests ------------------------------------------


class TestBuildRecomputeResult:
    """RecomputeResult construction from deltas."""

    def test_no_changes(self) -> None:
        delta = AssertionDelta(
            assertion_id="asrt_1",
            previous_confidence=0.8,
            new_confidence=0.8,
            previous_status="active",
            new_status="active",
        )
        result = build_recompute_result(
            "review_resolved", {"review_task_id": "rt_1"}, [delta]
        )
        assert result.assertions_updated == 0
        assert result.had_changes is False

    def test_confidence_change_counted(self) -> None:
        delta = AssertionDelta(
            assertion_id="asrt_1",
            previous_confidence=0.5,
            new_confidence=0.8,
            previous_status="active",
            new_status="active",
            confidence_changed=True,
        )
        result = build_recompute_result("claim_update", {}, [delta])
        assert result.assertions_updated == 1
        assert result.had_changes is True

    def test_status_change_counted(self) -> None:
        delta = AssertionDelta(
            assertion_id="asrt_1",
            previous_confidence=0.5,
            new_confidence=0.5,
            previous_status="active",
            new_status="disputed",
            status_changed=True,
        )
        result = build_recompute_result("claim_update", {}, [delta])
        assert result.assertions_updated == 1

    def test_edge_added(self) -> None:
        from src.assertions.edges import DerivedEdge

        edge = DerivedEdge(
            source_concept_id="c_a", target_concept_id="c_b",
            predicate="supplies_to", confidence=0.8,
            assertion_id="asrt_1", is_current=True,
        )
        delta = AssertionDelta(
            assertion_id="asrt_1",
            previous_confidence=0.0, new_confidence=0.8,
            previous_status="active", new_status="active",
            edge_before=None, edge_after=edge,
        )
        result = build_recompute_result("new_claim", {}, [delta])
        assert result.edges_added == 1
        assert result.edges_removed == 0

    def test_edge_removed(self) -> None:
        from src.assertions.edges import DerivedEdge

        edge = DerivedEdge(
            source_concept_id="c_a", target_concept_id="c_b",
            predicate="supplies_to", confidence=0.8,
            assertion_id="asrt_1", is_current=True,
        )
        delta = AssertionDelta(
            assertion_id="asrt_1",
            previous_confidence=0.8, new_confidence=0.1,
            previous_status="active", new_status="active",
            edge_before=edge, edge_after=None,
        )
        result = build_recompute_result("claim_retracted", {}, [delta])
        assert result.edges_removed == 1
        assert result.edges_added == 0

    def test_trigger_detail_preserved(self) -> None:
        result = build_recompute_result(
            "review_resolved",
            {"review_task_id": "rt_123", "resolution": "approved"},
            [],
        )
        assert result.trigger == "review_resolved"
        assert result.trigger_detail["review_task_id"] == "rt_123"

    def test_recomputed_at_set(self) -> None:
        result = build_recompute_result("test", {}, [])
        assert result.recomputed_at is not None


# -- find_affected_assertion_ids tests -------------------------------------


class TestFindAffectedAssertionIds:
    """Finding assertions affected by claim changes."""

    def test_finds_affected(self) -> None:
        links = [
            _make_link("c1", assertion_id="asrt_1"),
            _make_link("c2", assertion_id="asrt_1"),
            _make_link("c3", assertion_id="asrt_2"),
        ]
        affected = find_affected_assertion_ids(["c1"], links)
        assert affected == ["asrt_1"]

    def test_multiple_claims_multiple_assertions(self) -> None:
        links = [
            _make_link("c1", assertion_id="asrt_1"),
            _make_link("c2", assertion_id="asrt_2"),
            _make_link("c3", assertion_id="asrt_3"),
        ]
        affected = find_affected_assertion_ids(["c1", "c2"], links)
        assert affected == ["asrt_1", "asrt_2"]

    def test_no_matches(self) -> None:
        links = [
            _make_link("c1", assertion_id="asrt_1"),
        ]
        affected = find_affected_assertion_ids(["c999"], links)
        assert affected == []

    def test_deduplicates(self) -> None:
        links = [
            _make_link("c1", assertion_id="asrt_1"),
            _make_link("c2", assertion_id="asrt_1"),
        ]
        affected = find_affected_assertion_ids(["c1", "c2"], links)
        assert affected == ["asrt_1"]  # deduplicated

    def test_sorted_output(self) -> None:
        links = [
            _make_link("c1", assertion_id="asrt_b"),
            _make_link("c2", assertion_id="asrt_a"),
        ]
        affected = find_affected_assertion_ids(["c1", "c2"], links)
        assert affected == ["asrt_a", "asrt_b"]
