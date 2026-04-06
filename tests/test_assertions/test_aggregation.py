"""Tests for assertion aggregation logic.

Verifies that claims are aggregated into assertions with explainable
confidence decomposition. Tests cover support/contradiction, freshness
decay, source diversity, review bonuses, and edge cases.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.assertions.aggregation import (
    DEFAULT_REVIEW_BONUS,
    ConfidenceBreakdown,
    aggregate_assertion,
)
from src.assertions.schemas import AssertionClaimLink, ResolvedAssertion
from src.claims.schemas import EvidenceClaim

# -- Helpers ---------------------------------------------------------------

NOW = datetime(2026, 4, 1, tzinfo=UTC)


def _make_claim(
    claim_id: str = "claim_1",
    *,
    confidence: float = 0.8,
    source_type: str = "document",
    source_published_at: datetime | None = None,
    status: str = "active",
    claim_valid_from: datetime | None = None,
    claim_valid_to: datetime | None = None,
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        claim_key=f"clk_{claim_id}",
        lane="narrative",
        source_id=f"src_{claim_id}",
        predicate="supplies_to",
        subject_text="TSMC",
        confidence=confidence,
        source_type=source_type,
        source_published_at=source_published_at,
        status=status,
        claim_valid_from=claim_valid_from,
        claim_valid_to=claim_valid_to,
        contract_version="0.1.0",
    )


def _make_link(
    claim_id: str = "claim_1",
    link_type: str = "support",
    weight: float = 1.0,
    **metadata,
) -> AssertionClaimLink:
    return AssertionClaimLink(
        assertion_id="asrt_test",
        claim_id=claim_id,
        link_type=link_type,
        contribution_weight=weight,
        metadata=metadata,
    )


def _aggregate(
    claims: list[EvidenceClaim],
    links: list[AssertionClaimLink],
    **kwargs,
) -> tuple[ResolvedAssertion, ConfidenceBreakdown]:
    return aggregate_assertion(
        "concept_tsmc",
        "supplies_to",
        "concept_nvda",
        claims,
        links,
        now=NOW,
        **kwargs,
    )


# -- Basic aggregation tests -----------------------------------------------


class TestBasicAggregation:
    """Single-claim and basic multi-claim aggregation."""

    def test_single_support_claim(self) -> None:
        claim = _make_claim(confidence=0.8, source_published_at=NOW)
        link = _make_link()
        assertion, breakdown = _aggregate([claim], [link])

        assert assertion.support_count == 1
        assert assertion.contradiction_count == 0
        assert assertion.status == "active"
        assert breakdown.base == 0.8
        assert breakdown.support_ratio == 1.0
        assert breakdown.final > 0

    def test_multiple_support_claims(self) -> None:
        claims = [
            _make_claim("c1", confidence=0.7, source_published_at=NOW),
            _make_claim("c2", confidence=0.9, source_published_at=NOW),
        ]
        links = [_make_link("c1"), _make_link("c2")]
        assertion, breakdown = _aggregate(claims, links)

        assert assertion.support_count == 2
        assert breakdown.base == 0.8  # avg of 0.7 and 0.9

    def test_no_claims(self) -> None:
        assertion, breakdown = _aggregate([], [])
        assert assertion.confidence == 0.0
        assert assertion.support_count == 0
        assert assertion.status == "active"

    def test_assertion_id_is_deterministic(self) -> None:
        claim = _make_claim(source_published_at=NOW)
        link = _make_link()
        a1, _ = _aggregate([claim], [link])
        a2, _ = _aggregate([claim], [link])
        assert a1.assertion_id == a2.assertion_id

    def test_retracted_claims_excluded(self) -> None:
        claims = [
            _make_claim("c1", confidence=0.8, source_published_at=NOW),
            _make_claim("c2", confidence=0.9, status="retracted",
                        source_published_at=NOW),
        ]
        links = [_make_link("c1"), _make_link("c2")]
        assertion, breakdown = _aggregate(claims, links)
        assert assertion.support_count == 1
        assert breakdown.base == 0.8  # only c1 counted


# -- Contradiction tests ---------------------------------------------------


class TestContradiction:
    """Support vs contradiction handling."""

    def test_contradiction_reduces_support_ratio(self) -> None:
        claims = [
            _make_claim("c1", confidence=0.8, source_published_at=NOW),
            _make_claim("c2", confidence=0.7, source_published_at=NOW),
        ]
        links = [
            _make_link("c1", "support"),
            _make_link("c2", "contradiction"),
        ]
        assertion, breakdown = _aggregate(claims, links)
        assert assertion.support_count == 1
        assert assertion.contradiction_count == 1
        assert breakdown.support_ratio == 0.5

    def test_majority_contradiction_triggers_disputed(self) -> None:
        claims = [
            _make_claim("c1", confidence=0.8, source_published_at=NOW),
            _make_claim("c2", confidence=0.7, source_published_at=NOW),
            _make_claim("c3", confidence=0.6, source_published_at=NOW),
        ]
        links = [
            _make_link("c1", "support"),
            _make_link("c2", "contradiction"),
            _make_link("c3", "contradiction"),
        ]
        assertion, _ = _aggregate(claims, links)
        # support_ratio = 1/3 ≈ 0.33 < 0.6 → disputed
        assert assertion.status == "disputed"

    def test_minority_contradiction_stays_active(self) -> None:
        claims = [
            _make_claim("c1", confidence=0.8, source_published_at=NOW),
            _make_claim("c2", confidence=0.9, source_published_at=NOW),
            _make_claim("c3", confidence=0.5, source_published_at=NOW),
        ]
        links = [
            _make_link("c1", "support"),
            _make_link("c2", "support"),
            _make_link("c3", "contradiction"),
        ]
        assertion, _ = _aggregate(claims, links)
        # support_ratio = 2/3 ≈ 0.67 >= 0.6 → active
        assert assertion.status == "active"

    def test_is_disputed_property(self) -> None:
        claims = [
            _make_claim("c1", confidence=0.8, source_published_at=NOW),
            _make_claim("c2", confidence=0.7, source_published_at=NOW),
        ]
        links = [
            _make_link("c1", "support"),
            _make_link("c2", "contradiction"),
        ]
        assertion, _ = _aggregate(claims, links)
        assert assertion.is_disputed is True


# -- Freshness tests -------------------------------------------------------


class TestFreshness:
    """Time decay on evidence recency."""

    def test_recent_evidence_high_freshness(self) -> None:
        claim = _make_claim(source_published_at=NOW)
        link = _make_link()
        _, breakdown = _aggregate([claim], [link])
        assert breakdown.freshness >= 0.99

    def test_old_evidence_decayed_freshness(self) -> None:
        old = NOW - timedelta(days=90)
        claim = _make_claim(source_published_at=old)
        link = _make_link()
        _, breakdown = _aggregate([claim], [link])
        # exp(-0.01 * 90) ≈ 0.407
        assert 0.3 < breakdown.freshness < 0.5

    def test_no_timestamps_moderate_freshness(self) -> None:
        claim = _make_claim(source_published_at=None)
        link = _make_link()
        _, breakdown = _aggregate([claim], [link])
        assert breakdown.freshness == 0.5

    def test_freshness_uses_most_recent(self) -> None:
        old = NOW - timedelta(days=365)
        claims = [
            _make_claim("c1", source_published_at=old),
            _make_claim("c2", source_published_at=NOW),
        ]
        links = [_make_link("c1"), _make_link("c2")]
        _, breakdown = _aggregate(claims, links)
        # Most recent is NOW → freshness ≈ 1.0
        assert breakdown.freshness >= 0.99

    def test_custom_decay_rate(self) -> None:
        old = NOW - timedelta(days=30)
        claim = _make_claim(source_published_at=old)
        link = _make_link()
        _, slow = _aggregate([claim], [link], freshness_decay=0.001)
        _, fast = _aggregate([claim], [link], freshness_decay=0.1)
        assert slow.freshness > fast.freshness


# -- Source diversity tests -------------------------------------------------


class TestSourceDiversity:
    """Distinct source type counting."""

    def test_single_source_type(self) -> None:
        claim = _make_claim(source_type="document", source_published_at=NOW)
        link = _make_link()
        assertion, breakdown = _aggregate([claim], [link])
        assert assertion.source_diversity == 1
        # 1 / 3 (default target) ≈ 0.333
        assert 0.3 < breakdown.diversity < 0.4

    def test_meets_diversity_target(self) -> None:
        claims = [
            _make_claim("c1", source_type="document", source_published_at=NOW),
            _make_claim("c2", source_type="filing_section", source_published_at=NOW),
            _make_claim("c3", source_type="graph_edge", source_published_at=NOW),
        ]
        links = [_make_link("c1"), _make_link("c2"), _make_link("c3")]
        assertion, breakdown = _aggregate(claims, links)
        assert assertion.source_diversity == 3
        assert breakdown.diversity == 1.0

    def test_exceeds_diversity_target(self) -> None:
        """Diversity caps at 1.0 even with more types than target."""
        claims = [
            _make_claim("c1", source_type="document", source_published_at=NOW),
            _make_claim("c2", source_type="filing_section", source_published_at=NOW),
            _make_claim("c3", source_type="graph_edge", source_published_at=NOW),
            _make_claim("c4", source_type="manual", source_published_at=NOW),
        ]
        links = [_make_link(f"c{i}") for i in range(1, 5)]
        _, breakdown = _aggregate(claims, links)
        assert breakdown.diversity == 1.0

    def test_custom_diversity_target(self) -> None:
        claim = _make_claim(source_published_at=NOW)
        link = _make_link()
        _, breakdown = _aggregate(
            [claim], [link], diversity_target=1
        )
        assert breakdown.diversity == 1.0


# -- Review bonus tests ----------------------------------------------------


class TestReviewBonus:
    """Review-approved evidence adds a confidence bonus."""

    def test_review_approved_adds_bonus(self) -> None:
        claim = _make_claim(confidence=0.8, source_published_at=NOW)
        link = _make_link(review_approved=True)
        _, with_review = _aggregate([claim], [link])

        link_no = _make_link()
        _, without_review = _aggregate([claim], [link_no])

        assert with_review.review_bonus == DEFAULT_REVIEW_BONUS
        assert without_review.review_bonus == 0.0
        assert with_review.final > without_review.final

    def test_custom_review_bonus(self) -> None:
        claim = _make_claim(confidence=0.5, source_published_at=NOW)
        link = _make_link(review_approved=True)
        _, breakdown = _aggregate(
            [claim], [link], review_bonus=0.2
        )
        assert breakdown.review_bonus == 0.2


# -- Contribution weight tests ---------------------------------------------


class TestContributionWeight:
    """Claim weights affect the base confidence calculation."""

    def test_higher_weight_dominates(self) -> None:
        claims = [
            _make_claim("c1", confidence=0.9, source_published_at=NOW),
            _make_claim("c2", confidence=0.3, source_published_at=NOW),
        ]
        links = [
            _make_link("c1", weight=3.0 / 4.0),  # 0.75
            _make_link("c2", weight=1.0 / 4.0),  # 0.25
        ]
        _, breakdown = _aggregate(claims, links)
        # Weighted: (0.9*0.75 + 0.3*0.25) / (0.75 + 0.25) = 0.75
        assert abs(breakdown.base - 0.75) < 0.01


# -- Validity window tests -------------------------------------------------


class TestValidityWindow:
    """Assertion validity derived from supporting claims."""

    def test_derives_from_claims(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 1, tzinfo=UTC)
        t3 = datetime(2025, 12, 31, tzinfo=UTC)
        claims = [
            _make_claim("c1", source_published_at=NOW,
                        claim_valid_from=t1, claim_valid_to=t2),
            _make_claim("c2", source_published_at=NOW,
                        claim_valid_from=t2, claim_valid_to=t3),
        ]
        links = [_make_link("c1"), _make_link("c2")]
        assertion, _ = _aggregate(claims, links)
        assert assertion.valid_from == t1  # earliest
        assert assertion.valid_to == t3  # latest

    def test_no_validity_from_claims(self) -> None:
        claim = _make_claim(source_published_at=NOW)
        link = _make_link()
        assertion, _ = _aggregate([claim], [link])
        assert assertion.valid_from is None
        assert assertion.valid_to is None


# -- Metadata and breakdown tests ------------------------------------------


class TestMetadata:
    """Confidence breakdown stored in assertion metadata."""

    def test_breakdown_in_metadata(self) -> None:
        claim = _make_claim(confidence=0.8, source_published_at=NOW)
        link = _make_link()
        assertion, breakdown = _aggregate([claim], [link])
        meta_bd = assertion.metadata["breakdown"]
        assert meta_bd["base"] == breakdown.base
        assert meta_bd["freshness"] == breakdown.freshness
        assert meta_bd["diversity"] == breakdown.diversity

    def test_confidence_clamped_to_unit(self) -> None:
        """Even with high base + bonus, final is clamped to 1.0."""
        claim = _make_claim(confidence=1.0, source_published_at=NOW)
        link = _make_link(review_approved=True)
        _, breakdown = _aggregate(
            [claim], [link],
            review_bonus=0.5,
            diversity_target=1,
        )
        assert breakdown.final <= 1.0

    def test_confidence_never_negative(self) -> None:
        """No claims → 0.0, never negative."""
        assertion, breakdown = _aggregate([], [])
        assert breakdown.final >= 0.0
        assert assertion.confidence >= 0.0


# -- Timestamp tracking tests ----------------------------------------------


class TestTimestamps:
    """First-seen and last-evidence tracking."""

    def test_first_seen_is_earliest(self) -> None:
        t1 = NOW - timedelta(days=30)
        t2 = NOW - timedelta(days=5)
        claims = [
            _make_claim("c1", source_published_at=t1),
            _make_claim("c2", source_published_at=t2),
        ]
        links = [_make_link("c1"), _make_link("c2")]
        assertion, _ = _aggregate(claims, links)
        assert assertion.first_seen_at == t1
        assert assertion.last_evidence_at == t2

    def test_no_timestamps(self) -> None:
        claim = _make_claim(source_published_at=None)
        link = _make_link()
        assertion, _ = _aggregate([claim], [link])
        assert assertion.first_seen_at is None
        assert assertion.last_evidence_at is None
