"""Tests for resolved assertion schemas, claim links, and migration.

Tests cover the assertion data model, deterministic ID generation,
validity window enforcement, claim link constraints, and migration
structural validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.assertions.schemas import (
    VALID_ASSERTION_STATUSES,
    VALID_LINK_TYPES,
    AssertionClaimLink,
    ResolvedAssertion,
    make_assertion_id,
)

MIGRATION_PATH = Path("migrations/026_resolved_assertions.sql")


# -- Helpers ---------------------------------------------------------------


def _make_assertion(**overrides) -> ResolvedAssertion:
    defaults = dict(
        assertion_id="asrt_test_001",
        subject_concept_id="concept_issuer_tsmc",
        predicate="supplies_to",
        object_concept_id="concept_issuer_nvda",
        confidence=0.8,
    )
    defaults.update(overrides)
    return ResolvedAssertion(**defaults)


def _make_link(**overrides) -> AssertionClaimLink:
    defaults = dict(
        assertion_id="asrt_test_001",
        claim_id="claim_test_001",
        link_type="support",
    )
    defaults.update(overrides)
    return AssertionClaimLink(**defaults)


# -- ResolvedAssertion tests -----------------------------------------------


class TestResolvedAssertion:
    """ResolvedAssertion dataclass validation."""

    def test_valid_construction(self) -> None:
        a = _make_assertion()
        assert a.status == "active"
        assert a.support_count == 0
        assert a.contradiction_count == 0

    def test_invalid_status(self) -> None:
        with pytest.raises(ValueError, match="Invalid assertion status"):
            _make_assertion(status="unknown")

    def test_all_statuses_accepted(self) -> None:
        for s in VALID_ASSERTION_STATUSES:
            a = _make_assertion(status=s)
            assert a.status == s

    def test_valid_from_before_valid_to(self) -> None:
        a = _make_assertion(
            valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            valid_to=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        assert a.valid_from < a.valid_to

    def test_inverted_validity_rejected(self) -> None:
        with pytest.raises(ValueError, match="valid_from"):
            _make_assertion(
                valid_from=datetime(2025, 12, 31, tzinfo=timezone.utc),
                valid_to=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )

    def test_none_validity_allowed(self) -> None:
        a = _make_assertion(valid_from=None, valid_to=None)
        assert a.valid_from is None

    def test_open_ended_validity(self) -> None:
        """valid_from set, valid_to None = ongoing fact."""
        a = _make_assertion(
            valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            valid_to=None,
        )
        assert a.valid_to is None

    def test_unary_assertion(self) -> None:
        """Assertion without an object concept (unary predicate)."""
        a = _make_assertion(object_concept_id=None)
        assert a.object_concept_id is None

    def test_is_disputed_property(self) -> None:
        a = _make_assertion(contradiction_count=3)
        assert a.is_disputed is True

    def test_not_disputed_when_no_contradictions(self) -> None:
        a = _make_assertion(contradiction_count=0)
        assert a.is_disputed is False

    def test_net_support(self) -> None:
        a = _make_assertion(support_count=5, contradiction_count=2)
        assert a.net_support == 3

    def test_net_support_negative(self) -> None:
        a = _make_assertion(support_count=1, contradiction_count=4)
        assert a.net_support == -3


# -- make_assertion_id tests -----------------------------------------------


class TestMakeAssertionId:
    """Deterministic assertion ID generation."""

    def test_deterministic(self) -> None:
        id1 = make_assertion_id("concept_a", "supplies_to", "concept_b")
        id2 = make_assertion_id("concept_a", "supplies_to", "concept_b")
        assert id1 == id2

    def test_prefix(self) -> None:
        aid = make_assertion_id("concept_a", "supplies_to", "concept_b")
        assert aid.startswith("asrt_")

    def test_different_triples_different_ids(self) -> None:
        id1 = make_assertion_id("concept_a", "supplies_to", "concept_b")
        id2 = make_assertion_id("concept_a", "competes_with", "concept_b")
        assert id1 != id2

    def test_case_insensitive_predicate(self) -> None:
        id1 = make_assertion_id("concept_a", "supplies_to", "concept_b")
        id2 = make_assertion_id("concept_a", "SUPPLIES_TO", "concept_b")
        assert id1 == id2

    def test_none_object_handled(self) -> None:
        """Unary assertion ID is stable."""
        id1 = make_assertion_id("concept_a", "is_bankrupt", None)
        id2 = make_assertion_id("concept_a", "is_bankrupt", None)
        assert id1 == id2

    def test_none_vs_empty_object_same(self) -> None:
        id1 = make_assertion_id("concept_a", "pred", None)
        id2 = make_assertion_id("concept_a", "pred", "")
        assert id1 == id2


# -- AssertionClaimLink tests -----------------------------------------------


class TestAssertionClaimLink:
    """Claim link dataclass validation."""

    def test_valid_support(self) -> None:
        link = _make_link(link_type="support")
        assert link.link_type == "support"
        assert link.contribution_weight == 1.0

    def test_valid_contradiction(self) -> None:
        link = _make_link(link_type="contradiction")
        assert link.link_type == "contradiction"

    def test_invalid_link_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid link_type"):
            _make_link(link_type="neutral")

    def test_all_link_types(self) -> None:
        for lt in VALID_LINK_TYPES:
            link = _make_link(link_type=lt)
            assert link.link_type == lt

    def test_weight_zero(self) -> None:
        link = _make_link(contribution_weight=0.0)
        assert link.contribution_weight == 0.0

    def test_weight_one(self) -> None:
        link = _make_link(contribution_weight=1.0)
        assert link.contribution_weight == 1.0

    def test_weight_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="contribution_weight"):
            _make_link(contribution_weight=1.5)

    def test_negative_weight(self) -> None:
        with pytest.raises(ValueError, match="contribution_weight"):
            _make_link(contribution_weight=-0.1)


# -- Migration structural tests -------------------------------------------


class TestMigration026:
    """Structural validation of migration 026."""

    @pytest.fixture(autouse=True)
    def _load_sql(self) -> None:
        self.sql = MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_assertions_table(self) -> None:
        assert "CREATE TABLE IF NOT EXISTS news_intel.resolved_assertions" in self.sql

    def test_creates_links_table(self) -> None:
        assert "CREATE TABLE IF NOT EXISTS news_intel.assertion_claim_links" in self.sql

    def test_assertion_status_check(self) -> None:
        for s in VALID_ASSERTION_STATUSES:
            assert s in self.sql, f"Missing status {s!r}"

    def test_link_type_check(self) -> None:
        for lt in VALID_LINK_TYPES:
            assert lt in self.sql, f"Missing link_type {lt!r}"

    def test_concept_fks(self) -> None:
        assert "REFERENCES concepts(concept_id)" in self.sql

    def test_claim_fk(self) -> None:
        assert "REFERENCES news_intel.evidence_claims(claim_id)" in self.sql

    def test_assertion_fk_in_links(self) -> None:
        assert "REFERENCES news_intel.resolved_assertions(assertion_id)" in self.sql

    def test_composite_pk(self) -> None:
        assert "PRIMARY KEY (assertion_id, claim_id)" in self.sql

    def test_subject_index(self) -> None:
        assert "idx_assertions_subject" in self.sql

    def test_predicate_index(self) -> None:
        assert "idx_assertions_predicate" in self.sql

    def test_active_confidence_index(self) -> None:
        assert "idx_assertions_active_confidence" in self.sql

    def test_disputed_index(self) -> None:
        assert "idx_assertions_disputed" in self.sql

    def test_updated_at_trigger(self) -> None:
        assert "update_resolved_assertions_updated_at" in self.sql
        assert "update_updated_at_column" in self.sql

    def test_validity_columns(self) -> None:
        assert "valid_from" in self.sql
        assert "valid_to" in self.sql

    def test_evidence_count_columns(self) -> None:
        assert "support_count" in self.sql
        assert "contradiction_count" in self.sql
        assert "source_diversity" in self.sql
