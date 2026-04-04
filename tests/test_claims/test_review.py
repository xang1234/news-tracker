"""Tests for the review queue schemas, triggers, and migration.

Tests are pure/in-memory where possible. Migration structural tests
validate the SQL against the Python schema expectations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.claims.review import (
    REVIEW_TRANSITIONS,
    VALID_RESOLUTIONS,
    VALID_TASK_STATUSES,
    VALID_TASK_TYPES,
    VALID_TRIGGER_REASONS,
    ReviewTask,
    make_review_task_id,
    validate_review_transition,
)
from src.claims.resolver import ResolverResult, ResolverTier
from src.claims.schemas import EvidenceClaim
from src.claims.triggers import (
    HIGH_IMPACT_PREDICATES,
    LOW_CONFIDENCE_THRESHOLD,
    build_merge_proposal,
    build_split_proposal,
    check_competing_predicates,
    check_high_impact_predicate,
    check_llm_proposed,
    check_low_confidence,
)
from src.security_master.concept_schemas import Concept

MIGRATION_PATH = Path("migrations/024_review_queue.sql")


# -- Helpers ---------------------------------------------------------------


def _make_claim(
    claim_id: str = "claim_test_001",
    *,
    predicate: str = "supplies_to",
    subject_concept_id: str | None = "concept_issuer_tsmc",
    object_concept_id: str | None = "concept_issuer_nvda",
    confidence: float = 0.5,
    run_id: str = "run_1",
    lane: str = "narrative",
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        claim_key=f"clk_{claim_id}",
        lane=lane,
        source_id="doc_1",
        predicate=predicate,
        subject_text="TSMC",
        subject_concept_id=subject_concept_id,
        object_concept_id=object_concept_id,
        confidence=confidence,
        run_id=run_id,
        contract_version="0.1.0",
    )


def _make_concept(concept_id: str, name: str = "Test") -> Concept:
    return Concept(
        concept_id=concept_id,
        concept_type="issuer",
        canonical_name=name,
    )


# -- ReviewTask schema tests -----------------------------------------------


class TestReviewTask:
    """ReviewTask dataclass validation."""

    def test_valid_construction(self) -> None:
        task = ReviewTask(
            task_id="review_test",
            task_type="entity_review",
            trigger_reason="low_confidence",
            claim_ids=["claim_1"],
            concept_ids=["concept_1"],
        )
        assert task.status == "pending"
        assert task.priority == 2

    def test_invalid_task_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid task_type"):
            ReviewTask(
                task_id="r1",
                task_type="unknown_type",
                trigger_reason="manual",
            )

    def test_invalid_status(self) -> None:
        with pytest.raises(ValueError, match="Invalid task status"):
            ReviewTask(
                task_id="r1",
                task_type="entity_review",
                trigger_reason="manual",
                status="invalid",
            )

    def test_invalid_trigger_reason(self) -> None:
        with pytest.raises(ValueError, match="Invalid trigger_reason"):
            ReviewTask(
                task_id="r1",
                task_type="entity_review",
                trigger_reason="not_a_reason",
            )

    def test_invalid_resolution(self) -> None:
        with pytest.raises(ValueError, match="Invalid resolution"):
            ReviewTask(
                task_id="r1",
                task_type="entity_review",
                trigger_reason="manual",
                resolution="bad_resolution",
            )

    def test_none_resolution_allowed(self) -> None:
        task = ReviewTask(
            task_id="r1",
            task_type="entity_review",
            trigger_reason="manual",
            resolution=None,
        )
        assert task.resolution is None

    def test_all_task_types_accepted(self) -> None:
        for tt in VALID_TASK_TYPES:
            task = ReviewTask(
                task_id=f"r_{tt}",
                task_type=tt,
                trigger_reason="manual",
            )
            assert task.task_type == tt

    def test_all_statuses_accepted(self) -> None:
        for s in VALID_TASK_STATUSES:
            task = ReviewTask(
                task_id=f"r_{s}",
                task_type="entity_review",
                trigger_reason="manual",
                status=s,
            )
            assert task.status == s

    def test_all_resolutions_accepted(self) -> None:
        for r in VALID_RESOLUTIONS:
            task = ReviewTask(
                task_id=f"r_{r}",
                task_type="entity_review",
                trigger_reason="manual",
                resolution=r,
            )
            assert task.resolution == r


# -- Deterministic ID tests ------------------------------------------------


class TestMakeReviewTaskId:
    """Deterministic review task ID generation."""

    def test_deterministic(self) -> None:
        id1 = make_review_task_id("entity_review", ["c1"], ["x1"])
        id2 = make_review_task_id("entity_review", ["c1"], ["x1"])
        assert id1 == id2

    def test_different_inputs_different_ids(self) -> None:
        id1 = make_review_task_id("entity_review", ["c1"], ["x1"])
        id2 = make_review_task_id("entity_review", ["c2"], ["x1"])
        assert id1 != id2

    def test_order_independent(self) -> None:
        """Sorted inputs produce same ID regardless of input order."""
        id1 = make_review_task_id("entity_review", ["c2", "c1"], ["x1"])
        id2 = make_review_task_id("entity_review", ["c1", "c2"], ["x1"])
        assert id1 == id2

    def test_prefix(self) -> None:
        tid = make_review_task_id("entity_review", ["c1"], ["x1"])
        assert tid.startswith("review_")


# -- State transition tests ------------------------------------------------


class TestReviewTransitions:
    """Review task state machine."""

    def test_pending_to_assigned(self) -> None:
        validate_review_transition("pending", "assigned")

    def test_pending_to_resolved(self) -> None:
        validate_review_transition("pending", "resolved")

    def test_pending_to_dismissed(self) -> None:
        validate_review_transition("pending", "dismissed")

    def test_assigned_to_resolved(self) -> None:
        validate_review_transition("assigned", "resolved")

    def test_assigned_to_pending(self) -> None:
        validate_review_transition("assigned", "pending")

    def test_resolved_is_terminal(self) -> None:
        with pytest.raises(ValueError, match="Invalid review transition"):
            validate_review_transition("resolved", "pending")

    def test_dismissed_is_terminal(self) -> None:
        with pytest.raises(ValueError, match="Invalid review transition"):
            validate_review_transition("dismissed", "pending")

    def test_unknown_status(self) -> None:
        with pytest.raises(ValueError, match="Unknown review status"):
            validate_review_transition("unknown", "pending")

    def test_all_terminal_states_have_no_transitions(self) -> None:
        for status, targets in REVIEW_TRANSITIONS.items():
            if not targets:
                with pytest.raises(ValueError):
                    validate_review_transition(status, "pending")


# -- Trigger: low confidence -----------------------------------------------


class TestCheckLowConfidence:
    """Trigger for fuzzy matches with close alternatives."""

    def test_triggers_on_fuzzy_with_alternatives(self) -> None:
        alt = _make_concept("concept_alt", "Alternative Corp")
        result = ResolverResult(
            mention="Taiwan Semi",
            concept=_make_concept("concept_tsmc", "TSMC"),
            concept_id="concept_tsmc",
            tier=ResolverTier.FUZZY,
            confidence=0.55,
            alternatives=[alt],
        )
        claim = _make_claim()
        task = check_low_confidence(result, claim)
        assert task is not None
        assert task.task_type == "entity_review"
        assert task.trigger_reason == "close_alternatives"
        assert "concept_tsmc" in task.concept_ids
        assert "concept_alt" in task.concept_ids

    def test_skips_exact_tier(self) -> None:
        result = ResolverResult(
            mention="NVDA",
            concept=_make_concept("c1", "NVIDIA"),
            concept_id="c1",
            tier=ResolverTier.EXACT,
            confidence=1.0,
        )
        assert check_low_confidence(result, _make_claim()) is None

    def test_skips_high_confidence_fuzzy(self) -> None:
        result = ResolverResult(
            mention="TSMC",
            concept=_make_concept("c1", "TSMC"),
            concept_id="c1",
            tier=ResolverTier.FUZZY,
            confidence=0.8,
            alternatives=[_make_concept("c2", "Alt")],
        )
        assert check_low_confidence(result, _make_claim()) is None

    def test_skips_fuzzy_without_alternatives(self) -> None:
        result = ResolverResult(
            mention="TSMC",
            concept=_make_concept("c1", "TSMC"),
            concept_id="c1",
            tier=ResolverTier.FUZZY,
            confidence=0.5,
            alternatives=[],
        )
        assert check_low_confidence(result, _make_claim()) is None


# -- Trigger: LLM proposed -------------------------------------------------


class TestCheckLlmProposed:
    """Trigger for any LLM-proposed resolution."""

    def test_triggers_on_llm_proposed(self) -> None:
        result = ResolverResult(
            mention="Unknown Entity",
            tier=ResolverTier.LLM_PROPOSED,
            metadata={"proposed_concept_id": "concept_new"},
        )
        task = check_llm_proposed(result, _make_claim())
        assert task is not None
        assert task.task_type == "entity_review"
        assert task.trigger_reason == "llm_proposed"
        assert task.priority == 1

    def test_skips_non_llm_tier(self) -> None:
        result = ResolverResult(
            mention="NVDA",
            concept=_make_concept("c1", "NVIDIA"),
            concept_id="c1",
            tier=ResolverTier.EXACT,
        )
        assert check_llm_proposed(result, _make_claim()) is None

    def test_captures_gate_metadata(self) -> None:
        result = ResolverResult(
            mention="Unknown",
            tier=ResolverTier.LLM_PROPOSED,
            metadata={
                "match_type": "llm_gate_approved",
                "passage_length": 300,
            },
        )
        task = check_llm_proposed(result, _make_claim())
        assert task is not None
        assert task.payload["gate_metadata"]["passage_length"] == 300


# -- Trigger: contradiction -------------------------------------------------


class TestCheckCompetingPredicates:
    """Trigger for competing predicate claims."""

    def test_triggers_on_different_predicates(self) -> None:
        claim_a = _make_claim("claim_a", predicate="supplies_to")
        claim_b = _make_claim("claim_b", predicate="competes_with")
        task = check_competing_predicates(claim_a, claim_b)
        assert task is not None
        assert task.task_type == "claim_review"
        assert task.trigger_reason == "contradiction"
        assert "claim_a" in task.claim_ids
        assert "claim_b" in task.claim_ids

    def test_skips_same_predicate(self) -> None:
        claim_a = _make_claim("claim_a", predicate="supplies_to")
        claim_b = _make_claim("claim_b", predicate="supplies_to")
        assert check_competing_predicates(claim_a, claim_b) is None

    def test_skips_different_subjects(self) -> None:
        claim_a = _make_claim("claim_a", subject_concept_id="c1")
        claim_b = _make_claim("claim_b", subject_concept_id="c2")
        assert check_competing_predicates(claim_a, claim_b) is None

    def test_skips_unresolved_concepts(self) -> None:
        claim_a = _make_claim("claim_a", subject_concept_id=None)
        claim_b = _make_claim("claim_b", subject_concept_id=None)
        assert check_competing_predicates(claim_a, claim_b) is None

    def test_deterministic_task_id(self) -> None:
        claim_a = _make_claim("claim_a", predicate="supplies_to")
        claim_b = _make_claim("claim_b", predicate="competes_with")
        task1 = check_competing_predicates(claim_a, claim_b)
        task2 = check_competing_predicates(claim_b, claim_a)
        assert task1.task_id == task2.task_id


# -- Trigger: high impact predicate ----------------------------------------


class TestCheckHighImpactPredicate:
    """Trigger for low-confidence high-impact claims."""

    def test_triggers_on_low_confidence_high_impact(self) -> None:
        claim = _make_claim(predicate="supplies_to", confidence=0.4)
        task = check_high_impact_predicate(claim)
        assert task is not None
        assert task.trigger_reason == "high_impact_predicate"
        assert task.payload["predicate"] == "supplies_to"

    def test_skips_high_confidence(self) -> None:
        claim = _make_claim(predicate="supplies_to", confidence=0.9)
        assert check_high_impact_predicate(claim) is None

    def test_skips_non_high_impact(self) -> None:
        claim = _make_claim(predicate="mentions", confidence=0.3)
        assert check_high_impact_predicate(claim) is None

    def test_custom_threshold(self) -> None:
        claim = _make_claim(predicate="acquires", confidence=0.8)
        # Default threshold 0.7 would skip this, but custom 0.9 catches it
        task = check_high_impact_predicate(
            claim, confidence_threshold=0.9
        )
        assert task is not None

    def test_all_high_impact_predicates_covered(self) -> None:
        """All HIGH_IMPACT_PREDICATES trigger when confidence is low."""
        for pred in HIGH_IMPACT_PREDICATES:
            claim = _make_claim(predicate=pred, confidence=0.3)
            task = check_high_impact_predicate(claim)
            assert task is not None, f"Expected trigger for {pred}"


# -- Merge/split proposals -------------------------------------------------


class TestProposals:
    """Merge and split proposal builders."""

    def test_merge_proposal(self) -> None:
        task = build_merge_proposal(
            "concept_a",
            "concept_b",
            claim_ids=["c1"],
            confidence=0.8,
            evidence="Same company, different names",
        )
        assert task.task_type == "merge_proposal"
        assert task.payload["source_concept_id"] == "concept_a"
        assert task.payload["target_concept_id"] == "concept_b"
        assert task.payload["confidence"] == 0.8

    def test_merge_proposal_deterministic(self) -> None:
        t1 = build_merge_proposal("concept_a", "concept_b")
        t2 = build_merge_proposal("concept_a", "concept_b")
        assert t1.task_id == t2.task_id

    def test_split_proposal(self) -> None:
        task = build_split_proposal(
            "concept_ambiguous",
            ["Samsung Electronics", "Samsung SDI"],
            claim_ids=["c1", "c2"],
            evidence="Different subsidiaries",
        )
        assert task.task_type == "split_proposal"
        assert task.payload["proposed_names"] == [
            "Samsung Electronics",
            "Samsung SDI",
        ]

    def test_split_proposal_deterministic(self) -> None:
        t1 = build_split_proposal("c1", ["A", "B"])
        t2 = build_split_proposal("c1", ["A", "B"])
        assert t1.task_id == t2.task_id


# -- Migration structural tests -------------------------------------------


class TestMigration024:
    """Structural validation of migration 024."""

    @pytest.fixture(autouse=True)
    def _load_sql(self) -> None:
        self.sql = MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_review_tasks(self) -> None:
        assert "CREATE TABLE IF NOT EXISTS news_intel.review_tasks" in self.sql

    def test_task_type_check(self) -> None:
        for tt in VALID_TASK_TYPES:
            assert tt in self.sql, f"Missing task_type {tt!r} in CHECK"

    def test_status_check(self) -> None:
        for s in VALID_TASK_STATUSES:
            assert s in self.sql, f"Missing status {s!r} in CHECK"

    def test_trigger_reason_check(self) -> None:
        for tr in VALID_TRIGGER_REASONS:
            assert tr in self.sql, f"Missing trigger_reason {tr!r} in CHECK"

    def test_resolution_check(self) -> None:
        for r in VALID_RESOLUTIONS:
            assert r in self.sql, f"Missing resolution {r!r} in CHECK"

    def test_array_columns(self) -> None:
        assert "claim_ids" in self.sql
        assert "concept_ids" in self.sql
        assert "TEXT[]" in self.sql

    def test_gin_indexes(self) -> None:
        assert "USING GIN (claim_ids)" in self.sql
        assert "USING GIN (concept_ids)" in self.sql

    def test_queue_index(self) -> None:
        assert "idx_review_tasks_queue" in self.sql
        assert "priority ASC, created_at ASC" in self.sql

    def test_updated_at_trigger(self) -> None:
        assert "update_review_tasks_updated_at" in self.sql
        assert "update_updated_at_column" in self.sql

    def test_jsonb_columns(self) -> None:
        for col in ("payload", "lineage", "metadata"):
            assert f"{col}" in self.sql
