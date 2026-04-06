"""Tests for shadow-vs-current disagreement sets and QA summaries.

Verifies disagreement detection, severity classification, set
statistics, and rollout recommendation logic.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.backtest.shadow_comparison import (
    RECOMMEND_BLOCK,
    RECOMMEND_INVESTIGATE,
    RECOMMEND_PROCEED,
    SEVERITY_MATERIAL,
    SEVERITY_MINOR,
    SEVERITY_MISSING,
    Disagreement,
    _classify_severity,
    build_disagreement_set,
    build_qa_summary,
    compare_keyed_outputs,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Severity classification tests -------------------------------------------


class TestClassifySeverity:
    """Numeric tolerance and type-based classification."""

    def test_numeric_within_tolerance(self) -> None:
        assert _classify_severity(0.50, 0.505, 0.01) == SEVERITY_MINOR

    def test_numeric_beyond_tolerance(self) -> None:
        assert _classify_severity(0.50, 0.60, 0.01) == SEVERITY_MATERIAL

    def test_numeric_exact_boundary(self) -> None:
        """Value difference exactly at tolerance is minor."""
        assert _classify_severity(10, 11, 1.0) == SEVERITY_MINOR

    def test_string_difference_always_material(self) -> None:
        assert _classify_severity("high", "low", 0.01) == SEVERITY_MATERIAL

    def test_bool_difference_material(self) -> None:
        assert _classify_severity(True, False, 0.01) == SEVERITY_MATERIAL

    def test_list_difference_material(self) -> None:
        assert _classify_severity([1, 2], [1, 3], 0.01) == SEVERITY_MATERIAL

    def test_int_within_tolerance(self) -> None:
        assert _classify_severity(10, 10, 0.5) == SEVERITY_MINOR

    def test_mixed_numeric_types(self) -> None:
        assert _classify_severity(1, 1.005, 0.01) == SEVERITY_MINOR


# -- Keyed output comparison tests -------------------------------------------


class TestCompareKeyedOutputs:
    """Compare two dicts and find disagreements."""

    def test_identical_no_disagreements(self) -> None:
        current = {"a": 1, "b": 2}
        shadow = {"a": 1, "b": 2}
        assert compare_keyed_outputs(current, shadow) == []

    def test_value_difference(self) -> None:
        current = {"theme_rank": "TSM"}
        shadow = {"theme_rank": "NVDA"}
        ds = compare_keyed_outputs(current, shadow)
        assert len(ds) == 1
        assert ds[0].key == "theme_rank"
        assert ds[0].severity == SEVERITY_MATERIAL

    def test_numeric_minor(self) -> None:
        current = {"score": 0.50}
        shadow = {"score": 0.505}
        ds = compare_keyed_outputs(current, shadow, numeric_tolerance=0.01)
        assert len(ds) == 1
        assert ds[0].severity == SEVERITY_MINOR

    def test_missing_in_current(self) -> None:
        current = {}
        shadow = {"new_signal": True}
        ds = compare_keyed_outputs(current, shadow)
        assert len(ds) == 1
        assert ds[0].severity == SEVERITY_MISSING
        assert "shadow but not current" in ds[0].explanation

    def test_missing_in_shadow(self) -> None:
        current = {"old_signal": True}
        shadow = {}
        ds = compare_keyed_outputs(current, shadow)
        assert len(ds) == 1
        assert ds[0].severity == SEVERITY_MISSING
        assert "current but not shadow" in ds[0].explanation

    def test_mixed_disagreements(self) -> None:
        current = {"a": 1, "b": "x", "c": 0.5}
        shadow = {"a": 2, "b": "y", "d": 0.7}
        ds = compare_keyed_outputs(current, shadow)
        keys = {d.key for d in ds}
        assert keys == {"a", "b", "c", "d"}

    def test_category_applied(self) -> None:
        ds = compare_keyed_outputs({"a": 1}, {"a": 2}, category="ranking")
        assert ds[0].category == "ranking"

    def test_provenance_applied(self) -> None:
        prov = {"current": "run_001", "shadow": "run_002"}
        ds = compare_keyed_outputs({"a": 1}, {"a": 2}, run_provenance=prov)
        assert ds[0].run_provenance == prov

    def test_empty_inputs(self) -> None:
        assert compare_keyed_outputs({}, {}) == []

    def test_to_dict(self) -> None:
        ds = compare_keyed_outputs({"a": 1}, {"a": 2})
        d = ds[0].to_dict()
        assert d["key"] == "a"
        assert d["severity"] == SEVERITY_MATERIAL


# -- Disagreement set tests ---------------------------------------------------


class TestDisagreementSet:
    """Set statistics and agreement rate."""

    def test_perfect_agreement(self) -> None:
        ds = build_disagreement_set([], 10, now=NOW)
        assert ds.agreement_rate == 1.0
        assert ds.material_count == 0

    def test_partial_agreement(self) -> None:
        disagreements = [
            Disagreement(key="a", category="test", severity=SEVERITY_MATERIAL),
            Disagreement(key="b", category="test", severity=SEVERITY_MINOR),
        ]
        ds = build_disagreement_set(disagreements, 10, now=NOW)
        assert ds.agreement_rate == 0.8
        assert ds.material_count == 1

    def test_zero_comparisons(self) -> None:
        ds = build_disagreement_set([], 0, now=NOW)
        assert ds.agreement_rate == 1.0

    def test_category_counts(self) -> None:
        disagreements = [
            Disagreement(key="a", category="ranking", severity=SEVERITY_MATERIAL),
            Disagreement(key="b", category="ranking", severity=SEVERITY_MINOR),
            Disagreement(key="c", category="signal", severity=SEVERITY_MATERIAL),
        ]
        ds = build_disagreement_set(disagreements, 10, now=NOW)
        assert ds.category_counts["ranking"] == 2
        assert ds.category_counts["signal"] == 1

    def test_severity_counts(self) -> None:
        disagreements = [
            Disagreement(key="a", category="test", severity=SEVERITY_MATERIAL),
            Disagreement(key="b", category="test", severity=SEVERITY_MATERIAL),
            Disagreement(key="c", category="test", severity=SEVERITY_MINOR),
        ]
        ds = build_disagreement_set(disagreements, 10, now=NOW)
        assert ds.severity_counts[SEVERITY_MATERIAL] == 2
        assert ds.severity_counts[SEVERITY_MINOR] == 1

    def test_to_dict(self) -> None:
        ds = build_disagreement_set([], 5, now=NOW)
        d = ds.to_dict()
        assert d["total_comparisons"] == 5
        assert d["agreement_rate"] == 1.0
        assert isinstance(d["computed_at"], str)


# -- QA summary tests ---------------------------------------------------------


class TestQASummary:
    """Rollout recommendation logic."""

    def test_proceed(self) -> None:
        ds = build_disagreement_set([], 100, now=NOW)
        qa = build_qa_summary(ds)
        assert qa.recommendation == RECOMMEND_PROCEED
        assert "within acceptable" in qa.recommendation_reason

    def test_investigate(self) -> None:
        disagreements = [
            Disagreement(key=f"d{i}", category="test", severity=SEVERITY_MATERIAL) for i in range(8)
        ]
        ds = build_disagreement_set(disagreements, 100, now=NOW)
        qa = build_qa_summary(ds)
        assert qa.recommendation == RECOMMEND_INVESTIGATE

    def test_block(self) -> None:
        disagreements = [
            Disagreement(key=f"d{i}", category="test", severity=SEVERITY_MATERIAL)
            for i in range(20)
        ]
        ds = build_disagreement_set(disagreements, 100, now=NOW)
        qa = build_qa_summary(ds)
        assert qa.recommendation == RECOMMEND_BLOCK

    def test_minor_disagreements_dont_trigger(self) -> None:
        """Only material disagreements count toward thresholds."""
        disagreements = [
            Disagreement(key=f"d{i}", category="test", severity=SEVERITY_MINOR) for i in range(50)
        ]
        ds = build_disagreement_set(disagreements, 100, now=NOW)
        qa = build_qa_summary(ds)
        assert qa.recommendation == RECOMMEND_PROCEED

    def test_custom_thresholds(self) -> None:
        disagreements = [
            Disagreement(key="d0", category="test", severity=SEVERITY_MATERIAL),
        ]
        ds = build_disagreement_set(disagreements, 100, now=NOW)
        qa = build_qa_summary(
            ds,
            investigate_threshold=0.005,
            block_threshold=0.02,
        )
        assert qa.recommendation == RECOMMEND_INVESTIGATE

    def test_top_disagreements_material_first(self) -> None:
        disagreements = [
            Disagreement(key="minor1", category="test", severity=SEVERITY_MINOR),
            Disagreement(key="material1", category="test", severity=SEVERITY_MATERIAL),
            Disagreement(key="minor2", category="test", severity=SEVERITY_MINOR),
        ]
        ds = build_disagreement_set(disagreements, 10, now=NOW)
        qa = build_qa_summary(ds, top_n=2)
        assert len(qa.top_disagreements) == 2
        assert qa.top_disagreements[0].severity == SEVERITY_MATERIAL

    def test_top_n_limits(self) -> None:
        disagreements = [
            Disagreement(key=f"d{i}", category="test", severity=SEVERITY_MATERIAL)
            for i in range(20)
        ]
        ds = build_disagreement_set(disagreements, 100, now=NOW)
        qa = build_qa_summary(ds, top_n=5)
        assert len(qa.top_disagreements) == 5

    def test_run_provenance(self) -> None:
        ds = build_disagreement_set([], 10, now=NOW)
        qa = build_qa_summary(
            ds,
            current_run_id="run_cur",
            shadow_run_id="run_shd",
        )
        assert qa.current_run_id == "run_cur"
        assert qa.shadow_run_id == "run_shd"

    def test_to_dict(self) -> None:
        ds = build_disagreement_set([], 10, now=NOW)
        qa = build_qa_summary(ds)
        d = qa.to_dict()
        assert d["recommendation"] == RECOMMEND_PROCEED
        assert "agreement_rate" in d
        assert "material_count" in d

    def test_zero_comparisons(self) -> None:
        ds = build_disagreement_set([], 0, now=NOW)
        qa = build_qa_summary(ds)
        assert qa.recommendation == RECOMMEND_PROCEED


# -- Dataclass tests -----------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass invariants."""

    def test_disagreement_frozen(self) -> None:
        d = Disagreement(key="x", category="test", severity=SEVERITY_MATERIAL)
        with pytest.raises(AttributeError):
            d.key = "y"  # type: ignore[misc]

    def test_set_frozen(self) -> None:
        ds = build_disagreement_set([], 0, now=NOW)
        with pytest.raises(AttributeError):
            ds.total_comparisons = 99  # type: ignore[misc]

    def test_summary_frozen(self) -> None:
        ds = build_disagreement_set([], 0, now=NOW)
        qa = build_qa_summary(ds)
        with pytest.raises(AttributeError):
            qa.recommendation = "x"  # type: ignore[misc]
