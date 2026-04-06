"""Tests for publish thresholds, quarantine policy, and cutover checklist.

Verifies gate evaluation, quarantine trigger firing, and the
full cutover go/no-go recommendation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.backtest.cutover_policy import (
    DEFAULT_GATES,
    DEFAULT_QUARANTINE_TRIGGERS,
    CutoverChecklist,
    QuarantineTrigger,
    evaluate_cutover_checklist,
    evaluate_gate,
    evaluate_quarantine_triggers,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Helpers ---------------------------------------------------------------


def _passing_values() -> dict[str, float]:
    """Metric values that pass all default gates."""
    return {
        "lineage_completeness": 0.97,
        "unresolved_entities": 0.03,
        "filing_parse_quality": 0.95,
        "stale_evidence": 0.05,
        "lane_failure_rate": 0.05,
        "manifest_seal_rate": 0.95,
        "bundle_integrity": 1.0,
        "coverage": 1.0,
        "contract_compat": 1.0,
        "shadow_material_rate": 0.02,
    }


def _failing_values() -> dict[str, float]:
    """Metric values that fail several gates."""
    return {
        "lineage_completeness": 0.80,
        "unresolved_entities": 0.20,
        "filing_parse_quality": 0.60,
        "stale_evidence": 0.30,
        "lane_failure_rate": 0.30,
        "manifest_seal_rate": 0.70,
        "bundle_integrity": 0.90,
        "coverage": 0.25,
        "contract_compat": 0.0,
        "shadow_material_rate": 0.25,
    }


# -- PublishGate tests -----------------------------------------------------


class TestPublishGate:
    """Individual gate evaluation."""

    def test_higher_is_better_passes(self) -> None:
        gate = evaluate_gate("lineage_completeness", 0.97)
        assert gate is not None
        assert gate.passed is True

    def test_higher_is_better_fails(self) -> None:
        gate = evaluate_gate("lineage_completeness", 0.80)
        assert gate is not None
        assert gate.passed is False

    def test_lower_is_better_passes(self) -> None:
        gate = evaluate_gate("unresolved_entities", 0.03)
        assert gate is not None
        assert gate.passed is True

    def test_lower_is_better_fails(self) -> None:
        gate = evaluate_gate("unresolved_entities", 0.20)
        assert gate is not None
        assert gate.passed is False

    def test_exact_threshold_passes_higher(self) -> None:
        gate = evaluate_gate("lineage_completeness", 0.95)
        assert gate is not None
        assert gate.passed is True

    def test_exact_threshold_passes_lower(self) -> None:
        gate = evaluate_gate("unresolved_entities", 0.05)
        assert gate is not None
        assert gate.passed is True

    def test_unknown_gate(self) -> None:
        gate = evaluate_gate("nonexistent", 0.5)
        assert gate is None

    def test_to_dict(self) -> None:
        gate = evaluate_gate("lineage_completeness", 0.97)
        assert gate is not None
        d = gate.to_dict()
        assert d["name"] == "lineage_completeness"
        assert d["passed"] is True

    def test_frozen(self) -> None:
        gate = evaluate_gate("lineage_completeness", 0.97)
        assert gate is not None
        with pytest.raises(AttributeError):
            gate.passed = False  # type: ignore[misc]


# -- QuarantineTrigger tests -----------------------------------------------


class TestQuarantineTrigger:
    """Automatic quarantine trigger evaluation."""

    def test_higher_is_worse_fires(self) -> None:
        trigger = QuarantineTrigger(
            name="test", metric_type="failure_rate",
            threshold=0.25, action="quarantine",
            reason_template="{lane}: rate {value:.1%}",
        )
        assert trigger.evaluate(0.30) is True

    def test_higher_is_worse_safe(self) -> None:
        trigger = QuarantineTrigger(
            name="test", metric_type="failure_rate",
            threshold=0.25, action="quarantine",
            reason_template="",
        )
        assert trigger.evaluate(0.10) is False

    def test_lower_is_worse_fires(self) -> None:
        trigger = QuarantineTrigger(
            name="test", metric_type="integrity",
            threshold=0.95, action="quarantine",
            reason_template="", higher_is_worse=False,
        )
        assert trigger.evaluate(0.90) is True

    def test_lower_is_worse_safe(self) -> None:
        trigger = QuarantineTrigger(
            name="test", metric_type="integrity",
            threshold=0.95, action="quarantine",
            reason_template="", higher_is_worse=False,
        )
        assert trigger.evaluate(0.99) is False

    def test_format_reason(self) -> None:
        trigger = QuarantineTrigger(
            name="test", metric_type="failure_rate",
            threshold=0.25, action="quarantine",
            reason_template="{lane}: rate {value:.1%} > {threshold:.0%}",
        )
        reason = trigger.format_reason("narrative", 0.30)
        assert "narrative" in reason
        assert "30.0%" in reason

    def test_to_dict(self) -> None:
        trigger = DEFAULT_QUARANTINE_TRIGGERS[0]
        d = trigger.to_dict()
        assert "name" in d
        assert "action" in d


# -- evaluate_quarantine_triggers tests ------------------------------------


class TestEvaluateQuarantineTriggers:
    """Batch quarantine trigger evaluation."""

    def test_no_triggers_fire(self) -> None:
        values = {"lane_failure_rate": 0.05, "stale_evidence": 0.05}
        fired = evaluate_quarantine_triggers(values, "narrative")
        assert fired == []

    def test_quarantine_fires(self) -> None:
        values = {"lane_failure_rate": 0.30}
        fired = evaluate_quarantine_triggers(values, "narrative")
        quarantines = [f for f in fired if f["action"] == "quarantine"]
        assert len(quarantines) >= 1

    def test_watch_fires(self) -> None:
        values = {"lane_failure_rate": 0.15}
        fired = evaluate_quarantine_triggers(values, "narrative")
        watches = [f for f in fired if f["action"] == "watch"]
        assert len(watches) >= 1

    def test_missing_metric_skipped(self) -> None:
        """Trigger for a metric not in values is silently skipped."""
        fired = evaluate_quarantine_triggers({}, "narrative")
        assert fired == []

    def test_fired_contains_details(self) -> None:
        values = {"lane_failure_rate": 0.30}
        fired = evaluate_quarantine_triggers(values, "narrative")
        assert fired[0]["lane"] == "narrative"
        assert "reason" in fired[0]
        assert fired[0]["value"] == 0.30

    def test_bundle_integrity_trigger(self) -> None:
        values = {"bundle_integrity": 0.90}
        fired = evaluate_quarantine_triggers(values, "all")
        assert len(fired) >= 1
        assert any(f["trigger"] == "bundle_corruption" for f in fired)


# -- CutoverChecklist tests ------------------------------------------------


class TestCutoverChecklist:
    """Full cutover go/no-go evaluation."""

    def test_all_passing_go(self) -> None:
        checklist = evaluate_cutover_checklist(
            _passing_values(), now=NOW,
        )
        assert checklist.all_passed is True
        assert checklist.recommendation == "go"
        assert checklist.failed_gates == []

    def test_some_failing_nogo(self) -> None:
        checklist = evaluate_cutover_checklist(
            _failing_values(), now=NOW,
        )
        assert checklist.all_passed is False
        assert "no-go" in checklist.recommendation
        assert len(checklist.failed_gates) > 0

    def test_missing_values_fail(self) -> None:
        """Gates with no provided value fail (default 0.0)."""
        checklist = evaluate_cutover_checklist({}, now=NOW)
        assert checklist.all_passed is False

    def test_failed_gates_named(self) -> None:
        values = _passing_values()
        values["lineage_completeness"] = 0.80
        checklist = evaluate_cutover_checklist(values, now=NOW)
        failed_names = {g.name for g in checklist.failed_gates}
        assert "lineage_completeness" in failed_names

    def test_quarantine_triggers_included(self) -> None:
        values = _failing_values()
        checklist = evaluate_cutover_checklist(values, now=NOW)
        assert len(checklist.triggered_quarantines) > 0

    def test_no_quarantines_when_healthy(self) -> None:
        checklist = evaluate_cutover_checklist(
            _passing_values(), now=NOW,
        )
        assert checklist.triggered_quarantines == []

    def test_to_dict(self) -> None:
        checklist = evaluate_cutover_checklist(
            _passing_values(), now=NOW,
        )
        d = checklist.to_dict()
        assert d["all_passed"] is True
        assert d["recommendation"] == "go"
        assert d["total_gates"] == len(DEFAULT_GATES)
        assert isinstance(d["evaluated_at"], str)

    def test_evaluated_at(self) -> None:
        checklist = evaluate_cutover_checklist({}, now=NOW)
        assert checklist.evaluated_at == NOW

    def test_empty_gates_nogo(self) -> None:
        """No gates evaluated → no-go."""
        checklist = CutoverChecklist(evaluated_at=NOW)
        assert checklist.all_passed is False
        assert "no gates" in checklist.recommendation

    def test_custom_gates(self) -> None:
        custom = [{"name": "custom", "description": "Test", "threshold": 0.5}]
        checklist = evaluate_cutover_checklist(
            {"custom": 0.6}, gates=custom, now=NOW,
        )
        assert checklist.all_passed is True

    def test_frozen(self) -> None:
        checklist = evaluate_cutover_checklist({}, now=NOW)
        with pytest.raises(AttributeError):
            checklist.evaluated_at = NOW  # type: ignore[misc]


# -- Realistic scenario tests -----------------------------------------------


class TestRealisticScenario:
    """End-to-end cutover evaluation."""

    def test_production_ready(self) -> None:
        """All metrics healthy → go recommendation."""
        checklist = evaluate_cutover_checklist(
            _passing_values(), now=NOW,
        )
        assert checklist.recommendation == "go"
        assert len(checklist.gates) == 10
        assert all(g.passed for g in checklist.gates)

    def test_single_regression_blocks(self) -> None:
        """One gate failing blocks the entire cutover."""
        values = _passing_values()
        values["shadow_material_rate"] = 0.20  # exceeds 0.15
        checklist = evaluate_cutover_checklist(values, now=NOW)
        assert checklist.recommendation != "go"
        assert any(g.name == "shadow_material_rate" for g in checklist.failed_gates)
