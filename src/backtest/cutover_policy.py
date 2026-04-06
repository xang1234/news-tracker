"""Publish thresholds, quarantine policy, and production cutover checklist.

Codifies the explicit go/no-go rules for moving the intelligence
layer from shadow mode to production publication. Every gate is
a named check with a numeric threshold and pass/fail verdict.

Three policy layers:
    - Publish gates: per-metric quality floors for publication
    - Quarantine triggers: automatic lane-blocking rules
    - Cutover checklist: conjunction of all gates for go/no-go

All functions are stateless — the caller provides current metric
values, the policy evaluates them against thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# -- Publish gate --------------------------------------------------------------


@dataclass(frozen=True)
class PublishGate:
    """A single go/no-go gate for publication.

    Attributes:
        name: Gate identifier (e.g., "lineage_completeness").
        description: What this gate checks.
        threshold: Required value for passing.
        current_value: Measured value.
        passed: Whether the gate passed.
        higher_is_better: If True, current >= threshold passes.
    """

    name: str
    description: str
    threshold: float
    current_value: float
    passed: bool
    higher_is_better: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "current_value": round(self.current_value, 4),
            "passed": self.passed,
        }


# -- Quarantine trigger --------------------------------------------------------


@dataclass(frozen=True)
class QuarantineTrigger:
    """An automatic quarantine rule.

    When a metric crosses the threshold, the specified action
    should be taken on the lane. The caller uses
    operator_hooks.QuarantineAction to execute.

    Attributes:
        name: Trigger identifier.
        metric_type: Which quality metric this watches.
        threshold: Value at which the trigger fires.
        action: "quarantine" or "watch".
        reason_template: Template for the quarantine reason.
        higher_is_worse: If True, value > threshold triggers.
    """

    name: str
    metric_type: str
    threshold: float
    action: str
    reason_template: str
    higher_is_worse: bool = True

    def evaluate(self, value: float) -> bool:
        """Check whether the trigger should fire.

        Returns True if the metric value exceeds the threshold.
        """
        if self.higher_is_worse:
            return value > self.threshold
        return value < self.threshold

    def format_reason(self, lane: str, value: float) -> str:
        """Format the quarantine reason with current values."""
        return self.reason_template.format(
            lane=lane, value=value, threshold=self.threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_type": self.metric_type,
            "threshold": self.threshold,
            "action": self.action,
        }


# -- Cutover checklist ---------------------------------------------------------


@dataclass(frozen=True)
class CutoverChecklist:
    """Complete go/no-go evaluation for production cutover.

    All gates must pass for "go." Any failure produces "no-go"
    with the failing gates identified.

    Attributes:
        gates: All evaluated gates.
        triggered_quarantines: Quarantine triggers that fired.
        evaluated_at: When the checklist was evaluated.
    """

    gates: list[PublishGate] = field(default_factory=list)
    triggered_quarantines: list[dict[str, Any]] = field(default_factory=list)
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def all_passed(self) -> bool:
        """True only if every gate passed."""
        return all(g.passed for g in self.gates) if self.gates else False

    @property
    def failed_gates(self) -> list[PublishGate]:
        """Gates that did not pass."""
        return [g for g in self.gates if not g.passed]

    @property
    def recommendation(self) -> str:
        """Go/no-go recommendation."""
        if not self.gates:
            return "no-go: no gates evaluated"
        if self.all_passed:
            return "go"
        failed = ", ".join(g.name for g in self.failed_gates)
        return f"no-go: {failed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "recommendation": self.recommendation,
            "total_gates": len(self.gates),
            "passed_gates": sum(1 for g in self.gates if g.passed),
            "failed_gates": [g.to_dict() for g in self.failed_gates],
            "triggered_quarantines": self.triggered_quarantines,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# -- Default policy thresholds ------------------------------------------------

# Quality gates (from quality_metrics.py)
DEFAULT_GATES: list[dict[str, Any]] = [
    {
        "name": "lineage_completeness",
        "description": "Published objects with source lineage",
        "threshold": 0.95,
        "higher_is_better": True,
    },
    {
        "name": "unresolved_entities",
        "description": "Entity resolution failure rate",
        "threshold": 0.05,
        "higher_is_better": False,
    },
    {
        "name": "filing_parse_quality",
        "description": "Filing ingestion success rate",
        "threshold": 0.90,
        "higher_is_better": True,
    },
    {
        "name": "stale_evidence",
        "description": "Assertions with stale evidence",
        "threshold": 0.10,
        "higher_is_better": False,
    },
    {
        "name": "lane_failure_rate",
        "description": "Lane run failure rate (worst lane)",
        "threshold": 0.10,
        "higher_is_better": False,
    },
    {
        "name": "manifest_seal_rate",
        "description": "Manifest sealing success rate",
        "threshold": 0.90,
        "higher_is_better": True,
    },
    {
        "name": "bundle_integrity",
        "description": "Bundle checksum verification rate",
        "threshold": 0.99,
        "higher_is_better": True,
    },
    {
        "name": "coverage",
        "description": "Lane coverage in composite manifests",
        "threshold": 0.75,
        "higher_is_better": True,
    },
    {
        "name": "contract_compat",
        "description": "Contract version compatibility",
        "threshold": 1.0,
        "higher_is_better": True,
    },
    {
        "name": "shadow_material_rate",
        "description": "Shadow vs current material disagreement rate",
        "threshold": 0.15,
        "higher_is_better": False,
    },
]

# Quarantine auto-triggers
DEFAULT_QUARANTINE_TRIGGERS: list[QuarantineTrigger] = [
    QuarantineTrigger(
        name="critical_failure_rate",
        metric_type="lane_failure_rate",
        threshold=0.25,
        action="quarantine",
        reason_template="{lane}: failure rate {value:.1%} exceeds {threshold:.0%}",
    ),
    QuarantineTrigger(
        name="high_failure_rate",
        metric_type="lane_failure_rate",
        threshold=0.10,
        action="watch",
        reason_template="{lane}: failure rate {value:.1%} exceeds watch threshold {threshold:.0%}",
    ),
    QuarantineTrigger(
        name="critical_stale_evidence",
        metric_type="stale_evidence",
        threshold=0.25,
        action="quarantine",
        reason_template="{lane}: stale evidence {value:.1%} exceeds {threshold:.0%}",
    ),
    QuarantineTrigger(
        name="bundle_corruption",
        metric_type="bundle_integrity",
        threshold=0.95,
        action="quarantine",
        reason_template="Bundle integrity {value:.1%} below {threshold:.0%}",
        higher_is_worse=False,
    ),
]


# -- Gate evaluation -----------------------------------------------------------


def evaluate_gate(
    name: str,
    current_value: float,
    *,
    gates: list[dict[str, Any]] | None = None,
) -> PublishGate | None:
    """Evaluate a single named gate against the default policy.

    Args:
        name: Gate name (must match a gate in the policy).
        current_value: Measured metric value.
        gates: Override gate definitions (default: DEFAULT_GATES).

    Returns:
        PublishGate with pass/fail verdict, or None if gate not found.
    """
    if gates is None:
        gates = DEFAULT_GATES

    gate_def = next((g for g in gates if g["name"] == name), None)
    if gate_def is None:
        return None

    higher = gate_def.get("higher_is_better", True)
    threshold = gate_def["threshold"]
    passed = current_value >= threshold if higher else current_value <= threshold

    return PublishGate(
        name=name,
        description=gate_def["description"],
        threshold=threshold,
        current_value=current_value,
        passed=passed,
        higher_is_better=higher,
    )


def evaluate_quarantine_triggers(
    metric_values: dict[str, float],
    lane: str,
    *,
    triggers: list[QuarantineTrigger] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate quarantine triggers against current values.

    Returns a list of triggered actions (empty if none fire).
    Each entry has the trigger info and formatted reason.

    Args:
        metric_values: metric_type → current value.
        lane: Lane being evaluated.
        triggers: Override triggers (default: DEFAULT_QUARANTINE_TRIGGERS).

    Returns:
        List of triggered quarantine action dicts.
    """
    if triggers is None:
        triggers = DEFAULT_QUARANTINE_TRIGGERS

    fired: list[dict[str, Any]] = []
    for trigger in triggers:
        value = metric_values.get(trigger.metric_type)
        if value is None:
            continue
        if trigger.evaluate(value):
            fired.append({
                "trigger": trigger.name,
                "action": trigger.action,
                "lane": lane,
                "reason": trigger.format_reason(lane, value),
                "metric_type": trigger.metric_type,
                "value": value,
                "threshold": trigger.threshold,
            })
    return fired


# -- Full checklist evaluation -------------------------------------------------


def evaluate_cutover_checklist(
    metric_values: dict[str, float],
    *,
    lane: str = "all",
    gates: list[dict[str, Any]] | None = None,
    triggers: list[QuarantineTrigger] | None = None,
    now: datetime | None = None,
) -> CutoverChecklist:
    """Evaluate the full cutover checklist.

    Runs all gates and quarantine triggers against the provided
    metric values. Returns a CutoverChecklist with go/no-go
    recommendation.

    Args:
        metric_values: gate_name or metric_type → current value.
        lane: Lane context for quarantine triggers.
        gates: Override gate definitions.
        triggers: Override quarantine triggers.
        now: Evaluation timestamp.

    Returns:
        CutoverChecklist with pass/fail for each gate.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if gates is None:
        gates = DEFAULT_GATES

    evaluated_gates: list[PublishGate] = []
    for gate_def in gates:
        name = gate_def["name"]
        value = metric_values.get(name)
        if value is None:
            evaluated_gates.append(PublishGate(
                name=name,
                description=gate_def["description"],
                threshold=gate_def["threshold"],
                current_value=0.0,
                passed=False,
                higher_is_better=gate_def.get("higher_is_better", True),
            ))
        else:
            gate = evaluate_gate(name, value, gates=gates)
            if gate is not None:
                evaluated_gates.append(gate)

    triggered = evaluate_quarantine_triggers(
        metric_values, lane, triggers=triggers,
    )

    return CutoverChecklist(
        gates=evaluated_gates,
        triggered_quarantines=triggered,
        evaluated_at=now,
    )
