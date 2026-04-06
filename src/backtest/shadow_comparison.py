"""Shadow-vs-current disagreement sets and QA summaries.

Compares new intelligence outputs against current system behavior,
producing concrete disagreement evidence for rollout decisions.

Three artifact types:
    - Disagreement: a single case where outputs differ
    - DisagreementSet: collection with agreement statistics
    - QASummary: aggregate summary with rollout recommendation

Comparison is key-based: the caller provides two dicts of
keyed outputs (current and shadow), the framework identifies
where they disagree and classifies severity.

All functions are stateless — the caller provides serialized
outputs from both systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# -- Severity and recommendation constants ------------------------------------

SEVERITY_MATERIAL = "material"
SEVERITY_MINOR = "minor"
SEVERITY_MISSING = "missing"

RECOMMEND_PROCEED = "proceed"
RECOMMEND_INVESTIGATE = "investigate"
RECOMMEND_BLOCK = "block"

# Thresholds for QA recommendation
DEFAULT_INVESTIGATE_THRESHOLD = 0.05  # material disagreement rate for investigate
DEFAULT_BLOCK_THRESHOLD = 0.15  # material disagreement rate for block


# -- Disagreement dataclass ---------------------------------------------------


@dataclass(frozen=True)
class Disagreement:
    """A single case where shadow and current outputs disagree.

    Attributes:
        key: What entity or decision point disagrees.
        category: Type of comparison (e.g., "ranking", "signal", "score").
        severity: "material", "minor", or "missing".
        current_value: What the current system produced.
        shadow_value: What the shadow system produced.
        explanation: Human-readable description of the disagreement.
        run_provenance: Run IDs for traceability.
    """

    key: str
    category: str
    severity: str
    current_value: Any = None
    shadow_value: Any = None
    explanation: str = ""
    run_provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "category": self.category,
            "severity": self.severity,
            "current_value": self.current_value,
            "shadow_value": self.shadow_value,
            "explanation": self.explanation,
            "run_provenance": self.run_provenance,
        }


# -- Disagreement set ----------------------------------------------------------


@dataclass(frozen=True)
class DisagreementSet:
    """Collection of disagreements with agreement statistics.

    Attributes:
        disagreements: All identified disagreements.
        total_comparisons: How many keys were compared.
        computed_at: When the comparison was performed.
    """

    disagreements: list[Disagreement] = field(default_factory=list)
    total_comparisons: int = 0
    computed_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    @property
    def category_counts(self) -> dict[str, int]:
        """Disagreements per category."""
        counts: dict[str, int] = {}
        for d in self.disagreements:
            counts[d.category] = counts.get(d.category, 0) + 1
        return counts

    @property
    def severity_counts(self) -> dict[str, int]:
        """Disagreements per severity level."""
        counts: dict[str, int] = {}
        for d in self.disagreements:
            counts[d.severity] = counts.get(d.severity, 0) + 1
        return counts

    @property
    def agreement_rate(self) -> float:
        """Fraction of comparisons that agreed."""
        if self.total_comparisons == 0:
            return 1.0
        return 1.0 - len(self.disagreements) / self.total_comparisons

    @property
    def material_count(self) -> int:
        """Number of material disagreements."""
        return self.severity_counts.get(SEVERITY_MATERIAL, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_comparisons": self.total_comparisons,
            "disagreement_count": len(self.disagreements),
            "agreement_rate": round(self.agreement_rate, 4),
            "material_count": self.material_count,
            "category_counts": self.category_counts,
            "severity_counts": self.severity_counts,
            "computed_at": self.computed_at.isoformat(),
        }


# -- QA summary ----------------------------------------------------------------


@dataclass(frozen=True)
class QASummary:
    """Aggregate QA summary for rollout decision-making.

    Attributes:
        disagreement_set: The underlying disagreements.
        recommendation: "proceed", "investigate", or "block".
        recommendation_reason: Why this recommendation was made.
        top_disagreements: Most important disagreements for review.
        current_run_id: Run ID of the current system.
        shadow_run_id: Run ID of the shadow system.
    """

    disagreement_set: DisagreementSet
    recommendation: str
    recommendation_reason: str = ""
    top_disagreements: list[Disagreement] = field(default_factory=list)
    current_run_id: str = ""
    shadow_run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
            "agreement_rate": round(self.disagreement_set.agreement_rate, 4),
            "material_count": self.disagreement_set.material_count,
            "total_comparisons": self.disagreement_set.total_comparisons,
            "top_disagreement_count": len(self.top_disagreements),
            "current_run_id": self.current_run_id,
            "shadow_run_id": self.shadow_run_id,
        }


# -- Comparison functions (stateless) ------------------------------------------


def compare_keyed_outputs(
    current: dict[str, Any],
    shadow: dict[str, Any],
    *,
    category: str = "output",
    numeric_tolerance: float = 0.01,
    run_provenance: dict[str, str] | None = None,
) -> list[Disagreement]:
    """Compare two dicts of keyed outputs and find disagreements.

    For each key present in either dict:
        - Both present + equal → agreement (no disagreement)
        - Both present + different → material or minor disagreement
        - Only one present → missing disagreement

    Numeric values use tolerance for minor vs material classification.

    Args:
        current: Current system outputs keyed by entity/decision.
        shadow: Shadow system outputs.
        category: Category label for all disagreements.
        numeric_tolerance: Tolerance for numeric comparison.
        run_provenance: Run IDs for traceability.

    Returns:
        List of Disagreement objects.
    """
    provenance = run_provenance or {}
    disagreements: list[Disagreement] = []

    all_keys = sorted(set(current) | set(shadow))
    for key in all_keys:
        in_current = key in current
        in_shadow = key in shadow

        if not in_current:
            disagreements.append(Disagreement(
                key=key, category=category, severity=SEVERITY_MISSING,
                current_value=None, shadow_value=shadow[key],
                explanation=f"{key}: present in shadow but not current",
                run_provenance=provenance,
            ))
        elif not in_shadow:
            disagreements.append(Disagreement(
                key=key, category=category, severity=SEVERITY_MISSING,
                current_value=current[key], shadow_value=None,
                explanation=f"{key}: present in current but not shadow",
                run_provenance=provenance,
            ))
        elif current[key] != shadow[key]:
            severity = _classify_severity(
                current[key], shadow[key], numeric_tolerance,
            )
            disagreements.append(Disagreement(
                key=key, category=category, severity=severity,
                current_value=current[key], shadow_value=shadow[key],
                explanation=f"{key}: {current[key]} → {shadow[key]}",
                run_provenance=provenance,
            ))

    return disagreements


def _classify_severity(
    current_val: Any,
    shadow_val: Any,
    tolerance: float,
) -> str:
    """Classify a value difference as material or minor.

    Numeric values within tolerance are minor. All other
    differences (strings, bools, lists) are material.
    """
    if (
        isinstance(current_val, (int, float))
        and isinstance(shadow_val, (int, float))
        and abs(current_val - shadow_val) <= tolerance
    ):
        return SEVERITY_MINOR
    return SEVERITY_MATERIAL


# -- Set builder ---------------------------------------------------------------


def build_disagreement_set(
    disagreements: list[Disagreement],
    total_comparisons: int,
    *,
    now: datetime | None = None,
) -> DisagreementSet:
    """Build a DisagreementSet from disagreements.

    Args:
        disagreements: All disagreements from comparison.
        total_comparisons: How many keys were compared.
        now: Computation timestamp.

    Returns:
        DisagreementSet (category/severity counts derived via properties).
    """
    if now is None:
        now = datetime.now(UTC)

    return DisagreementSet(
        disagreements=disagreements,
        total_comparisons=total_comparisons,
        computed_at=now,
    )


# -- QA summary builder -------------------------------------------------------


def build_qa_summary(
    disagreement_set: DisagreementSet,
    *,
    current_run_id: str = "",
    shadow_run_id: str = "",
    investigate_threshold: float = DEFAULT_INVESTIGATE_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
    top_n: int = 10,
) -> QASummary:
    """Build a QA summary with rollout recommendation.

    Recommendation logic:
        - proceed: material rate < investigate_threshold
        - investigate: material rate < block_threshold
        - block: material rate >= block_threshold

    Args:
        disagreement_set: The comparison results.
        current_run_id: For traceability.
        shadow_run_id: For traceability.
        investigate_threshold: Material rate threshold for investigate.
        block_threshold: Material rate threshold for block.
        top_n: How many top disagreements to include.

    Returns:
        QASummary with recommendation and top disagreements.
    """
    total = disagreement_set.total_comparisons
    material = disagreement_set.material_count
    material_rate = material / total if total > 0 else 0.0

    if material_rate >= block_threshold:
        recommendation = RECOMMEND_BLOCK
        reason = (
            f"Material disagreement rate {material_rate:.1%} "
            f"({material}/{total}) exceeds block threshold "
            f"({block_threshold:.0%})"
        )
    elif material_rate >= investigate_threshold:
        recommendation = RECOMMEND_INVESTIGATE
        reason = (
            f"Material disagreement rate {material_rate:.1%} "
            f"({material}/{total}) exceeds investigate threshold "
            f"({investigate_threshold:.0%})"
        )
    else:
        recommendation = RECOMMEND_PROCEED
        reason = (
            f"Material disagreement rate {material_rate:.1%} "
            f"({material}/{total}) within acceptable bounds"
        )

    # Top disagreements: material first, then by key for determinism
    top = sorted(
        disagreement_set.disagreements,
        key=lambda d: (
            0 if d.severity == SEVERITY_MATERIAL else 1,
            d.key,
        ),
    )[:top_n]

    return QASummary(
        disagreement_set=disagreement_set,
        recommendation=recommendation,
        recommendation_reason=reason,
        top_disagreements=top,
        current_run_id=current_run_id,
        shadow_run_id=shadow_run_id,
    )
