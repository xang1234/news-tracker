"""Parity and validation checks for narrative lane V2.

Validates that the new narrative lane model (components, passage
mappings, lane adapter) produces consistent results with the
existing narrative machinery. These checks are the quality gate
before narrative publication is trusted.

Validation dimensions:
    - Component parity: new decomposed scores agree directionally
      with legacy conviction_score
    - Backfill stability: re-running produces deterministic results
    - Replay coverage: every triggered signal has a component breakdown

All validation functions are stateless — they take narrative data
as input and return validation results. The caller handles I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.narrative.components import (
    NarrativeComponents,
    compute_narrative_components,
)
from src.narrative.schemas import NarrativeRun


# -- Validation results ----------------------------------------------------


@dataclass(frozen=True)
class ParityCheck:
    """Result of a single parity validation.

    Attributes:
        check_name: What was validated.
        passed: Whether the check passed.
        message: Human-readable explanation.
        details: Structured validation data.
    """

    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParityReport:
    """Aggregated parity validation report.

    Attributes:
        checks: Individual check results.
        passed: Whether all checks passed.
        run_count: How many narrative runs were validated.
        validated_at: When the validation ran.
    """

    checks: list[ParityCheck] = field(default_factory=list)
    run_count: int = 0
    validated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[ParityCheck]:
        return [c for c in self.checks if not c.passed]

    def add(self, check: ParityCheck) -> None:
        self.checks.append(check)


# -- Component parity: new scores vs legacy conviction ---------------------

# Directional agreement threshold: component composite and legacy
# conviction should move in the same direction within this tolerance
DIRECTIONAL_TOLERANCE = 20.0  # points on 0-100 scale
FLOAT_TOLERANCE = 0.01  # for backfill metric comparison
STRONG_COMPONENT_THRESHOLD = 0.3  # min component score to explain a signal


def check_component_conviction_parity(
    run: NarrativeRun,
    components: NarrativeComponents,
) -> ParityCheck:
    """Check that component composite agrees directionally with legacy conviction.

    The new composite and legacy conviction_score should be in the
    same general range. Large divergence suggests a formula mismatch
    that needs investigation before publication.
    """
    legacy = run.conviction_score
    new = components.composite
    delta = abs(new - legacy)

    passed = delta <= DIRECTIONAL_TOLERANCE
    return ParityCheck(
        check_name="component_conviction_parity",
        passed=passed,
        message=(
            f"Component composite ({new:.1f}) vs legacy conviction "
            f"({legacy:.1f}): delta={delta:.1f} "
            f"({'within' if passed else 'exceeds'} tolerance {DIRECTIONAL_TOLERANCE})"
        ),
        details={
            "legacy_conviction": legacy,
            "component_composite": new,
            "delta": round(delta, 2),
            "tolerance": DIRECTIONAL_TOLERANCE,
        },
    )


# -- Backfill stability: same inputs → same outputs -----------------------


def check_backfill_stability(
    run_a: NarrativeRun,
    run_b: NarrativeRun,
) -> ParityCheck:
    """Check that two backfill runs of the same data produce matching results.

    Compares key metrics between two runs that should represent the
    same narrative. Differences suggest non-determinism in the
    backfill pipeline.
    """
    issues = []
    if run_a.doc_count != run_b.doc_count:
        issues.append(
            f"doc_count: {run_a.doc_count} vs {run_b.doc_count}"
        )
    if run_a.platform_count != run_b.platform_count:
        issues.append(
            f"platform_count: {run_a.platform_count} vs {run_b.platform_count}"
        )
    if abs(run_a.avg_sentiment - run_b.avg_sentiment) > FLOAT_TOLERANCE:
        issues.append(
            f"avg_sentiment: {run_a.avg_sentiment:.3f} vs {run_b.avg_sentiment:.3f}"
        )
    if abs(run_a.avg_authority - run_b.avg_authority) > FLOAT_TOLERANCE:
        issues.append(
            f"avg_authority: {run_a.avg_authority:.3f} vs {run_b.avg_authority:.3f}"
        )

    passed = len(issues) == 0
    return ParityCheck(
        check_name="backfill_stability",
        passed=passed,
        message=(
            "Backfill stable: metrics match"
            if passed
            else f"Backfill divergence: {'; '.join(issues)}"
        ),
        details={
            "run_a_id": run_a.run_id,
            "run_b_id": run_b.run_id,
            "issues": issues,
        },
    )


# -- Replay coverage: every signal has component breakdown -----------------


def check_replay_coverage(
    run: NarrativeRun,
    components: NarrativeComponents,
    signal_triggered: bool,
) -> ParityCheck:
    """Check that a triggered signal has a valid component breakdown.

    Every signal that fires should be explainable through the
    component scores. A triggered signal with zero component scores
    suggests the old signal system and new components disagree.
    """
    if not signal_triggered:
        return ParityCheck(
            check_name="replay_coverage",
            passed=True,
            message="No signal triggered — coverage check not applicable",
            details={"run_id": run.run_id, "signal_triggered": False},
        )

    has_strong_component = (
        components.attention.score > STRONG_COMPONENT_THRESHOLD
        or components.corroboration.score > STRONG_COMPONENT_THRESHOLD
        or components.confirmation.score > STRONG_COMPONENT_THRESHOLD
        or components.novelty_persistence.score > STRONG_COMPONENT_THRESHOLD
    )

    return ParityCheck(
        check_name="replay_coverage",
        passed=has_strong_component,
        message=(
            f"Signal triggered with component support "
            f"(attention={components.attention.score:.2f}, "
            f"corroboration={components.corroboration.score:.2f}, "
            f"confirmation={components.confirmation.score:.2f}, "
            f"novelty={components.novelty_persistence.score:.2f})"
            if has_strong_component
            else f"Signal triggered but no component score > {STRONG_COMPONENT_THRESHOLD} — investigate"
        ),
        details={
            "run_id": run.run_id,
            "signal_triggered": True,
            "attention": components.attention.score,
            "corroboration": components.corroboration.score,
            "confirmation": components.confirmation.score,
            "novelty_persistence": components.novelty_persistence.score,
            "composite": components.composite,
        },
    )


# -- Full parity validation ------------------------------------------------


def compute_components_for_run(
    run: NarrativeRun,
    *,
    source_type_count: int = 1,
    spread_hours: float | None = None,
    high_authority_doc_ratio: float = 0.0,
    now: datetime | None = None,
) -> NarrativeComponents:
    """Compute narrative components from a NarrativeRun's metrics.

    Convenience function that extracts the needed fields from
    NarrativeRun and passes them to compute_narrative_components.
    """
    return compute_narrative_components(
        current_rate_per_hour=run.current_rate_per_hour,
        current_acceleration=run.current_acceleration,
        doc_count=run.doc_count,
        platform_count=run.platform_count,
        source_type_count=source_type_count,
        spread_hours=spread_hours,
        avg_sentiment=run.avg_sentiment,
        avg_authority=run.avg_authority,
        high_authority_doc_ratio=high_authority_doc_ratio,
        last_document_at=run.last_document_at,
        started_at=run.started_at,
        now=now,
    )


def validate_run_parity(
    run: NarrativeRun,
    *,
    signal_triggered: bool = False,
    source_type_count: int = 1,
    spread_hours: float | None = None,
    high_authority_doc_ratio: float = 0.0,
    now: datetime | None = None,
) -> tuple[NarrativeComponents, list[ParityCheck]]:
    """Run all parity checks for a single narrative run.

    Computes components, then checks conviction parity and
    replay coverage. Returns the components and check results.
    """
    components = compute_components_for_run(
        run,
        source_type_count=source_type_count,
        spread_hours=spread_hours,
        high_authority_doc_ratio=high_authority_doc_ratio,
        now=now,
    )

    checks = [
        check_component_conviction_parity(run, components),
        check_replay_coverage(run, components, signal_triggered),
    ]

    return components, checks
