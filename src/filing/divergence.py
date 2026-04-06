"""Divergence alerts between narrative strength and filing confirmation.

Generates alerts when narrative momentum and SEC filing evidence
diverge materially. Each alert carries a reason code, severity,
human-readable explanation, and structured evidence so downstream
consumers can distinguish hype, lagging adoption, contradiction,
and adverse disclosure change.

Divergence scenarios:
    - narrative_without_filing: Strong narrative, weak filing adoption (hype risk)
    - filing_without_narrative: Strong filing adoption, weak narrative (under-appreciated)
    - adverse_drift: Unusual drift in risk/regulatory filing sections
    - contradictory_drift: Filings shrinking where narrative says growth
    - lagging_adoption: Partial adoption that hasn't yet caught up

All functions are stateless — the caller provides pre-computed
narrative strength, filing adoption scores, and drift decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.filing.adoption import FilingAdoptionScore
from src.filing.drift import DriftDecomposition

# -- Reason codes and severity ------------------------------------------------


class DivergenceReason(str, Enum):
    """Why narrative and filing signals diverge."""

    NARRATIVE_WITHOUT_FILING = "narrative_without_filing"
    FILING_WITHOUT_NARRATIVE = "filing_without_narrative"
    ADVERSE_DRIFT = "adverse_drift"
    CONTRADICTORY_DRIFT = "contradictory_drift"
    LAGGING_ADOPTION = "lagging_adoption"


VALID_SEVERITIES = frozenset({"critical", "warning", "info"})


# -- Thresholds ---------------------------------------------------------------

# Narrative strength (0-100 composite from NarrativeComponents)
STRONG_NARRATIVE = 50.0
VERY_STRONG_NARRATIVE = 70.0
WEAK_NARRATIVE = 20.0

# Filing adoption score (0-1)
WEAK_ADOPTION = 0.2
VERY_WEAK_ADOPTION = 0.1
STRONG_ADOPTION = 0.6

# Drift: which dimensions are considered adverse
ADVERSE_DIMENSIONS = frozenset({"risk", "regulatory"})

# Drift: word count increase threshold for adverse signal
ADVERSE_WORD_INCREASE = 100

# Drift: minimum magnitude for contradictory shrinkage signal
CONTRADICTORY_MIN_MAGNITUDE = 0.15

# Temporal consistency threshold for lagging adoption detection
LAGGING_TEMPORAL_MIN = 0.3
LAGGING_TEMPORAL_MAX = 0.7


# -- Alert dataclass ----------------------------------------------------------


@dataclass(frozen=True)
class DivergenceAlert:
    """A divergence alert between narrative and filing signals.

    Attributes:
        issuer_concept_id: Canonical issuer concept.
        theme_concept_id: Canonical theme concept.
        reason: Why the divergence was flagged.
        severity: "critical", "warning", or "info".
        title: Short summary for display.
        summary: Human-readable explanation.
        evidence: Structured evidence for UI/audit inspection.
        created_at: When this alert was generated.
    """

    issuer_concept_id: str
    theme_concept_id: str
    reason: str
    severity: str
    title: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity {self.severity!r}. Must be one of {sorted(VALID_SEVERITIES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for publication payloads."""
        return {
            "issuer_concept_id": self.issuer_concept_id,
            "theme_concept_id": self.theme_concept_id,
            "reason": self.reason,
            "severity": self.severity,
            "title": self.title,
            "summary": self.summary,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
        }


# -- Individual check functions (stateless) -----------------------------------


def _check_narrative_without_filing(
    issuer_concept_id: str,
    theme_concept_id: str,
    narrative_strength: float,
    adoption: FilingAdoptionScore,
    now: datetime,
) -> DivergenceAlert | None:
    """Strong narrative, weak filing adoption → hype risk.

    Critical if very strong narrative + very weak adoption.
    Warning if strong narrative + weak adoption.
    """
    if narrative_strength < STRONG_NARRATIVE:
        return None
    if adoption.score >= WEAK_ADOPTION:
        return None

    if narrative_strength >= VERY_STRONG_NARRATIVE and adoption.score < VERY_WEAK_ADOPTION:
        severity = "critical"
    else:
        severity = "warning"

    gap = narrative_strength / 100.0 - adoption.score
    return DivergenceAlert(
        issuer_concept_id=issuer_concept_id,
        theme_concept_id=theme_concept_id,
        reason=DivergenceReason.NARRATIVE_WITHOUT_FILING.value,
        severity=severity,
        title="Strong narrative without filing confirmation",
        summary=(
            f"Narrative strength {narrative_strength:.0f}/100 but filing "
            f"adoption only {adoption.score:.2f}. Theme may not be "
            f"reflected in operational disclosure."
        ),
        evidence={
            "narrative_strength": narrative_strength,
            "adoption_score": adoption.score,
            "adoption_section_signals": len(adoption.section_signals),
            "adoption_fact_signals": len(adoption.fact_signals),
            "gap": round(gap, 4),
        },
        created_at=now,
    )


def _check_filing_without_narrative(
    issuer_concept_id: str,
    theme_concept_id: str,
    narrative_strength: float,
    adoption: FilingAdoptionScore,
    now: datetime,
) -> DivergenceAlert | None:
    """Strong filing adoption, weak narrative → under-appreciated.

    Informational: filings show operational reality that narrative
    hasn't yet recognized.
    """
    if narrative_strength >= WEAK_NARRATIVE:
        return None
    if adoption.score < STRONG_ADOPTION:
        return None

    return DivergenceAlert(
        issuer_concept_id=issuer_concept_id,
        theme_concept_id=theme_concept_id,
        reason=DivergenceReason.FILING_WITHOUT_NARRATIVE.value,
        severity="info",
        title="Filing adoption without narrative recognition",
        summary=(
            f"Filing adoption {adoption.score:.2f} but narrative strength "
            f"only {narrative_strength:.0f}/100. Operational disclosure "
            f"may be under-appreciated by the market."
        ),
        evidence={
            "narrative_strength": narrative_strength,
            "adoption_score": adoption.score,
            "adoption_section_signals": len(adoption.section_signals),
            "adoption_fact_signals": len(adoption.fact_signals),
            "filing_count": adoption.filing_count,
        },
        created_at=now,
    )


def _check_adverse_drift(
    issuer_concept_id: str,
    theme_concept_id: str,
    drift: DriftDecomposition,
    now: datetime,
) -> DivergenceAlert | None:
    """Unusual drift in risk/regulatory filing sections.

    Fires when risk or regulatory dimensions show unusual change
    with increasing word count (more disclosure = more risk exposure).
    Does not require narrative comparison.
    """
    adverse_hits: list[dict[str, Any]] = []
    for dim in drift.dimensions:
        if dim.dimension not in ADVERSE_DIMENSIONS:
            continue
        if not dim.is_unusual:
            continue
        if dim.word_count_delta < ADVERSE_WORD_INCREASE:
            continue
        adverse_hits.append(
            {
                "dimension": dim.dimension,
                "magnitude": dim.magnitude,
                "z_score": dim.z_score,
                "word_count_delta": dim.word_count_delta,
                "section_names": dim.section_names,
            }
        )

    if not adverse_hits:
        return None

    dims = ", ".join(h["dimension"] for h in adverse_hits)
    severity = "critical" if any(h["z_score"] >= 2.5 for h in adverse_hits) else "warning"

    return DivergenceAlert(
        issuer_concept_id=issuer_concept_id,
        theme_concept_id=theme_concept_id,
        reason=DivergenceReason.ADVERSE_DRIFT.value,
        severity=severity,
        title=f"Adverse filing drift in {dims}",
        summary=(
            f"Unusual increase in {dims} disclosure compared to peers. "
            f"Filing sections grew significantly, signaling potential "
            f"new risk or regulatory exposure."
        ),
        evidence={
            "adverse_dimensions": adverse_hits,
            "base_accession": drift.base_accession,
            "target_accession": drift.target_accession,
        },
        created_at=now,
    )


def _check_contradictory_drift(
    issuer_concept_id: str,
    theme_concept_id: str,
    narrative_strength: float,
    drift: DriftDecomposition,
    now: datetime,
) -> DivergenceAlert | None:
    """Strong narrative but filings shrinking in strategy/capex.

    Fires when narrative says growth but the issuer is reducing
    disclosure in strategy or capex sections (negative word_count_delta
    with meaningful magnitude).
    """
    if narrative_strength < STRONG_NARRATIVE:
        return None

    shrinking: list[dict[str, Any]] = []
    for dim in drift.dimensions:
        if dim.dimension not in ("strategy", "capex"):
            continue
        if dim.magnitude < CONTRADICTORY_MIN_MAGNITUDE:
            continue
        if dim.word_count_delta >= 0:
            continue
        shrinking.append(
            {
                "dimension": dim.dimension,
                "magnitude": dim.magnitude,
                "word_count_delta": dim.word_count_delta,
                "section_names": dim.section_names,
            }
        )

    if not shrinking:
        return None

    dims = ", ".join(h["dimension"] for h in shrinking)
    return DivergenceAlert(
        issuer_concept_id=issuer_concept_id,
        theme_concept_id=theme_concept_id,
        reason=DivergenceReason.CONTRADICTORY_DRIFT.value,
        severity="warning",
        title=f"Narrative contradicted by {dims} filing reduction",
        summary=(
            f"Narrative strength {narrative_strength:.0f}/100 but issuer "
            f"reduced disclosure in {dims} sections. Filing changes "
            f"contradict narrative momentum."
        ),
        evidence={
            "narrative_strength": narrative_strength,
            "shrinking_dimensions": shrinking,
            "base_accession": drift.base_accession,
            "target_accession": drift.target_accession,
        },
        created_at=now,
    )


def _check_lagging_adoption(
    issuer_concept_id: str,
    theme_concept_id: str,
    narrative_strength: float,
    adoption: FilingAdoptionScore,
    now: datetime,
) -> DivergenceAlert | None:
    """Strong narrative with partial but growing filing adoption.

    Informational: theme is starting to appear in filings but
    hasn't fully caught up. Temporal consistency between thresholds
    indicates adoption is growing across periods.
    """
    if narrative_strength < STRONG_NARRATIVE:
        return None
    if adoption.score >= WEAK_ADOPTION:
        return None
    temporal = adoption.breakdown.temporal_consistency
    if temporal < LAGGING_TEMPORAL_MIN or temporal > LAGGING_TEMPORAL_MAX:
        return None
    if adoption.period_count < 2:
        return None

    return DivergenceAlert(
        issuer_concept_id=issuer_concept_id,
        theme_concept_id=theme_concept_id,
        reason=DivergenceReason.LAGGING_ADOPTION.value,
        severity="info",
        title="Filing adoption lagging behind narrative",
        summary=(
            f"Theme appears in {adoption.periods_with_signal}/{adoption.period_count} "
            f"filing periods (temporal consistency {temporal:.0%}). "
            f"Adoption may be catching up to narrative strength "
            f"({narrative_strength:.0f}/100)."
        ),
        evidence={
            "narrative_strength": narrative_strength,
            "adoption_score": adoption.score,
            "temporal_consistency": temporal,
            "periods_with_signal": adoption.periods_with_signal,
            "period_count": adoption.period_count,
        },
        created_at=now,
    )


# -- Combiner -----------------------------------------------------------------


def check_divergence(
    issuer_concept_id: str,
    theme_concept_id: str,
    narrative_strength: float,
    adoption: FilingAdoptionScore,
    drift: DriftDecomposition | None = None,
    *,
    now: datetime | None = None,
) -> list[DivergenceAlert]:
    """Check for narrative/filing divergence and generate alerts.

    Runs all divergence checks and returns any that fire. Multiple
    alerts can fire simultaneously (e.g., hype + adverse drift).

    Args:
        issuer_concept_id: Canonical issuer concept ID.
        theme_concept_id: Canonical theme concept ID.
        narrative_strength: Composite narrative score (0-100).
        adoption: Filing adoption score for this pair.
        drift: Drift decomposition (None if no consecutive filings).
        now: Current time for alert timestamps.

    Returns:
        List of DivergenceAlert objects (empty if no divergence).
    """
    if now is None:
        now = datetime.now(UTC)

    alerts: list[DivergenceAlert] = []

    # Lagging adoption is a more specific diagnosis than narrative-
    # without-filing (same preconditions plus temporal growth signal).
    # Check it first; if it fires, skip the blunter hype-risk alert.
    lagging = _check_lagging_adoption(
        issuer_concept_id,
        theme_concept_id,
        narrative_strength,
        adoption,
        now,
    )
    if lagging is not None:
        alerts.append(lagging)
    else:
        alert = _check_narrative_without_filing(
            issuer_concept_id,
            theme_concept_id,
            narrative_strength,
            adoption,
            now,
        )
        if alert is not None:
            alerts.append(alert)

    alert = _check_filing_without_narrative(
        issuer_concept_id,
        theme_concept_id,
        narrative_strength,
        adoption,
        now,
    )
    if alert is not None:
        alerts.append(alert)

    # Drift-based checks (require consecutive filings)
    if drift is not None:
        alert = _check_adverse_drift(
            issuer_concept_id,
            theme_concept_id,
            drift,
            now,
        )
        if alert is not None:
            alerts.append(alert)

        alert = _check_contradictory_drift(
            issuer_concept_id,
            theme_concept_id,
            narrative_strength,
            drift,
            now,
        )
        if alert is not None:
            alerts.append(alert)

    return alerts
