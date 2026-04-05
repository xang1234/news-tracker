"""Publish filing lane outputs keyed by manifest.

Orchestrates filing lane publication: transforms adoption scores
and divergence alerts into publishable payloads, builds per-issuer
divergence summaries, checks lane health, and produces a result
for manifest assembly.

This is the filing-lane analog of src/narrative/publisher.py.

Publication flow:
    1. Check lane health — abort if BLOCKED
    2. Build adoption payloads (strip internal signal lists)
    3. Build divergence payloads from alerts
    4. Build per-issuer divergence summaries
    5. Return publishable result for manifest inclusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.filing.adoption import FilingAdoptionScore
from src.filing.divergence import DivergenceAlert
from src.publish.lane_health import LaneHealthStatus, PublishReadiness


# -- Payload dataclasses (publishable, no internal state) --------------------


@dataclass(frozen=True)
class AdoptionPayload:
    """Publishable filing adoption payload.

    Strips internal signal lists from FilingAdoptionScore, keeping
    only the score, breakdown, and metadata needed by consumers.

    Attributes:
        issuer_concept_id: Canonical issuer concept.
        theme_concept_id: Canonical theme concept.
        score: Composite adoption score (0-1).
        section_coverage: Fraction of weighted sections with matches.
        section_depth: Density-weighted depth of mentions.
        fact_alignment: Fraction of theme XBRL concepts found.
        temporal_consistency: Fraction of periods showing adoption.
        section_signal_count: How many sections had matches.
        fact_signal_count: How many XBRL facts matched.
        filing_count: Distinct filings evaluated.
        period_count: Distinct filing periods.
        periods_with_signal: Periods that showed adoption.
    """

    issuer_concept_id: str
    theme_concept_id: str
    score: float
    section_coverage: float
    section_depth: float
    fact_alignment: float
    temporal_consistency: float
    section_signal_count: int
    fact_signal_count: int
    filing_count: int
    period_count: int
    periods_with_signal: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "issuer_concept_id": self.issuer_concept_id,
            "theme_concept_id": self.theme_concept_id,
            "score": round(self.score, 4),
            "section_coverage": round(self.section_coverage, 4),
            "section_depth": round(self.section_depth, 4),
            "fact_alignment": round(self.fact_alignment, 4),
            "temporal_consistency": round(self.temporal_consistency, 4),
            "section_signal_count": self.section_signal_count,
            "fact_signal_count": self.fact_signal_count,
            "filing_count": self.filing_count,
            "period_count": self.period_count,
            "periods_with_signal": self.periods_with_signal,
        }


@dataclass(frozen=True)
class DivergencePayload:
    """Publishable divergence alert payload.

    Attributes:
        issuer_concept_id: Canonical issuer concept.
        theme_concept_id: Canonical theme concept.
        reason: Divergence reason code.
        severity: Alert severity.
        title: Short summary.
        summary: Human-readable explanation.
        evidence: Structured evidence for audit.
    """

    issuer_concept_id: str
    theme_concept_id: str
    reason: str
    severity: str
    title: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issuer_concept_id": self.issuer_concept_id,
            "theme_concept_id": self.theme_concept_id,
            "reason": self.reason,
            "severity": self.severity,
            "title": self.title,
            "summary": self.summary,
            "evidence": self.evidence,
        }


# -- Issuer divergence summary -----------------------------------------------


@dataclass(frozen=True)
class IssuerDivergenceSummary:
    """Per-issuer divergence summary across all evaluated themes.

    Answers: "What is the overall divergence picture for this issuer?"

    Attributes:
        issuer_concept_id: Canonical issuer concept.
        theme_count: Themes evaluated for this issuer.
        max_adoption: Highest adoption score across themes.
        min_adoption: Lowest adoption score across themes.
        avg_adoption: Mean adoption score.
        alert_count: Total divergence alerts.
        critical_count: Critical-severity alerts.
        warning_count: Warning-severity alerts.
        reason_counts: Alert count per reason code.
        contributing_themes: Which themes contributed.
    """

    issuer_concept_id: str
    theme_count: int
    max_adoption: float
    min_adoption: float
    avg_adoption: float
    alert_count: int
    critical_count: int
    warning_count: int
    reason_counts: dict[str, int] = field(default_factory=dict)
    contributing_themes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issuer_concept_id": self.issuer_concept_id,
            "theme_count": self.theme_count,
            "max_adoption": round(self.max_adoption, 4),
            "min_adoption": round(self.min_adoption, 4),
            "avg_adoption": round(self.avg_adoption, 4),
            "alert_count": self.alert_count,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "reason_counts": self.reason_counts,
            "contributing_themes": self.contributing_themes,
        }


# -- Publication result -------------------------------------------------------


@dataclass
class FilingPublicationResult:
    """Result of a filing lane publication attempt.

    Attributes:
        published: Whether publication succeeded.
        lane_health: The health check result.
        adoption_payloads: Publishable adoption scores.
        divergence_payloads: Publishable divergence alerts.
        issuer_summaries: Per-issuer divergence summaries.
        object_count: Total publishable objects produced.
        block_reason: Why publication was blocked (if applicable).
    """

    published: bool
    lane_health: LaneHealthStatus
    adoption_payloads: list[AdoptionPayload] = field(default_factory=list)
    divergence_payloads: list[DivergencePayload] = field(default_factory=list)
    issuer_summaries: list[IssuerDivergenceSummary] = field(
        default_factory=list
    )
    object_count: int = 0
    block_reason: str | None = None


# -- Payload builders (stateless) --------------------------------------------


def build_adoption_payload(
    adoption: FilingAdoptionScore,
) -> AdoptionPayload:
    """Convert a FilingAdoptionScore into a publishable payload.

    Strips internal signal lists, keeping only the score and metadata.
    """
    return AdoptionPayload(
        issuer_concept_id=adoption.issuer_concept_id,
        theme_concept_id=adoption.theme_concept_id,
        score=adoption.score,
        section_coverage=adoption.breakdown.section_coverage,
        section_depth=adoption.breakdown.section_depth,
        fact_alignment=adoption.breakdown.fact_alignment,
        temporal_consistency=adoption.breakdown.temporal_consistency,
        section_signal_count=len(adoption.section_signals),
        fact_signal_count=len(adoption.fact_signals),
        filing_count=adoption.filing_count,
        period_count=adoption.period_count,
        periods_with_signal=adoption.periods_with_signal,
    )


def build_divergence_payload(
    alert: DivergenceAlert,
) -> DivergencePayload:
    """Convert a DivergenceAlert into a publishable payload."""
    return DivergencePayload(
        issuer_concept_id=alert.issuer_concept_id,
        theme_concept_id=alert.theme_concept_id,
        reason=alert.reason,
        severity=alert.severity,
        title=alert.title,
        summary=alert.summary,
        evidence=dict(alert.evidence),
    )


# -- Summary builder (stateless) ---------------------------------------------


def build_issuer_summaries(
    adoptions: list[FilingAdoptionScore],
    alerts: list[DivergenceAlert],
) -> list[IssuerDivergenceSummary]:
    """Build per-issuer divergence summaries.

    Groups adoption scores and alerts by issuer, aggregates metrics.
    """
    def _new_entry() -> dict[str, Any]:
        return {
            "themes": set(),
            "scores": [],
            "alert_count": 0,
            "critical_count": 0,
            "warning_count": 0,
            "reason_counts": {},
        }

    issuer_data: dict[str, dict[str, Any]] = {}

    for adoption in adoptions:
        iid = adoption.issuer_concept_id
        d = issuer_data.setdefault(iid, _new_entry())
        d["themes"].add(adoption.theme_concept_id)
        d["scores"].append(adoption.score)

    for alert in alerts:
        iid = alert.issuer_concept_id
        d = issuer_data.setdefault(iid, _new_entry())
        d["themes"].add(alert.theme_concept_id)
        d["alert_count"] += 1
        if alert.severity == "critical":
            d["critical_count"] += 1
        elif alert.severity == "warning":
            d["warning_count"] += 1
        d["reason_counts"][alert.reason] = (
            d["reason_counts"].get(alert.reason, 0) + 1
        )

    summaries: list[IssuerDivergenceSummary] = []
    for iid, d in sorted(issuer_data.items()):
        scores = d["scores"]
        avg = sum(scores) / len(scores) if scores else 0.0
        summaries.append(
            IssuerDivergenceSummary(
                issuer_concept_id=iid,
                theme_count=len(d["themes"]),
                max_adoption=max(scores) if scores else 0.0,
                min_adoption=min(scores) if scores else 0.0,
                avg_adoption=avg,
                alert_count=d["alert_count"],
                critical_count=d["critical_count"],
                warning_count=d["warning_count"],
                reason_counts=d["reason_counts"],
                contributing_themes=sorted(d["themes"]),
            )
        )
    return summaries


# -- Publisher ----------------------------------------------------------------


def prepare_filing_publication(
    adoptions: list[FilingAdoptionScore],
    alerts: list[DivergenceAlert],
    lane_health: LaneHealthStatus,
    *,
    now: datetime | None = None,
) -> FilingPublicationResult:
    """Prepare filing lane outputs for manifest publication.

    Checks lane health, builds publishable payloads and issuer-level
    summaries. Returns a result the caller can use to create manifest
    objects.

    Does NOT persist anything — the caller handles manifest creation
    and object insertion.

    Args:
        adoptions: Filing adoption scores for issuer-theme pairs.
        alerts: Divergence alerts from check_divergence().
        lane_health: Pre-computed lane health status.
        now: Current time (unused but kept for API symmetry).

    Returns:
        FilingPublicationResult with payloads, summaries, and counts.
    """
    if lane_health.readiness == PublishReadiness.BLOCKED:
        return FilingPublicationResult(
            published=False,
            lane_health=lane_health,
            block_reason=lane_health.format_block_reason(),
        )

    adoption_payloads = [build_adoption_payload(a) for a in adoptions]
    divergence_payloads = [build_divergence_payload(a) for a in alerts]
    issuer_summaries = build_issuer_summaries(adoptions, alerts)

    object_count = (
        len(adoption_payloads)
        + len(divergence_payloads)
        + len(issuer_summaries)
    )

    return FilingPublicationResult(
        published=True,
        lane_health=lane_health,
        adoption_payloads=adoption_payloads,
        divergence_payloads=divergence_payloads,
        issuer_summaries=issuer_summaries,
        object_count=object_count,
    )
