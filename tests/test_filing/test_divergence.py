"""Tests for narrative/filing divergence alerts.

Verifies that divergence reason codes, severity levels, and
explanation evidence are correctly produced for all five
divergence scenarios.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.filing.adoption import (
    AdoptionBreakdown,
    FactSignal,
    FilingAdoptionScore,
    SectionSignal,
)
from src.filing.divergence import (
    STRONG_ADOPTION,
    STRONG_NARRATIVE,
    WEAK_ADOPTION,
    WEAK_NARRATIVE,
    DivergenceAlert,
    DivergenceReason,
    _check_adverse_drift,
    _check_contradictory_drift,
    _check_filing_without_narrative,
    _check_lagging_adoption,
    _check_narrative_without_filing,
    check_divergence,
)
from src.filing.drift import DimensionDrift, DriftDecomposition

NOW = datetime(2026, 4, 1, tzinfo=UTC)

ISSUER = "concept_issuer_abc123"
THEME = "concept_theme_xyz789"


# -- Helpers ---------------------------------------------------------------


def _make_adoption(
    score: float = 0.5,
    section_coverage: float = 0.5,
    section_depth: float = 0.3,
    fact_alignment: float = 0.4,
    temporal_consistency: float = 0.6,
    section_signals: int = 2,
    fact_signals: int = 1,
    filing_count: int = 3,
    period_count: int = 4,
    periods_with_signal: int = 3,
) -> FilingAdoptionScore:
    return FilingAdoptionScore(
        issuer_concept_id=ISSUER,
        theme_concept_id=THEME,
        score=score,
        breakdown=AdoptionBreakdown(
            section_coverage=section_coverage,
            section_depth=section_depth,
            fact_alignment=fact_alignment,
            temporal_consistency=temporal_consistency,
        ),
        section_signals=[
            SectionSignal(section_id=f"s{i}", section_name=f"Section {i}")
            for i in range(section_signals)
        ],
        fact_signals=[
            FactSignal(concept_name=f"Concept{i}", value="1B") for i in range(fact_signals)
        ],
        filing_count=filing_count,
        period_count=period_count,
        periods_with_signal=periods_with_signal,
        computed_at=NOW,
    )


def _make_drift(
    dimensions: list[DimensionDrift] | None = None,
    unusual_dimensions: list[str] | None = None,
) -> DriftDecomposition:
    if dimensions is None:
        dimensions = [
            DimensionDrift(
                dimension=d,
                magnitude=0.1,
                word_count_delta=10,
                peer_mean=0.1,
                peer_std=0.05,
                z_score=0.0,
                is_unusual=False,
            )
            for d in ["strategy", "risk", "capex", "customer_supplier", "regulatory"]
        ]
    return DriftDecomposition(
        issuer_concept_id=ISSUER,
        base_accession="acc-001",
        target_accession="acc-002",
        dimensions=dimensions,
        unusual_dimensions=unusual_dimensions or [],
        computed_at=NOW,
    )


def _dim(
    dimension: str = "risk",
    magnitude: float = 0.3,
    word_count_delta: int = 200,
    z_score: float = 2.0,
    is_unusual: bool = True,
    section_names: list[str] | None = None,
) -> DimensionDrift:
    return DimensionDrift(
        dimension=dimension,
        magnitude=magnitude,
        word_count_delta=word_count_delta,
        peer_mean=0.1,
        peer_std=0.05,
        z_score=z_score,
        is_unusual=is_unusual,
        section_names=section_names or ["risk factors"],
    )


# -- Narrative without filing tests ----------------------------------------


class TestNarrativeWithoutFiling:
    """Strong narrative, weak filing adoption (hype risk)."""

    def test_critical_very_strong_very_weak(self) -> None:
        alert = _check_narrative_without_filing(
            ISSUER,
            THEME,
            80.0,
            _make_adoption(score=0.05),
            NOW,
        )
        assert alert is not None
        assert alert.reason == DivergenceReason.NARRATIVE_WITHOUT_FILING.value
        assert alert.severity == "critical"
        assert alert.evidence["narrative_strength"] == 80.0
        assert alert.evidence["adoption_score"] == 0.05

    def test_warning_strong_weak(self) -> None:
        alert = _check_narrative_without_filing(
            ISSUER,
            THEME,
            55.0,
            _make_adoption(score=0.15),
            NOW,
        )
        assert alert is not None
        assert alert.severity == "warning"

    def test_no_alert_weak_narrative(self) -> None:
        alert = _check_narrative_without_filing(
            ISSUER,
            THEME,
            30.0,
            _make_adoption(score=0.05),
            NOW,
        )
        assert alert is None

    def test_no_alert_strong_adoption(self) -> None:
        alert = _check_narrative_without_filing(
            ISSUER,
            THEME,
            80.0,
            _make_adoption(score=0.3),
            NOW,
        )
        assert alert is None

    def test_boundary_narrative_exactly_strong(self) -> None:
        alert = _check_narrative_without_filing(
            ISSUER,
            THEME,
            STRONG_NARRATIVE,
            _make_adoption(score=0.05),
            NOW,
        )
        assert alert is not None

    def test_boundary_adoption_exactly_weak(self) -> None:
        """Adoption at WEAK_ADOPTION threshold is NOT weak — no alert."""
        alert = _check_narrative_without_filing(
            ISSUER,
            THEME,
            60.0,
            _make_adoption(score=WEAK_ADOPTION),
            NOW,
        )
        assert alert is None

    def test_evidence_gap(self) -> None:
        alert = _check_narrative_without_filing(
            ISSUER,
            THEME,
            60.0,
            _make_adoption(score=0.1),
            NOW,
        )
        assert alert is not None
        assert alert.evidence["gap"] == round(0.6 - 0.1, 4)

    def test_signal_counts_in_evidence(self) -> None:
        alert = _check_narrative_without_filing(
            ISSUER,
            THEME,
            60.0,
            _make_adoption(score=0.1, section_signals=3, fact_signals=2),
            NOW,
        )
        assert alert is not None
        assert alert.evidence["adoption_section_signals"] == 3
        assert alert.evidence["adoption_fact_signals"] == 2


# -- Filing without narrative tests ----------------------------------------


class TestFilingWithoutNarrative:
    """Strong filing adoption, weak narrative (under-appreciated)."""

    def test_info_alert(self) -> None:
        alert = _check_filing_without_narrative(
            ISSUER,
            THEME,
            10.0,
            _make_adoption(score=0.7),
            NOW,
        )
        assert alert is not None
        assert alert.reason == DivergenceReason.FILING_WITHOUT_NARRATIVE.value
        assert alert.severity == "info"

    def test_no_alert_strong_narrative(self) -> None:
        alert = _check_filing_without_narrative(
            ISSUER,
            THEME,
            30.0,
            _make_adoption(score=0.7),
            NOW,
        )
        assert alert is None

    def test_no_alert_weak_adoption(self) -> None:
        alert = _check_filing_without_narrative(
            ISSUER,
            THEME,
            10.0,
            _make_adoption(score=0.4),
            NOW,
        )
        assert alert is None

    def test_boundary_narrative_exactly_weak(self) -> None:
        """Narrative at WEAK_NARRATIVE is NOT weak — no alert."""
        alert = _check_filing_without_narrative(
            ISSUER,
            THEME,
            WEAK_NARRATIVE,
            _make_adoption(score=0.7),
            NOW,
        )
        assert alert is None

    def test_boundary_adoption_exactly_strong(self) -> None:
        alert = _check_filing_without_narrative(
            ISSUER,
            THEME,
            10.0,
            _make_adoption(score=STRONG_ADOPTION),
            NOW,
        )
        assert alert is not None


# -- Adverse drift tests ---------------------------------------------------


class TestAdverseDrift:
    """Unusual drift in risk/regulatory sections."""

    def test_risk_drift(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("risk", magnitude=0.5, word_count_delta=300, z_score=2.5),
                _dim("strategy", magnitude=0.1, word_count_delta=50, is_unusual=False),
                _dim("capex", magnitude=0.05, word_count_delta=20, is_unusual=False),
                _dim("customer_supplier", magnitude=0.0, word_count_delta=0, is_unusual=False),
                _dim("regulatory", magnitude=0.1, word_count_delta=50, is_unusual=False),
            ]
        )
        alert = _check_adverse_drift(ISSUER, THEME, drift, NOW)
        assert alert is not None
        assert alert.reason == DivergenceReason.ADVERSE_DRIFT.value
        assert alert.severity == "critical"  # z_score >= 2.5

    def test_regulatory_drift(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("risk", magnitude=0.1, is_unusual=False, word_count_delta=10),
                _dim("strategy", magnitude=0.1, is_unusual=False, word_count_delta=10),
                _dim("capex", magnitude=0.1, is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", magnitude=0.1, is_unusual=False, word_count_delta=10),
                _dim("regulatory", magnitude=0.4, word_count_delta=200, z_score=2.0),
            ]
        )
        alert = _check_adverse_drift(ISSUER, THEME, drift, NOW)
        assert alert is not None
        assert "regulatory" in alert.title

    def test_warning_below_critical_zscore(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("risk", magnitude=0.3, word_count_delta=150, z_score=1.8),
                _dim("strategy", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_adverse_drift(ISSUER, THEME, drift, NOW)
        assert alert is not None
        assert alert.severity == "warning"

    def test_no_alert_not_unusual(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("risk", magnitude=0.3, word_count_delta=200, is_unusual=False),
                _dim("strategy", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_adverse_drift(ISSUER, THEME, drift, NOW)
        assert alert is None

    def test_no_alert_low_word_count(self) -> None:
        """Unusual but trivial word count change → no alert."""
        drift = _make_drift(
            dimensions=[
                _dim("risk", magnitude=0.3, word_count_delta=50, z_score=2.0),
                _dim("strategy", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_adverse_drift(ISSUER, THEME, drift, NOW)
        assert alert is None

    def test_no_alert_non_adverse_dimension(self) -> None:
        """Unusual drift in strategy (not risk/regulatory) → no alert."""
        drift = _make_drift(
            dimensions=[
                _dim("strategy", magnitude=0.5, word_count_delta=300, z_score=2.5),
                _dim("risk", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_adverse_drift(ISSUER, THEME, drift, NOW)
        assert alert is None

    def test_both_adverse_dimensions(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("risk", magnitude=0.4, word_count_delta=200, z_score=2.0),
                _dim("strategy", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", magnitude=0.3, word_count_delta=150, z_score=1.8),
            ]
        )
        alert = _check_adverse_drift(ISSUER, THEME, drift, NOW)
        assert alert is not None
        assert len(alert.evidence["adverse_dimensions"]) == 2

    def test_evidence_contains_accessions(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("risk", word_count_delta=200),
                _dim("strategy", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_adverse_drift(ISSUER, THEME, drift, NOW)
        assert alert is not None
        assert alert.evidence["base_accession"] == "acc-001"
        assert alert.evidence["target_accession"] == "acc-002"


# -- Contradictory drift tests ---------------------------------------------


class TestContradictoryDrift:
    """Strong narrative but filings shrinking in strategy/capex."""

    def test_strategy_shrinking(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("strategy", magnitude=0.2, word_count_delta=-500),
                _dim("risk", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_contradictory_drift(ISSUER, THEME, 60.0, drift, NOW)
        assert alert is not None
        assert alert.reason == DivergenceReason.CONTRADICTORY_DRIFT.value
        assert alert.severity == "warning"

    def test_capex_shrinking(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("strategy", is_unusual=False, word_count_delta=10),
                _dim("risk", is_unusual=False, word_count_delta=10),
                _dim("capex", magnitude=0.25, word_count_delta=-300),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_contradictory_drift(ISSUER, THEME, 55.0, drift, NOW)
        assert alert is not None
        assert "capex" in alert.title

    def test_no_alert_weak_narrative(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("strategy", magnitude=0.3, word_count_delta=-500),
                _dim("risk", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_contradictory_drift(ISSUER, THEME, 30.0, drift, NOW)
        assert alert is None

    def test_no_alert_growing_sections(self) -> None:
        """Sections growing = consistent with narrative, not contradictory."""
        drift = _make_drift(
            dimensions=[
                _dim("strategy", magnitude=0.3, word_count_delta=500),
                _dim("risk", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_contradictory_drift(ISSUER, THEME, 60.0, drift, NOW)
        assert alert is None

    def test_no_alert_small_magnitude(self) -> None:
        drift = _make_drift(
            dimensions=[
                _dim("strategy", magnitude=0.05, word_count_delta=-50),
                _dim("risk", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_contradictory_drift(ISSUER, THEME, 60.0, drift, NOW)
        assert alert is None

    def test_no_alert_risk_shrinking(self) -> None:
        """Risk section shrinking is not contradictory — only strategy/capex."""
        drift = _make_drift(
            dimensions=[
                _dim("strategy", is_unusual=False, word_count_delta=10),
                _dim("risk", magnitude=0.3, word_count_delta=-500),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alert = _check_contradictory_drift(ISSUER, THEME, 60.0, drift, NOW)
        assert alert is None


# -- Lagging adoption tests ------------------------------------------------


class TestLaggingAdoption:
    """Moderate narrative with partial temporal adoption."""

    def test_lagging_alert(self) -> None:
        adoption = _make_adoption(
            score=0.15,
            temporal_consistency=0.5,
            period_count=4,
            periods_with_signal=2,
        )
        alert = _check_lagging_adoption(ISSUER, THEME, 55.0, adoption, NOW)
        assert alert is not None
        assert alert.reason == DivergenceReason.LAGGING_ADOPTION.value
        assert alert.severity == "info"

    def test_no_alert_strong_adoption(self) -> None:
        adoption = _make_adoption(score=0.3, temporal_consistency=0.5)
        alert = _check_lagging_adoption(ISSUER, THEME, 55.0, adoption, NOW)
        assert alert is None

    def test_no_alert_weak_narrative(self) -> None:
        adoption = _make_adoption(score=0.1, temporal_consistency=0.5)
        alert = _check_lagging_adoption(ISSUER, THEME, 30.0, adoption, NOW)
        assert alert is None

    def test_no_alert_low_temporal(self) -> None:
        """Below LAGGING_TEMPORAL_MIN → not growing enough."""
        adoption = _make_adoption(
            score=0.1,
            temporal_consistency=0.2,
            period_count=5,
            periods_with_signal=1,
        )
        alert = _check_lagging_adoption(ISSUER, THEME, 55.0, adoption, NOW)
        assert alert is None

    def test_no_alert_high_temporal(self) -> None:
        """Above LAGGING_TEMPORAL_MAX → already adopted, not lagging."""
        adoption = _make_adoption(
            score=0.15,
            temporal_consistency=0.8,
            period_count=5,
            periods_with_signal=4,
        )
        alert = _check_lagging_adoption(ISSUER, THEME, 55.0, adoption, NOW)
        assert alert is None

    def test_no_alert_single_period(self) -> None:
        """Need at least 2 periods to assess temporal trend."""
        adoption = _make_adoption(
            score=0.1,
            temporal_consistency=0.5,
            period_count=1,
            periods_with_signal=1,
        )
        alert = _check_lagging_adoption(ISSUER, THEME, 55.0, adoption, NOW)
        assert alert is None

    def test_evidence_contains_temporal_info(self) -> None:
        adoption = _make_adoption(
            score=0.12,
            temporal_consistency=0.5,
            period_count=4,
            periods_with_signal=2,
        )
        alert = _check_lagging_adoption(ISSUER, THEME, 60.0, adoption, NOW)
        assert alert is not None
        assert alert.evidence["temporal_consistency"] == 0.5
        assert alert.evidence["periods_with_signal"] == 2
        assert alert.evidence["period_count"] == 4


# -- check_divergence combiner tests --------------------------------------


class TestCheckDivergence:
    """Full divergence check pipeline."""

    def test_hype_scenario(self) -> None:
        """Strong narrative, weak adoption, no temporal growth."""
        alerts = check_divergence(
            ISSUER,
            THEME,
            75.0,
            _make_adoption(score=0.05, temporal_consistency=0.1, period_count=1),
            now=NOW,
        )
        reasons = {a.reason for a in alerts}
        assert DivergenceReason.NARRATIVE_WITHOUT_FILING.value in reasons

    def test_under_appreciated_scenario(self) -> None:
        """Weak narrative, strong adoption."""
        alerts = check_divergence(
            ISSUER,
            THEME,
            10.0,
            _make_adoption(score=0.7),
            now=NOW,
        )
        reasons = {a.reason for a in alerts}
        assert DivergenceReason.FILING_WITHOUT_NARRATIVE.value in reasons

    def test_no_divergence(self) -> None:
        """Moderate narrative, moderate adoption — no alerts."""
        alerts = check_divergence(
            ISSUER,
            THEME,
            40.0,
            _make_adoption(score=0.4),
            now=NOW,
        )
        assert alerts == []

    def test_multiple_alerts(self) -> None:
        """Hype + adverse drift can fire together."""
        drift = _make_drift(
            dimensions=[
                _dim("risk", magnitude=0.5, word_count_delta=300, z_score=2.5),
                _dim("strategy", is_unusual=False, word_count_delta=10),
                _dim("capex", is_unusual=False, word_count_delta=10),
                _dim("customer_supplier", is_unusual=False, word_count_delta=10),
                _dim("regulatory", is_unusual=False, word_count_delta=10),
            ]
        )
        alerts = check_divergence(
            ISSUER,
            THEME,
            75.0,
            _make_adoption(score=0.05, temporal_consistency=0.1, period_count=1),
            drift=drift,
            now=NOW,
        )
        reasons = {a.reason for a in alerts}
        assert DivergenceReason.NARRATIVE_WITHOUT_FILING.value in reasons
        assert DivergenceReason.ADVERSE_DRIFT.value in reasons

    def test_drift_checks_skipped_without_drift(self) -> None:
        """No drift provided → no adverse/contradictory checks."""
        alerts = check_divergence(
            ISSUER,
            THEME,
            75.0,
            _make_adoption(score=0.05, temporal_consistency=0.1, period_count=1),
            drift=None,
            now=NOW,
        )
        reasons = {a.reason for a in alerts}
        assert DivergenceReason.ADVERSE_DRIFT.value not in reasons
        assert DivergenceReason.CONTRADICTORY_DRIFT.value not in reasons

    def test_all_alerts_have_correct_ids(self) -> None:
        alerts = check_divergence(
            ISSUER,
            THEME,
            75.0,
            _make_adoption(score=0.05, temporal_consistency=0.1, period_count=1),
            now=NOW,
        )
        for alert in alerts:
            assert alert.issuer_concept_id == ISSUER
            assert alert.theme_concept_id == THEME
            assert alert.created_at == NOW

    def test_confirmed_scenario_no_alerts(self) -> None:
        """Strong narrative + strong adoption → no divergence."""
        alerts = check_divergence(
            ISSUER,
            THEME,
            70.0,
            _make_adoption(score=0.7),
            now=NOW,
        )
        assert alerts == []

    def test_lagging_suppresses_hype(self) -> None:
        """When lagging adoption fires, narrative_without_filing is suppressed."""
        adoption = _make_adoption(
            score=0.05,
            temporal_consistency=0.5,
            period_count=4,
            periods_with_signal=2,
        )
        alerts = check_divergence(
            ISSUER,
            THEME,
            75.0,
            adoption,
            now=NOW,
        )
        reasons = {a.reason for a in alerts}
        assert DivergenceReason.LAGGING_ADOPTION.value in reasons
        assert DivergenceReason.NARRATIVE_WITHOUT_FILING.value not in reasons

    def test_to_dict(self) -> None:
        alerts = check_divergence(
            ISSUER,
            THEME,
            75.0,
            _make_adoption(score=0.05, temporal_consistency=0.1, period_count=1),
            now=NOW,
        )
        assert len(alerts) > 0
        d = alerts[0].to_dict()
        assert d["issuer_concept_id"] == ISSUER
        assert d["reason"] == DivergenceReason.NARRATIVE_WITHOUT_FILING.value
        assert isinstance(d["created_at"], str)
        assert "evidence" in d


# -- Dataclass validation tests --------------------------------------------


class TestDivergenceAlertValidation:
    """Alert dataclass invariants."""

    def test_invalid_severity(self) -> None:
        with pytest.raises(ValueError, match="Invalid severity"):
            DivergenceAlert(
                issuer_concept_id=ISSUER,
                theme_concept_id=THEME,
                reason="test",
                severity="extreme",
                title="test",
                summary="test",
            )

    def test_valid_severities(self) -> None:
        for sev in ("critical", "warning", "info"):
            alert = DivergenceAlert(
                issuer_concept_id=ISSUER,
                theme_concept_id=THEME,
                reason="test",
                severity=sev,
                title="test",
                summary="test",
            )
            assert alert.severity == sev

    def test_frozen(self) -> None:
        alert = DivergenceAlert(
            issuer_concept_id=ISSUER,
            theme_concept_id=THEME,
            reason="test",
            severity="info",
            title="test",
            summary="test",
        )
        with pytest.raises(AttributeError):
            alert.severity = "critical"  # type: ignore[misc]

    def test_reason_enum_values(self) -> None:
        """All reason codes are distinct strings."""
        values = [r.value for r in DivergenceReason]
        assert len(values) == len(set(values))
        assert len(values) == 5
