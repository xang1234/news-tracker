"""Tests for filing lane publication.

Verifies that filing adoption scores and divergence alerts are
prepared for manifest publication with per-issuer summaries
and lane health gating.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.filing.adoption import (
    AdoptionBreakdown,
    FilingAdoptionScore,
)
from src.filing.divergence import DivergenceAlert, DivergenceReason
from src.filing.publisher import (
    build_adoption_payload,
    build_divergence_payload,
    build_issuer_summaries,
    prepare_filing_publication,
)
from src.publish.lane_health import (
    FreshnessLevel,
    LaneHealthStatus,
    PublishReadiness,
    QualityLevel,
    QuarantineState,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)

ISSUER_A = "concept_issuer_aaa"
ISSUER_B = "concept_issuer_bbb"
THEME_1 = "concept_theme_111"
THEME_2 = "concept_theme_222"


# -- Helpers ---------------------------------------------------------------


def _make_adoption(
    issuer: str = ISSUER_A,
    theme: str = THEME_1,
    score: float = 0.5,
    section_coverage: float = 0.4,
    section_depth: float = 0.3,
    fact_alignment: float = 0.2,
    temporal_consistency: float = 0.6,
    filing_count: int = 3,
    period_count: int = 4,
    periods_with_signal: int = 3,
) -> FilingAdoptionScore:
    return FilingAdoptionScore(
        issuer_concept_id=issuer,
        theme_concept_id=theme,
        score=score,
        breakdown=AdoptionBreakdown(
            section_coverage=section_coverage,
            section_depth=section_depth,
            fact_alignment=fact_alignment,
            temporal_consistency=temporal_consistency,
        ),
        filing_count=filing_count,
        period_count=period_count,
        periods_with_signal=periods_with_signal,
        computed_at=NOW,
    )


def _make_alert(
    issuer: str = ISSUER_A,
    theme: str = THEME_1,
    reason: str = DivergenceReason.NARRATIVE_WITHOUT_FILING.value,
    severity: str = "warning",
) -> DivergenceAlert:
    return DivergenceAlert(
        issuer_concept_id=issuer,
        theme_concept_id=theme,
        reason=reason,
        severity=severity,
        title="Test alert",
        summary="Test summary",
        evidence={"test": True},
        created_at=NOW,
    )


def _healthy_status() -> LaneHealthStatus:
    return LaneHealthStatus(
        lane="filing",
        freshness=FreshnessLevel.FRESH,
        quality=QualityLevel.HEALTHY,
        quarantine=QuarantineState.CLEAR,
        readiness=PublishReadiness.READY,
    )


def _blocked_status() -> LaneHealthStatus:
    return LaneHealthStatus(
        lane="filing",
        freshness=FreshnessLevel.STALE,
        quality=QualityLevel.CRITICAL,
        quarantine=QuarantineState.CLEAR,
        readiness=PublishReadiness.BLOCKED,
    )


# -- Adoption payload tests ------------------------------------------------


class TestAdoptionPayload:
    """Build publishable adoption payloads from scores."""

    def test_basic_conversion(self) -> None:
        adoption = _make_adoption(score=0.65)
        payload = build_adoption_payload(adoption)
        assert payload.issuer_concept_id == ISSUER_A
        assert payload.theme_concept_id == THEME_1
        assert payload.score == 0.65
        assert payload.section_coverage == 0.4
        assert payload.filing_count == 3

    def test_strips_internal_signals(self) -> None:
        """Payload has signal counts, not the full signal lists."""
        adoption = _make_adoption()
        payload = build_adoption_payload(adoption)
        assert isinstance(payload.section_signal_count, int)
        assert isinstance(payload.fact_signal_count, int)
        assert not hasattr(payload, "section_signals")

    def test_to_dict(self) -> None:
        payload = build_adoption_payload(_make_adoption())
        d = payload.to_dict()
        assert d["issuer_concept_id"] == ISSUER_A
        assert "score" in d
        assert "section_coverage" in d

    def test_frozen(self) -> None:
        payload = build_adoption_payload(_make_adoption())
        try:
            payload.score = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# -- Divergence payload tests -----------------------------------------------


class TestDivergencePayload:
    """Build publishable divergence payloads from alerts."""

    def test_basic_conversion(self) -> None:
        alert = _make_alert(severity="critical")
        payload = build_divergence_payload(alert)
        assert payload.issuer_concept_id == ISSUER_A
        assert payload.reason == DivergenceReason.NARRATIVE_WITHOUT_FILING.value
        assert payload.severity == "critical"
        assert payload.evidence == {"test": True}

    def test_evidence_is_copied(self) -> None:
        """Payload evidence is a copy, not the original dict."""
        evidence = {"mutable": [1, 2, 3]}
        alert = DivergenceAlert(
            issuer_concept_id=ISSUER_A,
            theme_concept_id=THEME_1,
            reason="test",
            severity="info",
            title="t",
            summary="s",
            evidence=evidence,
            created_at=NOW,
        )
        payload = build_divergence_payload(alert)
        assert payload.evidence is not evidence
        assert payload.evidence == evidence

    def test_to_dict(self) -> None:
        payload = build_divergence_payload(_make_alert())
        d = payload.to_dict()
        assert d["reason"] == DivergenceReason.NARRATIVE_WITHOUT_FILING.value
        assert "evidence" in d

    def test_frozen(self) -> None:
        payload = build_divergence_payload(_make_alert())
        try:
            payload.severity = "info"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


# -- Issuer summary tests ---------------------------------------------------


class TestIssuerSummaries:
    """Per-issuer divergence summaries across themes."""

    def test_single_issuer_single_theme(self) -> None:
        adoptions = [_make_adoption(issuer=ISSUER_A, theme=THEME_1, score=0.4)]
        alerts = [_make_alert(issuer=ISSUER_A, theme=THEME_1, severity="warning")]
        summaries = build_issuer_summaries(adoptions, alerts)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.issuer_concept_id == ISSUER_A
        assert s.theme_count == 1
        assert s.max_adoption == 0.4
        assert s.avg_adoption == 0.4
        assert s.alert_count == 1
        assert s.warning_count == 1

    def test_single_issuer_multiple_themes(self) -> None:
        adoptions = [
            _make_adoption(issuer=ISSUER_A, theme=THEME_1, score=0.3),
            _make_adoption(issuer=ISSUER_A, theme=THEME_2, score=0.7),
        ]
        alerts = [
            _make_alert(issuer=ISSUER_A, theme=THEME_1, severity="critical"),
        ]
        summaries = build_issuer_summaries(adoptions, alerts)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.theme_count == 2
        assert s.max_adoption == 0.7
        assert s.min_adoption == 0.3
        assert abs(s.avg_adoption - 0.5) < 0.001
        assert s.critical_count == 1

    def test_multiple_issuers(self) -> None:
        adoptions = [
            _make_adoption(issuer=ISSUER_A, score=0.5),
            _make_adoption(issuer=ISSUER_B, score=0.2),
        ]
        summaries = build_issuer_summaries(adoptions, [])
        assert len(summaries) == 2
        ids = {s.issuer_concept_id for s in summaries}
        assert ids == {ISSUER_A, ISSUER_B}

    def test_alerts_only_no_adoptions(self) -> None:
        alerts = [_make_alert(issuer=ISSUER_A, severity="critical")]
        summaries = build_issuer_summaries([], alerts)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.theme_count == 1
        assert s.max_adoption == 0.0
        assert s.alert_count == 1

    def test_empty_inputs(self) -> None:
        summaries = build_issuer_summaries([], [])
        assert summaries == []

    def test_reason_counts(self) -> None:
        alerts = [
            _make_alert(reason=DivergenceReason.NARRATIVE_WITHOUT_FILING.value),
            _make_alert(reason=DivergenceReason.ADVERSE_DRIFT.value),
            _make_alert(reason=DivergenceReason.NARRATIVE_WITHOUT_FILING.value),
        ]
        summaries = build_issuer_summaries([], alerts)
        s = summaries[0]
        assert s.reason_counts["narrative_without_filing"] == 2
        assert s.reason_counts["adverse_drift"] == 1

    def test_contributing_themes_sorted(self) -> None:
        adoptions = [
            _make_adoption(theme="theme_z"),
            _make_adoption(theme="theme_a"),
        ]
        summaries = build_issuer_summaries(adoptions, [])
        assert summaries[0].contributing_themes == ["theme_a", "theme_z"]

    def test_to_dict(self) -> None:
        adoptions = [_make_adoption(score=0.5)]
        summaries = build_issuer_summaries(adoptions, [])
        d = summaries[0].to_dict()
        assert d["issuer_concept_id"] == ISSUER_A
        assert "avg_adoption" in d
        assert "reason_counts" in d


# -- prepare_filing_publication integration tests ----------------------------


class TestPrepareFilingPublication:
    """Full filing publication pipeline."""

    def test_healthy_publication(self) -> None:
        adoptions = [_make_adoption()]
        alerts = [_make_alert()]
        result = prepare_filing_publication(
            adoptions,
            alerts,
            _healthy_status(),
        )
        assert result.published is True
        assert len(result.adoption_payloads) == 1
        assert len(result.divergence_payloads) == 1
        assert len(result.issuer_summaries) == 1
        assert result.object_count == 3  # 1 + 1 + 1
        assert result.block_reason is None

    def test_blocked_publication(self) -> None:
        result = prepare_filing_publication(
            [_make_adoption()],
            [_make_alert()],
            _blocked_status(),
        )
        assert result.published is False
        assert result.block_reason is not None
        assert result.adoption_payloads == []
        assert result.divergence_payloads == []
        assert result.issuer_summaries == []
        assert result.object_count == 0

    def test_empty_inputs(self) -> None:
        result = prepare_filing_publication([], [], _healthy_status())
        assert result.published is True
        assert result.object_count == 0

    def test_object_count(self) -> None:
        adoptions = [
            _make_adoption(issuer=ISSUER_A, theme=THEME_1),
            _make_adoption(issuer=ISSUER_A, theme=THEME_2),
            _make_adoption(issuer=ISSUER_B, theme=THEME_1),
        ]
        alerts = [
            _make_alert(issuer=ISSUER_A, theme=THEME_1),
            _make_alert(issuer=ISSUER_B, theme=THEME_1),
        ]
        result = prepare_filing_publication(
            adoptions,
            alerts,
            _healthy_status(),
        )
        # 3 adoption + 2 divergence + 2 issuer summaries = 7
        assert result.object_count == 7

    def test_warn_status_still_publishes(self) -> None:
        warn_health = LaneHealthStatus(
            lane="filing",
            freshness=FreshnessLevel.AGING,
            quality=QualityLevel.HEALTHY,
            quarantine=QuarantineState.CLEAR,
            readiness=PublishReadiness.WARN,
        )
        result = prepare_filing_publication(
            [_make_adoption()],
            [],
            warn_health,
        )
        assert result.published is True

    def test_block_reason_includes_health_details(self) -> None:
        result = prepare_filing_publication(
            [],
            [],
            _blocked_status(),
        )
        assert result.block_reason is not None
        assert "stale" in result.block_reason.lower() or "blocked" in result.block_reason.lower()

    def test_multiple_issuers_produce_separate_summaries(self) -> None:
        adoptions = [
            _make_adoption(issuer=ISSUER_A, score=0.3),
            _make_adoption(issuer=ISSUER_B, score=0.8),
        ]
        result = prepare_filing_publication(
            adoptions,
            [],
            _healthy_status(),
        )
        assert len(result.issuer_summaries) == 2
        ids = {s.issuer_concept_id for s in result.issuer_summaries}
        assert ids == {ISSUER_A, ISSUER_B}

    def test_divergence_payloads_preserve_evidence(self) -> None:
        alert = _make_alert()
        result = prepare_filing_publication(
            [],
            [alert],
            _healthy_status(),
        )
        assert len(result.divergence_payloads) == 1
        assert result.divergence_payloads[0].evidence == {"test": True}
