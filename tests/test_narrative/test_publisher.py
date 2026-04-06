"""Tests for narrative lane publication.

Verifies that narrative runs are prepared for manifest publication
with component scores, symbol/theme rollups, and lane health gating.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from src.narrative.components import NarrativeComponents, compute_narrative_components
from src.narrative.publisher import (
    SymbolRollup,
    ThemeRollup,
    build_symbol_rollups,
    build_theme_rollups,
    prepare_narrative_publication,
)
from src.narrative.schemas import NarrativeRun
from src.publish.lane_health import (
    FreshnessLevel,
    LaneHealthStatus,
    PublishReadiness,
    QualityLevel,
    QuarantineState,
)

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Helpers ---------------------------------------------------------------


def _make_run(
    run_id: str = "nr_001",
    theme_id: str = "theme_hbm",
    label: str = "HBM Surge",
    doc_count: int = 15,
    **overrides,
) -> NarrativeRun:
    defaults = {
        "run_id": run_id,
        "theme_id": theme_id,
        "status": "active",
        "centroid": np.zeros(768),
        "label": label,
        "started_at": NOW - timedelta(hours=4),
        "last_document_at": NOW,
        "doc_count": doc_count,
        "platform_count": 3,
        "avg_sentiment": 0.6,
        "avg_authority": 0.7,
        "current_rate_per_hour": 20.0,
        "current_acceleration": 8.0,
        "conviction_score": 50.0,
        "ticker_counts": {"TSM": 5, "NVDA": 3},
    }
    defaults.update(overrides)
    return NarrativeRun(**defaults)


def _healthy_status() -> LaneHealthStatus:
    return LaneHealthStatus(
        lane="narrative",
        freshness=FreshnessLevel.FRESH,
        quality=QualityLevel.HEALTHY,
        quarantine=QuarantineState.CLEAR,
        readiness=PublishReadiness.READY,
    )


def _blocked_status() -> LaneHealthStatus:
    return LaneHealthStatus(
        lane="narrative",
        freshness=FreshnessLevel.STALE,
        quality=QualityLevel.CRITICAL,
        quarantine=QuarantineState.CLEAR,
        readiness=PublishReadiness.BLOCKED,
    )


# -- Symbol rollup tests ---------------------------------------------------


class TestSymbolRollups:
    """Per-symbol aggregation across runs."""

    def test_single_run(self) -> None:
        run = _make_run(ticker_counts={"TSM": 5, "NVDA": 3})
        comp = compute_narrative_components(
            current_rate_per_hour=20.0, current_acceleration=8.0,
            doc_count=15, platform_count=3, avg_sentiment=0.6,
            avg_authority=0.7, last_document_at=NOW,
            started_at=NOW - timedelta(hours=4), now=NOW,
        )
        rollups = build_symbol_rollups([run], {run.run_id: comp})
        assert len(rollups) == 2
        symbols = {r.symbol for r in rollups}
        assert symbols == {"TSM", "NVDA"}

    def test_multiple_runs_same_symbol(self) -> None:
        r1 = _make_run("nr_1", ticker_counts={"TSM": 5})
        r2 = _make_run("nr_2", ticker_counts={"TSM": 3}, doc_count=10)
        comp1 = compute_narrative_components(
            current_rate_per_hour=20.0, current_acceleration=8.0,
            doc_count=15, platform_count=3, avg_sentiment=0.6,
            avg_authority=0.7, last_document_at=NOW,
            started_at=NOW - timedelta(hours=4), now=NOW,
        )
        comp2 = compute_narrative_components(
            current_rate_per_hour=10.0, current_acceleration=4.0,
            doc_count=10, platform_count=2, avg_sentiment=0.4,
            avg_authority=0.5, last_document_at=NOW,
            started_at=NOW - timedelta(hours=2), now=NOW,
        )
        rollups = build_symbol_rollups(
            [r1, r2], {"nr_1": comp1, "nr_2": comp2}
        )
        tsm = next(r for r in rollups if r.symbol == "TSM")
        assert tsm.run_count == 2
        assert tsm.total_doc_count == 8  # 5 + 3

    def test_rollup_to_dict(self) -> None:
        rollup = SymbolRollup(
            symbol="TSM", run_count=2, total_doc_count=10,
            max_composite=45.0, avg_sentiment=0.55,
            contributing_run_ids=["nr_1", "nr_2"],
        )
        d = rollup.to_dict()
        assert d["symbol"] == "TSM"
        assert d["run_count"] == 2

    def test_empty_runs(self) -> None:
        rollups = build_symbol_rollups([], {})
        assert rollups == []


# -- Theme rollup tests ----------------------------------------------------


class TestThemeRollups:
    """Per-theme aggregation across runs."""

    def test_single_theme(self) -> None:
        r1 = _make_run("nr_1", theme_id="theme_hbm", label="HBM Surge")
        r2 = _make_run("nr_2", theme_id="theme_hbm", label="HBM Surge",
                        doc_count=10)
        comps = {
            "nr_1": compute_narrative_components(
                current_rate_per_hour=20.0, current_acceleration=8.0,
                doc_count=15, platform_count=3, avg_sentiment=0.6,
                avg_authority=0.7, last_document_at=NOW,
                started_at=NOW - timedelta(hours=4), now=NOW,
            ),
            "nr_2": compute_narrative_components(
                current_rate_per_hour=10.0, current_acceleration=4.0,
                doc_count=10, platform_count=2, avg_sentiment=0.4,
                avg_authority=0.5, last_document_at=NOW,
                started_at=NOW - timedelta(hours=2), now=NOW,
            ),
        }
        rollups = build_theme_rollups([r1, r2], comps)
        assert len(rollups) == 1
        assert rollups[0].theme_id == "theme_hbm"
        assert rollups[0].run_count == 2
        assert rollups[0].total_doc_count == 25  # 15 + 10

    def test_top_symbols(self) -> None:
        run = _make_run(ticker_counts={"TSM": 10, "NVDA": 5, "AMD": 3})
        comp = compute_narrative_components(
            current_rate_per_hour=20.0, current_acceleration=8.0,
            doc_count=15, platform_count=3, avg_sentiment=0.6,
            avg_authority=0.7, last_document_at=NOW,
            started_at=NOW - timedelta(hours=4), now=NOW,
        )
        rollups = build_theme_rollups([run], {run.run_id: comp})
        assert rollups[0].top_symbols[0] == "TSM"  # highest count

    def test_rollup_to_dict(self) -> None:
        rollup = ThemeRollup(
            theme_id="theme_hbm", theme_label="HBM", run_count=1,
            total_doc_count=15, max_composite=40.0, avg_sentiment=0.6,
            top_symbols=["TSM"], contributing_run_ids=["nr_1"],
        )
        d = rollup.to_dict()
        assert d["theme_id"] == "theme_hbm"
        assert d["top_symbols"] == ["TSM"]


# -- prepare_narrative_publication tests -----------------------------------


class TestPreparePublication:
    """Full publication preparation pipeline."""

    def test_healthy_publication(self) -> None:
        runs = [_make_run()]
        result = prepare_narrative_publication(
            runs, _healthy_status(), now=NOW
        )
        assert result.published is True
        assert len(result.run_payloads) == 1
        assert len(result.components) == 1
        assert result.object_count > 0
        assert result.block_reason is None

    def test_blocked_publication(self) -> None:
        runs = [_make_run()]
        result = prepare_narrative_publication(
            runs, _blocked_status(), now=NOW
        )
        assert result.published is False
        assert result.block_reason is not None
        assert "blocked" in result.block_reason.lower() or "stale" in result.block_reason.lower()
        assert result.run_payloads == []
        assert result.object_count == 0

    def test_components_per_run(self) -> None:
        runs = [_make_run("nr_1"), _make_run("nr_2")]
        result = prepare_narrative_publication(
            runs, _healthy_status(), now=NOW
        )
        assert "nr_1" in result.components
        assert "nr_2" in result.components
        assert isinstance(result.components["nr_1"], NarrativeComponents)

    def test_symbol_rollups_included(self) -> None:
        runs = [_make_run(ticker_counts={"TSM": 5, "NVDA": 3})]
        result = prepare_narrative_publication(
            runs, _healthy_status(), now=NOW
        )
        assert len(result.symbol_rollups) == 2

    def test_theme_rollups_included(self) -> None:
        runs = [_make_run(theme_id="theme_a"), _make_run("nr_2", theme_id="theme_b")]
        result = prepare_narrative_publication(
            runs, _healthy_status(), now=NOW
        )
        assert len(result.theme_rollups) == 2

    def test_empty_runs(self) -> None:
        result = prepare_narrative_publication(
            [], _healthy_status(), now=NOW
        )
        assert result.published is True
        assert result.object_count == 0

    def test_object_count(self) -> None:
        runs = [_make_run(ticker_counts={"TSM": 5, "NVDA": 3})]
        result = prepare_narrative_publication(
            runs, _healthy_status(), now=NOW
        )
        # 1 run payload + 2 symbol rollups + 1 theme rollup = 4
        assert result.object_count == 4

    def test_warn_status_still_publishes(self) -> None:
        warn_health = LaneHealthStatus(
            lane="narrative",
            freshness=FreshnessLevel.AGING,
            quality=QualityLevel.HEALTHY,
            quarantine=QuarantineState.CLEAR,
            readiness=PublishReadiness.WARN,
        )
        result = prepare_narrative_publication(
            [_make_run()], warn_health, now=NOW
        )
        assert result.published is True
