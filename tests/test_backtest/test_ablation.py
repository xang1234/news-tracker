"""Tests for ablation experiment framework.

Verifies ablation configurations, snapshot filtering, result
comparison, and per-layer contribution quantification.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.backtest.ablation import (
    ALL_LAYERS,
    LAYER_FILING,
    LAYER_NARRATIVE,
    AblationConfig,
    AblationResult,
    LayerContribution,
    build_standard_suite,
    compare_ablation_results,
    filter_snapshot_by_layers,
)
from src.backtest.intelligence_pit import IntelligenceSnapshot

NOW = datetime(2026, 4, 1, tzinfo=UTC)
PAST = NOW - timedelta(days=7)


# -- Helpers ---------------------------------------------------------------


def _snapshot(
    claims: int = 3,
    assertions: int = 2,
    filings: int = 4,
) -> IntelligenceSnapshot:
    return IntelligenceSnapshot(
        as_of=NOW,
        claims=[{"claim_id": f"c{i}", "created_at": PAST} for i in range(claims)],
        assertions=[],
        filings=[{"accession_number": f"f{i}", "ingested_at": PAST} for i in range(filings)],
        lane_runs=[],
        manifests=[],
    )


def _result(
    name: str = "baseline",
    layers: frozenset[str] = frozenset(),
    metrics: dict[str, float] | None = None,
) -> AblationResult:
    return AblationResult(
        config=AblationConfig(
            name=name,
            description=f"Test config: {name}",
            enabled_layers=layers,
        ),
        metrics=metrics or {},
        run_id=f"run_{name}",
        evaluated_at=NOW,
    )


# -- AblationConfig tests -------------------------------------------------


class TestAblationConfig:
    """Configuration construction and validation."""

    def test_valid_config(self) -> None:
        config = AblationConfig(
            name="test",
            description="Test",
            enabled_layers=frozenset({LAYER_NARRATIVE}),
        )
        assert config.name == "test"
        assert config.layer_count == 1

    def test_empty_layers(self) -> None:
        config = AblationConfig(name="baseline", description="No layers")
        assert config.layer_count == 0

    def test_all_layers(self) -> None:
        config = AblationConfig(
            name="full",
            description="All",
            enabled_layers=ALL_LAYERS,
        )
        assert config.layer_count == 4

    def test_invalid_layer(self) -> None:
        with pytest.raises(ValueError, match="Unknown layers"):
            AblationConfig(
                name="bad",
                description="Bad",
                enabled_layers=frozenset({"nonexistent"}),
            )

    def test_to_dict(self) -> None:
        config = AblationConfig(
            name="test",
            description="Desc",
            enabled_layers=frozenset({LAYER_NARRATIVE}),
            contract_version="0.1.0",
        )
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["enabled_layers"] == ["narrative"]
        assert d["layer_count"] == 1

    def test_frozen(self) -> None:
        config = AblationConfig(name="x", description="x")
        with pytest.raises(AttributeError):
            config.name = "y"  # type: ignore[misc]


# -- Standard suite tests --------------------------------------------------


class TestStandardSuite:
    """Standard 5-slice ablation suite."""

    def test_five_configs(self) -> None:
        suite = build_standard_suite()
        assert len(suite) == 5

    def test_names(self) -> None:
        suite = build_standard_suite()
        names = {c.name for c in suite}
        assert names == {
            "baseline",
            "narrative_v2",
            "narrative_filing",
            "narrative_structural",
            "full",
        }

    def test_baseline_has_no_layers(self) -> None:
        suite = build_standard_suite()
        baseline = next(c for c in suite if c.name == "baseline")
        assert baseline.enabled_layers == frozenset()

    def test_full_has_all_layers(self) -> None:
        suite = build_standard_suite()
        full = next(c for c in suite if c.name == "full")
        assert full.enabled_layers == ALL_LAYERS

    def test_incremental_layers(self) -> None:
        """Each non-baseline config includes narrative."""
        suite = build_standard_suite()
        for config in suite:
            if config.name != "baseline":
                assert LAYER_NARRATIVE in config.enabled_layers

    def test_custom_version(self) -> None:
        suite = build_standard_suite(contract_version="1.2.3")
        assert all(c.contract_version == "1.2.3" for c in suite)


# -- Snapshot filtering tests ----------------------------------------------


class TestFilterSnapshot:
    """Filter PIT snapshots by enabled layers."""

    def test_all_layers_pass_through(self) -> None:
        snap = _snapshot(filings=4)
        filtered = filter_snapshot_by_layers(snap, ALL_LAYERS)
        assert len(filtered.filings) == 4

    def test_no_filing_layer_removes_filings(self) -> None:
        snap = _snapshot(filings=4)
        filtered = filter_snapshot_by_layers(
            snap,
            frozenset({LAYER_NARRATIVE}),
        )
        assert len(filtered.filings) == 0

    def test_filing_layer_preserves_filings(self) -> None:
        snap = _snapshot(filings=4)
        filtered = filter_snapshot_by_layers(
            snap,
            frozenset({LAYER_FILING}),
        )
        assert len(filtered.filings) == 4

    def test_claims_always_included(self) -> None:
        snap = _snapshot(claims=5)
        filtered = filter_snapshot_by_layers(snap, frozenset())
        assert len(filtered.claims) == 5

    def test_assertions_always_included(self) -> None:
        """Assertions are always included — they feed all layers."""
        snap = _snapshot()
        filtered = filter_snapshot_by_layers(snap, frozenset())
        assert filtered.assertions == snap.assertions

    def test_lane_runs_always_included(self) -> None:
        snap = _snapshot()
        filtered = filter_snapshot_by_layers(snap, frozenset())
        assert filtered.lane_runs == snap.lane_runs

    def test_as_of_preserved(self) -> None:
        snap = _snapshot()
        filtered = filter_snapshot_by_layers(snap, frozenset())
        assert filtered.as_of == snap.as_of


# -- AblationResult tests -------------------------------------------------


class TestAblationResult:
    """Result construction and serialization."""

    def test_to_dict(self) -> None:
        result = _result(
            "test",
            frozenset({LAYER_NARRATIVE}),
            metrics={"hit_rate": 0.65},
        )
        d = result.to_dict()
        assert d["config_name"] == "test"
        assert d["metrics"]["hit_rate"] == 0.65
        assert d["enabled_layers"] == ["narrative"]

    def test_frozen(self) -> None:
        result = _result()
        with pytest.raises(AttributeError):
            result.run_id = "x"  # type: ignore[misc]


# -- Comparison tests ------------------------------------------------------


class TestCompareResults:
    """Cross-config ablation comparison."""

    def test_basic_comparison(self) -> None:
        results = [
            _result("baseline", frozenset(), {"hit_rate": 0.50, "sharpe": 0.8}),
            _result(
                "narrative_v2", frozenset({LAYER_NARRATIVE}), {"hit_rate": 0.55, "sharpe": 0.9}
            ),
        ]
        comp = compare_ablation_results(results, now=NOW)
        assert len(comp.contributions) == 1
        c = comp.contributions[0]
        assert c.layer_added == "narrative"
        assert abs(c.metric_deltas["hit_rate"] - 0.05) < 1e-6
        assert abs(c.metric_deltas["sharpe"] - 0.1) < 1e-6

    def test_multiple_layers_added(self) -> None:
        results = [
            _result("baseline", frozenset(), {"hit_rate": 0.50}),
            _result("full", ALL_LAYERS, {"hit_rate": 0.60}),
        ]
        comp = compare_ablation_results(results, now=NOW)
        c = comp.contributions[0]
        assert "narrative" in c.layer_added
        assert abs(c.metric_deltas["hit_rate"] - 0.10) < 1e-6

    def test_no_baseline_no_contributions(self) -> None:
        results = [
            _result("narrative_v2", frozenset({LAYER_NARRATIVE}), {"hit_rate": 0.55}),
        ]
        comp = compare_ablation_results(results, now=NOW)
        assert comp.contributions == []

    def test_baseline_skipped_in_contributions(self) -> None:
        results = [
            _result("baseline", frozenset(), {"hit_rate": 0.50}),
        ]
        comp = compare_ablation_results(results, now=NOW)
        assert comp.contributions == []

    def test_results_keyed_by_name(self) -> None:
        results = [
            _result("baseline"),
            _result("full", ALL_LAYERS),
        ]
        comp = compare_ablation_results(results, now=NOW)
        assert "baseline" in comp.results
        assert "full" in comp.results

    def test_missing_metrics_default_zero(self) -> None:
        """Metrics absent from one config default to 0."""
        results = [
            _result("baseline", frozenset(), {"hit_rate": 0.50}),
            _result(
                "narrative_v2", frozenset({LAYER_NARRATIVE}), {"hit_rate": 0.55, "sharpe": 1.0}
            ),
        ]
        comp = compare_ablation_results(results, now=NOW)
        c = comp.contributions[0]
        # sharpe: 1.0 - 0.0 = 1.0 (baseline didn't have sharpe)
        assert abs(c.metric_deltas["sharpe"] - 1.0) < 1e-6

    def test_negative_delta(self) -> None:
        """A layer can make things worse."""
        results = [
            _result("baseline", frozenset(), {"hit_rate": 0.60}),
            _result("narrative_v2", frozenset({LAYER_NARRATIVE}), {"hit_rate": 0.55}),
        ]
        comp = compare_ablation_results(results, now=NOW)
        assert comp.contributions[0].metric_deltas["hit_rate"] < 0

    def test_to_dict(self) -> None:
        results = [
            _result("baseline", frozenset(), {"hit_rate": 0.50}),
            _result("full", ALL_LAYERS, {"hit_rate": 0.60}),
        ]
        comp = compare_ablation_results(results, now=NOW)
        d = comp.to_dict()
        assert d["result_count"] == 2
        assert len(d["contributions"]) == 1

    def test_compared_at(self) -> None:
        comp = compare_ablation_results([], now=NOW)
        assert comp.compared_at == NOW


# -- LayerContribution tests -----------------------------------------------


class TestLayerContribution:
    """Per-layer contribution quantification."""

    def test_to_dict_rounds(self) -> None:
        c = LayerContribution(
            layer_added="narrative",
            baseline_name="baseline",
            variant_name="narrative_v2",
            metric_deltas={"hit_rate": 0.0500001},
        )
        d = c.to_dict()
        assert d["metric_deltas"]["hit_rate"] == 0.05

    def test_frozen(self) -> None:
        c = LayerContribution(
            layer_added="x",
            baseline_name="b",
            variant_name="v",
        )
        with pytest.raises(AttributeError):
            c.layer_added = "y"  # type: ignore[misc]
