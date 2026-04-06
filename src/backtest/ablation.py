"""Ablation experiment framework for the intelligence layer.

Defines reproducible experiment slices that enable intelligence
layers one at a time, enabling quantification of which layers
add value and where the stack may be overfitting.

Standard ablation suite:
    - baseline: no new intelligence layers
    - narrative_v2: add new narrative component scores
    - narrative_filing: add filing adoption and divergence
    - narrative_structural: add structural paths and baskets
    - full: all layers enabled

Each config specifies which layers are enabled. The framework
filters PIT snapshots accordingly and compares results across
configurations.

All functions are stateless — the caller runs the backtest
engine with each config, the framework compares the outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.backtest.intelligence_pit import IntelligenceSnapshot
from src.contracts.intelligence.version import ContractRegistry


# -- Intelligence layer names --------------------------------------------------

LAYER_NARRATIVE = "narrative"
LAYER_FILING = "filing"
LAYER_STRUCTURAL = "structural"
LAYER_DIVERGENCE = "divergence"

ALL_LAYERS = frozenset({
    LAYER_NARRATIVE,
    LAYER_FILING,
    LAYER_STRUCTURAL,
    LAYER_DIVERGENCE,
})


# -- Ablation configuration ---------------------------------------------------


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for a single ablation experiment slice.

    Attributes:
        name: Short identifier (e.g., "baseline", "full").
        description: Human-readable explanation of what this slice tests.
        enabled_layers: Which intelligence layers are active.
        contract_version: For run-version attribution.
    """

    name: str
    description: str
    enabled_layers: frozenset[str] = field(default_factory=frozenset)
    contract_version: str = ""

    def __post_init__(self) -> None:
        unknown = self.enabled_layers - ALL_LAYERS
        if unknown:
            raise ValueError(
                f"Unknown layers in config {self.name!r}: {sorted(unknown)}. "
                f"Must be from {sorted(ALL_LAYERS)}"
            )

    @property
    def layer_count(self) -> int:
        return len(self.enabled_layers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "enabled_layers": sorted(self.enabled_layers),
            "layer_count": self.layer_count,
            "contract_version": self.contract_version,
        }


# -- Standard ablation suite ---------------------------------------------------


def build_standard_suite(
    contract_version: str | None = None,
) -> list[AblationConfig]:
    """Build the standard 5-slice ablation suite.

    Returns configs for baseline, narrative_v2, narrative_filing,
    narrative_structural, and full experiments.
    """
    version = contract_version or str(ContractRegistry.CURRENT)

    return [
        AblationConfig(
            name="baseline",
            description="No new intelligence layers — existing system only",
            enabled_layers=frozenset(),
            contract_version=version,
        ),
        AblationConfig(
            name="narrative_v2",
            description="New narrative component scores (attention, corroboration, confirmation, novelty)",
            enabled_layers=frozenset({LAYER_NARRATIVE}),
            contract_version=version,
        ),
        AblationConfig(
            name="narrative_filing",
            description="Narrative + filing adoption, drift, and divergence",
            enabled_layers=frozenset({LAYER_NARRATIVE, LAYER_FILING}),
            contract_version=version,
        ),
        AblationConfig(
            name="narrative_structural",
            description="Narrative + structural paths, baskets, and beneficiaries",
            enabled_layers=frozenset({LAYER_NARRATIVE, LAYER_STRUCTURAL}),
            contract_version=version,
        ),
        AblationConfig(
            name="full",
            description="All intelligence layers enabled",
            enabled_layers=ALL_LAYERS,
            contract_version=version,
        ),
    ]


# -- Snapshot filtering by enabled layers --------------------------------------


def filter_snapshot_by_layers(
    snapshot: IntelligenceSnapshot,
    enabled_layers: frozenset[str],
) -> IntelligenceSnapshot:
    """Filter a PIT snapshot to only include enabled layer data.

    Zeroes out entities from disabled layers so the backtest
    sees a controlled view of the intelligence state.

    Layer → entity mapping:
        narrative: assertions with narrative-lane lineage
        filing: filings (all filing artifacts)
        structural: (no filtering — structural derives from assertions)
        divergence: (no filtering — divergence derives from narrative + filing)

    Claims are always included (they feed assertions).
    Lane runs and manifests are always included (infrastructure).
    """
    if enabled_layers >= ALL_LAYERS:
        return snapshot

    filings = snapshot.filings if LAYER_FILING in enabled_layers else []

    return IntelligenceSnapshot(
        as_of=snapshot.as_of,
        claims=snapshot.claims,
        assertions=snapshot.assertions,
        filings=filings,
        lane_runs=snapshot.lane_runs,
        manifests=snapshot.manifests,
    )


# -- Ablation result -----------------------------------------------------------


@dataclass(frozen=True)
class AblationResult:
    """Result of a single ablation experiment run.

    Attributes:
        config: Which ablation configuration was used.
        snapshot_summary: PIT snapshot summary at evaluation time.
        metrics: Evaluation metrics from the backtest engine.
        run_id: For attribution to a concrete backtest run.
        evaluated_at: When the evaluation was performed.
    """

    config: AblationConfig
    snapshot_summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    run_id: str = ""
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config.name,
            "enabled_layers": sorted(self.config.enabled_layers),
            "layer_count": self.config.layer_count,
            "metrics": self.metrics,
            "run_id": self.run_id,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# -- Ablation comparison -------------------------------------------------------


@dataclass(frozen=True)
class LayerContribution:
    """Quantified contribution of adding a layer vs a baseline.

    Attributes:
        layer_added: Which layer was added.
        baseline_name: Config name of the comparison baseline.
        variant_name: Config name with the layer added.
        metric_deltas: Metric → (variant_value - baseline_value).
    """

    layer_added: str
    baseline_name: str
    variant_name: str
    metric_deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_added": self.layer_added,
            "baseline_name": self.baseline_name,
            "variant_name": self.variant_name,
            "metric_deltas": {
                k: round(v, 6) for k, v in self.metric_deltas.items()
            },
        }


@dataclass(frozen=True)
class AblationComparison:
    """Comparison of ablation results across configurations.

    Attributes:
        results: All ablation results keyed by config name.
        contributions: Per-layer contribution estimates.
        compared_at: When the comparison was performed.
    """

    results: dict[str, AblationResult] = field(default_factory=dict)
    contributions: list[LayerContribution] = field(default_factory=list)
    compared_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_count": len(self.results),
            "contributions": [c.to_dict() for c in self.contributions],
            "compared_at": self.compared_at.isoformat(),
        }


def compare_ablation_results(
    results: list[AblationResult],
    *,
    baseline_name: str = "baseline",
    now: datetime | None = None,
) -> AblationComparison:
    """Compare ablation results to quantify per-layer contributions.

    For each non-baseline result, computes metric deltas against
    the baseline. If the baseline is missing, deltas are empty.

    Args:
        results: All ablation results.
        baseline_name: Which config is the comparison baseline.
        now: Comparison timestamp.

    Returns:
        AblationComparison with per-layer contribution estimates.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    by_name = {r.config.name: r for r in results}
    baseline = by_name.get(baseline_name)

    contributions: list[LayerContribution] = []

    if baseline is not None:
        for result in results:
            if result.config.name == baseline_name:
                continue

            added = result.config.enabled_layers - baseline.config.enabled_layers
            if not added:
                continue

            deltas: dict[str, float] = {}
            all_metrics = set(baseline.metrics) | set(result.metrics)
            for metric in sorted(all_metrics):
                base_val = baseline.metrics.get(metric, 0.0)
                var_val = result.metrics.get(metric, 0.0)
                deltas[metric] = var_val - base_val

            contributions.append(
                LayerContribution(
                    layer_added=", ".join(sorted(added)),
                    baseline_name=baseline_name,
                    variant_name=result.config.name,
                    metric_deltas=deltas,
                )
            )

    return AblationComparison(
        results=by_name,
        contributions=contributions,
        compared_at=now,
    )
