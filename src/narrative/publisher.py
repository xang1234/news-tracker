"""Publish narrative lane outputs keyed by manifest.

Orchestrates the narrative publication pipeline: extracts payloads
from narrative runs, computes component scores, checks lane health,
builds symbol/theme rollups, and produces publishable objects for
the manifest.

This is the integration point — it wires together:
    - NarrativeLaneAdapter (payload extraction)
    - compute_narrative_components (component scoring)
    - compute_lane_health (readiness check)
    - Symbol/theme rollup aggregation

Publication flow:
    1. Check lane health — abort if BLOCKED
    2. Extract payloads from active/cooling narrative runs
    3. Compute component scores for each run
    4. Build symbol and theme rollups
    5. Return publishable objects for manifest inclusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.narrative.components import NarrativeComponents, compute_narrative_components
from src.narrative.lane_adapter import (
    NarrativeLaneAdapter,
    NarrativeRunPayload,
)
from src.narrative.schemas import NarrativeRun
from src.publish.lane_health import LaneHealthStatus, PublishReadiness


# -- Rollup dataclasses ----------------------------------------------------


@dataclass(frozen=True)
class SymbolRollup:
    """Per-symbol narrative summary across runs.

    Attributes:
        symbol: Ticker symbol.
        run_count: Number of narrative runs mentioning this symbol.
        total_doc_count: Documents across all runs for this symbol.
        max_composite: Highest component composite among runs.
        avg_sentiment: Weighted average sentiment across runs.
        contributing_run_ids: Which runs contribute to this rollup.
    """

    symbol: str
    run_count: int
    total_doc_count: int
    max_composite: float
    avg_sentiment: float
    contributing_run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "run_count": self.run_count,
            "total_doc_count": self.total_doc_count,
            "max_composite": self.max_composite,
            "avg_sentiment": round(self.avg_sentiment, 4),
            "contributing_run_ids": self.contributing_run_ids,
        }


@dataclass(frozen=True)
class ThemeRollup:
    """Per-theme narrative summary across runs.

    Attributes:
        theme_id: Theme identifier.
        theme_label: Human-readable theme label.
        run_count: Narrative runs for this theme.
        total_doc_count: Documents across all runs.
        max_composite: Highest component composite.
        avg_sentiment: Weighted average sentiment.
        top_symbols: Most-mentioned symbols.
        contributing_run_ids: Which runs contribute.
    """

    theme_id: str
    theme_label: str
    run_count: int
    total_doc_count: int
    max_composite: float
    avg_sentiment: float
    top_symbols: list[str] = field(default_factory=list)
    contributing_run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "theme_label": self.theme_label,
            "run_count": self.run_count,
            "total_doc_count": self.total_doc_count,
            "max_composite": self.max_composite,
            "avg_sentiment": round(self.avg_sentiment, 4),
            "top_symbols": self.top_symbols,
            "contributing_run_ids": self.contributing_run_ids,
        }


# -- Publication result ----------------------------------------------------


@dataclass
class NarrativePublicationResult:
    """Result of a narrative publication attempt.

    Attributes:
        published: Whether publication succeeded.
        lane_health: The health check result.
        run_payloads: Extracted run payloads.
        components: Component scores per run.
        symbol_rollups: Per-symbol summaries.
        theme_rollups: Per-theme summaries.
        object_count: Total publishable objects produced.
        block_reason: Why publication was blocked (if applicable).
    """

    published: bool
    lane_health: LaneHealthStatus
    run_payloads: list[NarrativeRunPayload] = field(default_factory=list)
    components: dict[str, NarrativeComponents] = field(default_factory=dict)
    symbol_rollups: list[SymbolRollup] = field(default_factory=list)
    theme_rollups: list[ThemeRollup] = field(default_factory=list)
    object_count: int = 0
    block_reason: str | None = None


# -- Rollup computation (stateless) ----------------------------------------


def build_symbol_rollups(
    runs: list[NarrativeRun],
    components: dict[str, NarrativeComponents],
) -> list[SymbolRollup]:
    """Build per-symbol rollups from narrative runs and their components.

    Groups runs by ticker symbols and aggregates metrics.
    """
    symbol_data: dict[str, dict[str, Any]] = {}

    for run in runs:
        comp = components.get(run.run_id)
        composite = comp.composite if comp else 0.0

        for symbol, count in run.ticker_counts.items():
            if symbol not in symbol_data:
                symbol_data[symbol] = {
                    "run_ids": [],
                    "doc_count": 0,
                    "max_composite": 0.0,
                    "sentiment_sum": 0.0,
                    "sentiment_weight": 0.0,
                }
            s = symbol_data[symbol]
            s["run_ids"].append(run.run_id)
            s["doc_count"] += count
            s["max_composite"] = max(s["max_composite"], composite)
            s["sentiment_sum"] += run.avg_sentiment * count
            s["sentiment_weight"] += count

    rollups = []
    for symbol, s in sorted(symbol_data.items()):
        avg_sent = (
            s["sentiment_sum"] / s["sentiment_weight"]
            if s["sentiment_weight"] > 0
            else 0.0
        )
        rollups.append(SymbolRollup(
            symbol=symbol,
            run_count=len(s["run_ids"]),
            total_doc_count=s["doc_count"],
            max_composite=s["max_composite"],
            avg_sentiment=avg_sent,
            contributing_run_ids=s["run_ids"],
        ))
    return rollups


def build_theme_rollups(
    runs: list[NarrativeRun],
    components: dict[str, NarrativeComponents],
) -> list[ThemeRollup]:
    """Build per-theme rollups from narrative runs and their components."""
    theme_data: dict[str, dict[str, Any]] = {}

    for run in runs:
        tid = run.theme_id
        comp = components.get(run.run_id)
        composite = comp.composite if comp else 0.0

        if tid not in theme_data:
            theme_data[tid] = {
                "label": run.label,
                "run_ids": [],
                "doc_count": 0,
                "max_composite": 0.0,
                "sentiment_sum": 0.0,
                "sentiment_weight": 0.0,
                "symbol_counts": {},
            }
        t = theme_data[tid]
        t["run_ids"].append(run.run_id)
        t["doc_count"] += run.doc_count
        t["max_composite"] = max(t["max_composite"], composite)
        t["sentiment_sum"] += run.avg_sentiment * run.doc_count
        t["sentiment_weight"] += run.doc_count
        for sym, cnt in run.ticker_counts.items():
            t["symbol_counts"][sym] = t["symbol_counts"].get(sym, 0) + cnt

    rollups = []
    for tid, t in sorted(theme_data.items()):
        avg_sent = (
            t["sentiment_sum"] / t["sentiment_weight"]
            if t["sentiment_weight"] > 0
            else 0.0
        )
        top_symbols = sorted(
            t["symbol_counts"], key=lambda s: -t["symbol_counts"][s]
        )[:5]
        rollups.append(ThemeRollup(
            theme_id=tid,
            theme_label=t["label"],
            run_count=len(t["run_ids"]),
            total_doc_count=t["doc_count"],
            max_composite=t["max_composite"],
            avg_sentiment=avg_sent,
            top_symbols=top_symbols,
            contributing_run_ids=t["run_ids"],
        ))
    return rollups


# -- Publisher -------------------------------------------------------------


def prepare_narrative_publication(
    runs: list[NarrativeRun],
    lane_health: LaneHealthStatus,
    *,
    adapter: NarrativeLaneAdapter | None = None,
    now: datetime | None = None,
) -> NarrativePublicationResult:
    """Prepare narrative lane outputs for manifest publication.

    Checks lane health, extracts payloads, computes components,
    and builds rollups. Returns a result the caller can use to
    create manifest objects.

    Does NOT persist anything — the caller handles manifest creation
    and object insertion.

    Args:
        runs: Active/cooling narrative runs to publish.
        lane_health: Pre-computed lane health status.
        adapter: Lane adapter (created with defaults if not provided).
        now: Current time for component computation.

    Returns:
        NarrativePublicationResult with payloads, components, rollups.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if adapter is None:
        adapter = NarrativeLaneAdapter()

    # Check lane health
    if lane_health.readiness == PublishReadiness.BLOCKED:
        return NarrativePublicationResult(
            published=False,
            lane_health=lane_health,
            block_reason=lane_health.format_block_reason(),
        )

    # Extract payloads
    payloads = adapter.extract_run_payloads(runs)

    # Compute components for each run
    components: dict[str, NarrativeComponents] = {}
    for run in runs:
        components[run.run_id] = compute_narrative_components(
            current_rate_per_hour=run.current_rate_per_hour,
            current_acceleration=run.current_acceleration,
            doc_count=run.doc_count,
            platform_count=run.platform_count,
            avg_sentiment=run.avg_sentiment,
            avg_authority=run.avg_authority,
            last_document_at=run.last_document_at,
            started_at=run.started_at,
            now=now,
        )

    # Build rollups
    symbol_rollups = build_symbol_rollups(runs, components)
    theme_rollups = build_theme_rollups(runs, components)

    object_count = len(payloads) + len(symbol_rollups) + len(theme_rollups)

    return NarrativePublicationResult(
        published=True,
        lane_health=lane_health,
        run_payloads=payloads,
        components=components,
        symbol_rollups=symbol_rollups,
        theme_rollups=theme_rollups,
        object_count=object_count,
    )
