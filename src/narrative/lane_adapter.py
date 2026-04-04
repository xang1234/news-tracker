"""Adapter that positions narrative runs as lane state for the publish pipeline.

Bridges the existing narrative processing model (NarrativeRun,
signals, buckets) to the intelligence layer's lane/publish
infrastructure. The adapter does NOT modify existing narrative
workflows — it provides a lane-compatible interface on top.

Responsibilities:
    - Map narrative worker processing cycles to LaneRun records
    - Wrap narrative run outputs into publishable object payloads
    - Extract claim-ready triples from narrative signals
    - Track lane-level metadata for the narrative pipeline

Usage:
    adapter = NarrativeLaneAdapter(contract_version="0.1.0")
    lane_run = adapter.create_lane_run(config_snapshot={...})
    payloads = adapter.extract_payloads(narrative_runs, signals)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.contracts.intelligence.lanes import LANE_NARRATIVE
from src.narrative.schemas import NarrativeRun, NarrativeSignalState

# -- Valid output types for the narrative lane ------------------------------

NARRATIVE_OBJECT_TYPES = frozenset(
    {"narrative_run_snapshot", "narrative_signal"}
)


# -- Payload dataclasses ---------------------------------------------------


@dataclass(frozen=True)
class NarrativeRunPayload:
    """A publishable snapshot of a narrative run.

    Wraps the key metrics and state of a NarrativeRun into a
    payload suitable for the publish pipeline. Does not include
    the full centroid (too large for serialization).

    Attributes:
        run_id: The narrative run.
        theme_id: Associated theme.
        status: Run lifecycle state.
        label: Human-readable run label.
        doc_count: Documents in the run.
        platform_count: Distinct platforms.
        avg_sentiment: Average sentiment score.
        avg_authority: Average authority score.
        current_rate_per_hour: Current document rate.
        conviction_score: Signal conviction (0-100).
        ticker_counts: Per-ticker document counts.
        started_at: Run start time.
        last_document_at: Most recent document.
    """

    run_id: str
    theme_id: str
    status: str
    label: str
    doc_count: int
    platform_count: int
    avg_sentiment: float
    avg_authority: float
    current_rate_per_hour: float
    conviction_score: float
    ticker_counts: dict[str, int] = field(default_factory=dict)
    started_at: datetime | None = None
    last_document_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for published object payload."""
        return {
            "run_id": self.run_id,
            "theme_id": self.theme_id,
            "status": self.status,
            "label": self.label,
            "doc_count": self.doc_count,
            "platform_count": self.platform_count,
            "avg_sentiment": self.avg_sentiment,
            "avg_authority": self.avg_authority,
            "current_rate_per_hour": self.current_rate_per_hour,
            "conviction_score": self.conviction_score,
            "ticker_counts": self.ticker_counts,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_document_at": (
                self.last_document_at.isoformat()
                if self.last_document_at
                else None
            ),
        }


@dataclass(frozen=True)
class NarrativeSignalPayload:
    """A publishable narrative signal event.

    Attributes:
        run_id: The narrative run that triggered the signal.
        trigger_type: Signal type (narrative_surge, etc.).
        state: Current signal state (active/inactive).
        last_score: Most recent evaluation score.
        metadata: Signal-specific metadata.
    """

    run_id: str
    trigger_type: str
    state: str
    last_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trigger_type": self.trigger_type,
            "state": self.state,
            "last_score": self.last_score,
            "metadata": self.metadata,
        }


# -- Lane adapter ----------------------------------------------------------


class NarrativeLaneAdapter:
    """Bridges narrative processing to the lane/publish pipeline.

    Does not modify existing narrative workflows. Provides a
    lane-compatible interface so later publish tasks can target
    narrative outputs cleanly.

    Usage:
        adapter = NarrativeLaneAdapter(contract_version="0.1.0")
        lane_run_fields = adapter.start_lane_run()
        payloads = adapter.extract_run_payloads(narrative_runs)
    """

    def __init__(self, contract_version: str = "0.1.0") -> None:
        self._contract_version = contract_version

    @property
    def lane(self) -> str:
        return LANE_NARRATIVE

    @property
    def contract_version(self) -> str:
        return self._contract_version

    def start_lane_run(
        self,
        *,
        config_snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create lane run fields for a narrative processing cycle.

        Returns a dict with the fields needed to create a LaneRun
        record. The caller handles persistence.
        """
        return {
            "run_id": f"narrative_run_{uuid.uuid4().hex[:12]}",
            "lane": LANE_NARRATIVE,
            "status": "pending",
            "contract_version": self._contract_version,
            "config_snapshot": config_snapshot or {},
            "metadata": metadata or {},
        }

    def extract_run_payload(
        self, narrative_run: NarrativeRun
    ) -> NarrativeRunPayload:
        """Extract a publishable payload from a narrative run."""
        return NarrativeRunPayload(
            run_id=narrative_run.run_id,
            theme_id=narrative_run.theme_id,
            status=narrative_run.status,
            label=narrative_run.label,
            doc_count=narrative_run.doc_count,
            platform_count=narrative_run.platform_count,
            avg_sentiment=narrative_run.avg_sentiment,
            avg_authority=narrative_run.avg_authority,
            current_rate_per_hour=narrative_run.current_rate_per_hour,
            conviction_score=narrative_run.conviction_score,
            ticker_counts=dict(narrative_run.ticker_counts),
            started_at=narrative_run.started_at,
            last_document_at=narrative_run.last_document_at,
        )

    def extract_run_payloads(
        self, narrative_runs: list[NarrativeRun]
    ) -> list[NarrativeRunPayload]:
        """Extract publishable payloads from multiple narrative runs."""
        return [self.extract_run_payload(r) for r in narrative_runs]

    def extract_signal_payload(
        self, signal: NarrativeSignalState
    ) -> NarrativeSignalPayload:
        """Extract a publishable payload from a signal state."""
        return NarrativeSignalPayload(
            run_id=signal.run_id,
            trigger_type=signal.trigger_type,
            state=signal.state,
            last_score=signal.last_score,
            metadata=signal.metadata,
        )

    def make_source_ids(
        self, narrative_run: NarrativeRun
    ) -> list[str]:
        """Build source ID list for a published object's lineage.

        Links back to the narrative run and its theme for traceability.
        """
        return [narrative_run.run_id, narrative_run.theme_id]
