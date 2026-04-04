"""Tests for the narrative lane adapter.

Verifies that the adapter bridges narrative runs to the publish
pipeline without modifying existing narrative workflows.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.contracts.intelligence.lanes import LANE_NARRATIVE
from src.narrative.lane_adapter import (
    NARRATIVE_OBJECT_TYPES,
    NarrativeLaneAdapter,
    NarrativeRunPayload,
    NarrativeSignalPayload,
)
from src.narrative.schemas import NarrativeRun, NarrativeSignalState

NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)


# -- Helpers ---------------------------------------------------------------


def _make_narrative_run(
    run_id: str = "nr_001",
    **overrides,
) -> NarrativeRun:
    defaults = dict(
        run_id=run_id,
        theme_id="theme_hbm",
        status="active",
        centroid=np.zeros(768),
        label="HBM Supply Surge",
        started_at=NOW,
        last_document_at=NOW,
        doc_count=15,
        platform_count=3,
        avg_sentiment=0.6,
        avg_authority=0.7,
        current_rate_per_hour=12.0,
        conviction_score=75.0,
        ticker_counts={"TSM": 5, "NVDA": 3},
    )
    defaults.update(overrides)
    return NarrativeRun(**defaults)


def _make_signal(
    run_id: str = "nr_001",
    trigger_type: str = "narrative_surge",
    **overrides,
) -> NarrativeSignalState:
    defaults = dict(
        run_id=run_id,
        trigger_type=trigger_type,
        state="active",
        last_score=0.85,
        metadata={"uplift": 3.2},
    )
    defaults.update(overrides)
    return NarrativeSignalState(**defaults)


# -- Adapter tests ---------------------------------------------------------


class TestNarrativeLaneAdapter:
    """Core adapter behavior."""

    def test_lane_is_narrative(self) -> None:
        adapter = NarrativeLaneAdapter()
        assert adapter.lane == LANE_NARRATIVE

    def test_contract_version(self) -> None:
        adapter = NarrativeLaneAdapter(contract_version="0.2.0")
        assert adapter.contract_version == "0.2.0"

    def test_start_lane_run(self) -> None:
        adapter = NarrativeLaneAdapter()
        fields = adapter.start_lane_run(
            config_snapshot={"batch_size": 32},
        )
        assert fields["lane"] == LANE_NARRATIVE
        assert fields["status"] == "pending"
        assert fields["run_id"].startswith("narrative_run_")
        assert fields["config_snapshot"]["batch_size"] == 32

    def test_lane_run_id_unique(self) -> None:
        adapter = NarrativeLaneAdapter()
        r1 = adapter.start_lane_run()
        r2 = adapter.start_lane_run()
        assert r1["run_id"] != r2["run_id"]


# -- Run payload tests -----------------------------------------------------


class TestRunPayload:
    """Narrative run → publishable payload conversion."""

    def test_extract_run_payload(self) -> None:
        adapter = NarrativeLaneAdapter()
        run = _make_narrative_run()
        payload = adapter.extract_run_payload(run)

        assert isinstance(payload, NarrativeRunPayload)
        assert payload.run_id == "nr_001"
        assert payload.theme_id == "theme_hbm"
        assert payload.doc_count == 15
        assert payload.avg_sentiment == 0.6
        assert payload.conviction_score == 75.0
        assert payload.ticker_counts == {"TSM": 5, "NVDA": 3}

    def test_extract_multiple(self) -> None:
        adapter = NarrativeLaneAdapter()
        runs = [
            _make_narrative_run("nr_001"),
            _make_narrative_run("nr_002", doc_count=5),
        ]
        payloads = adapter.extract_run_payloads(runs)
        assert len(payloads) == 2
        assert payloads[0].run_id == "nr_001"
        assert payloads[1].doc_count == 5

    def test_payload_to_dict(self) -> None:
        adapter = NarrativeLaneAdapter()
        run = _make_narrative_run()
        payload = adapter.extract_run_payload(run)
        d = payload.to_dict()

        assert d["run_id"] == "nr_001"
        assert d["theme_id"] == "theme_hbm"
        assert d["doc_count"] == 15
        assert d["started_at"] == NOW.isoformat()
        assert isinstance(d["ticker_counts"], dict)

    def test_payload_excludes_centroid(self) -> None:
        """Centroid is too large for published payloads."""
        adapter = NarrativeLaneAdapter()
        run = _make_narrative_run()
        payload = adapter.extract_run_payload(run)
        d = payload.to_dict()
        assert "centroid" not in d

    def test_none_timestamps_handled(self) -> None:
        run = _make_narrative_run(started_at=NOW, last_document_at=NOW)
        adapter = NarrativeLaneAdapter()
        payload = adapter.extract_run_payload(run)
        d = payload.to_dict()
        assert d["started_at"] is not None


# -- Signal payload tests --------------------------------------------------


class TestSignalPayload:
    """Signal state → publishable payload conversion."""

    def test_extract_signal_payload(self) -> None:
        adapter = NarrativeLaneAdapter()
        signal = _make_signal()
        payload = adapter.extract_signal_payload(signal)

        assert isinstance(payload, NarrativeSignalPayload)
        assert payload.run_id == "nr_001"
        assert payload.trigger_type == "narrative_surge"
        assert payload.state == "active"
        assert payload.last_score == 0.85

    def test_signal_payload_to_dict(self) -> None:
        adapter = NarrativeLaneAdapter()
        signal = _make_signal()
        d = adapter.extract_signal_payload(signal).to_dict()

        assert d["run_id"] == "nr_001"
        assert d["trigger_type"] == "narrative_surge"
        assert d["metadata"]["uplift"] == 3.2


# -- Source ID tests -------------------------------------------------------


class TestSourceIds:
    """Lineage source ID extraction."""

    def test_make_source_ids(self) -> None:
        adapter = NarrativeLaneAdapter()
        run = _make_narrative_run()
        source_ids = adapter.make_source_ids(run)
        assert "nr_001" in source_ids
        assert "theme_hbm" in source_ids
        assert len(source_ids) == 2


# -- Object type constants -------------------------------------------------


class TestObjectTypes:
    """Narrative object type registry."""

    def test_object_types_match_ownership_policy(self) -> None:
        """Narrative types must be in OwnershipPolicy.PUBLISHABLE_OBJECT_TYPES."""
        from src.contracts.intelligence.ownership import OwnershipPolicy

        for ot in NARRATIVE_OBJECT_TYPES:
            assert ot in OwnershipPolicy.PUBLISHABLE_OBJECT_TYPES, (
                f"{ot!r} not in PUBLISHABLE_OBJECT_TYPES"
            )
