"""Tests for intelligence layer DB-mapped schemas."""

import pytest

from src.contracts.intelligence.db_schemas import (
    VALID_EXPORT_FORMATS,
    VALID_PUBLISH_STATES,
    VALID_RUN_STATUSES,
    ExportBundle,
    LaneRun,
    Manifest,
    ManifestPointer,
    PublishedObject,
)
from src.contracts.intelligence.lanes import ALL_LANES, LANE_NARRATIVE


class TestLaneRun:
    """LaneRun dataclass validation."""

    def test_minimal_valid(self) -> None:
        run = LaneRun(run_id="run_001", lane=LANE_NARRATIVE)
        assert run.status == "pending"
        assert run.config_snapshot == {}
        assert run.started_at is None

    def test_all_lanes_accepted(self) -> None:
        for lane in ALL_LANES:
            run = LaneRun(run_id=f"run_{lane}", lane=lane)
            assert run.lane == lane

    def test_all_statuses_accepted(self) -> None:
        for status in VALID_RUN_STATUSES:
            run = LaneRun(run_id="run_001", lane=LANE_NARRATIVE, status=status)
            assert run.status == status

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid lane"):
            LaneRun(run_id="run_001", lane="bad_lane")

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid run status"):
            LaneRun(run_id="run_001", lane=LANE_NARRATIVE, status="bad")


class TestManifest:
    """Manifest dataclass validation."""

    def test_minimal_valid(self) -> None:
        m = Manifest(
            manifest_id="m_001",
            lane=LANE_NARRATIVE,
            run_id="run_001",
            contract_version="0.1.0",
        )
        assert m.object_count == 0
        assert m.checksum is None
        assert m.metadata == {}

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid lane"):
            Manifest(
                manifest_id="m_001",
                lane="bad",
                run_id="run_001",
                contract_version="0.1.0",
            )


class TestManifestPointer:
    """ManifestPointer dataclass validation."""

    def test_minimal_valid(self) -> None:
        ptr = ManifestPointer(lane=LANE_NARRATIVE, manifest_id="m_001")
        assert ptr.previous_manifest_id is None
        assert ptr.metadata == {}

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid lane"):
            ManifestPointer(lane="bad", manifest_id="m_001")


class TestPublishedObject:
    """PublishedObject dataclass validation."""

    def test_minimal_valid(self) -> None:
        obj = PublishedObject(
            object_id="obj_001",
            object_type="claim",
            manifest_id="m_001",
            lane=LANE_NARRATIVE,
        )
        assert obj.publish_state == "draft"
        assert obj.source_ids == []
        assert obj.payload == {}

    def test_all_publish_states_accepted(self) -> None:
        for state in VALID_PUBLISH_STATES:
            obj = PublishedObject(
                object_id="obj_001",
                object_type="claim",
                manifest_id="m_001",
                lane=LANE_NARRATIVE,
                publish_state=state,
            )
            assert obj.publish_state == state

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid lane"):
            PublishedObject(
                object_id="obj_001",
                object_type="claim",
                manifest_id="m_001",
                lane="bad",
            )

    def test_invalid_publish_state_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid publish state"):
            PublishedObject(
                object_id="obj_001",
                object_type="claim",
                manifest_id="m_001",
                lane=LANE_NARRATIVE,
                publish_state="bad",
            )


class TestExportBundle:
    """ExportBundle dataclass validation."""

    def test_minimal_valid(self) -> None:
        bundle = ExportBundle(
            bundle_id="b_001",
            manifest_id="m_001",
            lane=LANE_NARRATIVE,
            contract_version="0.1.0",
        )
        assert bundle.format == "jsonl"
        assert bundle.object_count == 0
        assert bundle.size_bytes is None

    def test_all_formats_accepted(self) -> None:
        for fmt in VALID_EXPORT_FORMATS:
            bundle = ExportBundle(
                bundle_id="b_001",
                manifest_id="m_001",
                lane=LANE_NARRATIVE,
                contract_version="0.1.0",
                format=fmt,
            )
            assert bundle.format == fmt

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid lane"):
            ExportBundle(
                bundle_id="b_001",
                manifest_id="m_001",
                lane="bad",
                contract_version="0.1.0",
            )

    def test_invalid_format_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid export format"):
            ExportBundle(
                bundle_id="b_001",
                manifest_id="m_001",
                lane=LANE_NARRATIVE,
                contract_version="0.1.0",
                format="xml",
            )


class TestStateConstants:
    """State constant sets are complete and consistent."""

    def test_run_statuses_include_terminal_states(self) -> None:
        assert "completed" in VALID_RUN_STATUSES
        assert "failed" in VALID_RUN_STATUSES

    def test_publish_states_include_lifecycle(self) -> None:
        assert "draft" in VALID_PUBLISH_STATES
        assert "published" in VALID_PUBLISH_STATES
        assert "retracted" in VALID_PUBLISH_STATES

    def test_export_formats_include_jsonl(self) -> None:
        assert "jsonl" in VALID_EXPORT_FORMATS
