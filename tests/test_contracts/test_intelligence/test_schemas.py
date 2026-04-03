"""Tests for intelligence contract schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.contracts.intelligence.lanes import LANE_NARRATIVE
from src.contracts.intelligence.schemas import (
    Lineage,
    ManifestHeader,
    PublishedObjectRef,
    PublishState,
    ReviewDecision,
)
from src.contracts.intelligence.version import ContractRegistry


class TestPublishState:
    """PublishState enum."""

    def test_all_states(self) -> None:
        assert set(PublishState) == {
            PublishState.DRAFT,
            PublishState.REVIEW,
            PublishState.PUBLISHED,
            PublishState.RETRACTED,
        }

    def test_string_values(self) -> None:
        assert PublishState.DRAFT == "draft"
        assert PublishState.PUBLISHED == "published"


class TestReviewDecision:
    """ReviewDecision enum."""

    def test_all_decisions(self) -> None:
        assert set(ReviewDecision) == {
            ReviewDecision.APPROVE,
            ReviewDecision.REJECT,
            ReviewDecision.REVISE,
        }


class TestLineage:
    """Lineage schema validation."""

    def test_minimal_valid(self) -> None:
        lineage = Lineage(lane=LANE_NARRATIVE, run_id="run_001")
        assert lineage.lane == LANE_NARRATIVE
        assert lineage.run_id == "run_001"
        assert lineage.contract_version == str(ContractRegistry.CURRENT)
        assert lineage.source_ids == []
        assert lineage.valid_from is None
        assert lineage.valid_to is None

    def test_full_fields(self) -> None:
        lineage = Lineage(
            source_ids=["doc_1", "doc_2"],
            lane=LANE_NARRATIVE,
            run_id="run_002",
        )
        assert lineage.source_ids == ["doc_1", "doc_2"]

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown lane"):
            Lineage(lane="bad_lane", run_id="run_001")

    def test_created_at_auto_populated(self) -> None:
        lineage = Lineage(lane=LANE_NARRATIVE, run_id="run_001")
        assert lineage.created_at is not None

    def test_contract_version_defaults_to_current(self) -> None:
        lineage = Lineage(lane=LANE_NARRATIVE, run_id="run_001")
        assert lineage.contract_version == str(ContractRegistry.CURRENT)

    def test_valid_temporal_window(self) -> None:
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        lineage = Lineage(
            lane=LANE_NARRATIVE,
            run_id="run_001",
            valid_from=t1,
            valid_to=t2,
        )
        assert lineage.valid_from == t1
        assert lineage.valid_to == t2

    def test_same_from_to_allowed(self) -> None:
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        lineage = Lineage(
            lane=LANE_NARRATIVE,
            run_id="run_001",
            valid_from=t,
            valid_to=t,
        )
        assert lineage.valid_from == lineage.valid_to

    def test_inverted_temporal_window_rejected(self) -> None:
        t1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValidationError, match="valid_to.*must not be before"):
            Lineage(
                lane=LANE_NARRATIVE,
                run_id="run_001",
                valid_from=t1,
                valid_to=t2,
            )

    def test_serialization_roundtrip(self) -> None:
        lineage = Lineage(
            source_ids=["doc_1"],
            lane=LANE_NARRATIVE,
            run_id="run_001",
        )
        data = lineage.model_dump()
        restored = Lineage(**data)
        assert restored.lane == lineage.lane
        assert restored.run_id == lineage.run_id
        assert restored.source_ids == lineage.source_ids


class TestManifestHeader:
    """ManifestHeader schema validation."""

    def test_minimal_valid(self) -> None:
        header = ManifestHeader(
            manifest_id="manifest_001",
            lane=LANE_NARRATIVE,
            run_id="run_001",
        )
        assert header.manifest_id == "manifest_001"
        assert header.object_count == 0
        assert header.checksum is None
        assert header.metadata == {}

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown lane"):
            ManifestHeader(
                manifest_id="m_001",
                lane="bad_lane",
                run_id="run_001",
            )

    def test_negative_object_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ManifestHeader(
                manifest_id="m_001",
                lane=LANE_NARRATIVE,
                run_id="run_001",
                object_count=-1,
            )

    def test_metadata_extensible(self) -> None:
        header = ManifestHeader(
            manifest_id="m_001",
            lane=LANE_NARRATIVE,
            run_id="run_001",
            metadata={"coverage_tier": "full", "model_version": "mv_abc123"},
        )
        assert header.metadata["coverage_tier"] == "full"

    def test_serialization_roundtrip(self) -> None:
        header = ManifestHeader(
            manifest_id="m_001",
            lane=LANE_NARRATIVE,
            run_id="run_001",
            object_count=42,
            checksum="sha256:abc123",
        )
        data = header.model_dump()
        restored = ManifestHeader(**data)
        assert restored.manifest_id == header.manifest_id
        assert restored.object_count == 42
        assert restored.checksum == "sha256:abc123"


class TestPublishedObjectRef:
    """PublishedObjectRef schema validation."""

    def test_minimal_valid(self) -> None:
        ref = PublishedObjectRef(
            object_id="obj_001",
            object_type="claim",
            manifest_id="m_001",
            lane=LANE_NARRATIVE,
        )
        assert ref.publish_state == PublishState.DRAFT
        assert ref.contract_version == str(ContractRegistry.CURRENT)

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown lane"):
            PublishedObjectRef(
                object_id="obj_001",
                object_type="claim",
                manifest_id="m_001",
                lane="bad_lane",
            )

    def test_invalid_object_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not publishable"):
            PublishedObjectRef(
                object_id="obj_001",
                object_type="not_a_real_type",
                manifest_id="m_001",
                lane=LANE_NARRATIVE,
            )

    def test_valid_object_types_accepted(self) -> None:
        for obj_type in ("claim", "assertion", "signal"):
            ref = PublishedObjectRef(
                object_id="obj_001",
                object_type=obj_type,
                manifest_id="m_001",
                lane=LANE_NARRATIVE,
            )
            assert ref.object_type == obj_type

    def test_invalid_contract_version_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Invalid contract version"):
            PublishedObjectRef(
                object_id="obj_001",
                object_type="claim",
                manifest_id="m_001",
                lane=LANE_NARRATIVE,
                contract_version="not_a_version",
            )

    def test_publish_state_transition(self) -> None:
        ref = PublishedObjectRef(
            object_id="obj_001",
            object_type="claim",
            manifest_id="m_001",
            lane=LANE_NARRATIVE,
            publish_state=PublishState.PUBLISHED,
        )
        assert ref.publish_state == PublishState.PUBLISHED
