"""Tests for compatibility fixtures.

Verifies that all fixture factories produce valid instances that
can be serialized and deserialized without errors. These tests
serve as the consumer-side compatibility verification target.
"""

import pytest

from src.contracts.intelligence.fixtures import (
    make_export_bundle,
    make_lane_run,
    make_lineage,
    make_manifest,
    make_manifest_header,
    make_manifest_pointer,
    make_published_object,
    make_published_object_ref,
)
from src.contracts.intelligence.lanes import LANE_FILING, LANE_NARRATIVE
from src.contracts.intelligence.version import ContractRegistry


class TestPydanticFixtures:
    """Pydantic contract model fixtures."""

    def test_lineage_defaults(self) -> None:
        lineage = make_lineage()
        assert lineage.lane == LANE_NARRATIVE
        assert lineage.contract_version == str(ContractRegistry.CURRENT)

    def test_lineage_override(self) -> None:
        lineage = make_lineage(lane=LANE_FILING, run_id="custom_run")
        assert lineage.lane == LANE_FILING
        assert lineage.run_id == "custom_run"

    def test_lineage_roundtrip(self) -> None:
        lineage = make_lineage()
        data = lineage.model_dump()
        restored = type(lineage)(**data)
        assert restored.run_id == lineage.run_id

    def test_manifest_header_defaults(self) -> None:
        header = make_manifest_header()
        assert header.object_count == 5
        assert header.checksum is not None

    def test_manifest_header_roundtrip(self) -> None:
        header = make_manifest_header()
        data = header.model_dump()
        restored = type(header)(**data)
        assert restored.manifest_id == header.manifest_id

    def test_published_object_ref_defaults(self) -> None:
        ref = make_published_object_ref()
        assert ref.object_type == "claim"
        assert ref.publish_state.value == "published"

    def test_published_object_ref_roundtrip(self) -> None:
        ref = make_published_object_ref()
        data = ref.model_dump()
        restored = type(ref)(**data)
        assert restored.object_id == ref.object_id


class TestDbSchemaFixtures:
    """DB-mapped dataclass fixtures."""

    def test_lane_run_defaults(self) -> None:
        run = make_lane_run()
        assert run.status == "completed"
        assert run.lane == LANE_NARRATIVE

    def test_lane_run_override(self) -> None:
        run = make_lane_run(status="running", lane=LANE_FILING)
        assert run.status == "running"
        assert run.lane == LANE_FILING

    def test_manifest_defaults(self) -> None:
        m = make_manifest()
        assert m.object_count == 5
        assert m.contract_version == str(ContractRegistry.CURRENT)

    def test_manifest_pointer_defaults(self) -> None:
        ptr = make_manifest_pointer()
        assert ptr.lane == LANE_NARRATIVE
        assert ptr.previous_manifest_id is None

    def test_published_object_defaults(self) -> None:
        obj = make_published_object()
        assert obj.publish_state == "published"
        assert obj.payload["text"] == "Fixture claim content"
        assert len(obj.source_ids) > 0

    def test_export_bundle_defaults(self) -> None:
        bundle = make_export_bundle()
        assert bundle.format == "jsonl"
        assert bundle.object_count == 5

    def test_all_fixtures_have_contract_version(self) -> None:
        """All fixtures that carry contract_version use the current one."""
        current = str(ContractRegistry.CURRENT)
        assert make_lineage().contract_version == current
        assert make_manifest_header().contract_version == current
        assert make_published_object_ref().contract_version == current
        assert make_lane_run().contract_version == current
        assert make_manifest().contract_version == current
        assert make_published_object().contract_version == current
        assert make_export_bundle().contract_version == current
