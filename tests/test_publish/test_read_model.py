"""Tests for the read-model builder.

Verifies that published objects are materialized into stable,
consumer-friendly read-model records with proper denormalization,
filtering, and deterministic IDs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.contracts.intelligence.db_schemas import Manifest, PublishedObject
from src.publish.read_model import (
    ReadModelBuilder,
    ReadModelRecord,
    make_record_id,
)

MIGRATION_PATH = Path("migrations/028_read_model.sql")
NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Helpers ---------------------------------------------------------------


def _make_manifest(
    manifest_id: str = "mfst_001",
    **overrides,
) -> Manifest:
    defaults = {
        "manifest_id": manifest_id,
        "lane": "narrative",
        "run_id": "run_001",
        "contract_version": "0.1.0",
        "published_at": NOW,
        "object_count": 3,
        "checksum": "sha256_abc",
    }
    defaults.update(overrides)
    return Manifest(**defaults)


def _make_object(
    object_id: str = "obj_001",
    object_type: str = "claim",
    publish_state: str = "published",
    **overrides,
) -> PublishedObject:
    defaults = {
        "object_id": object_id,
        "object_type": object_type,
        "manifest_id": "mfst_001",
        "lane": "narrative",
        "publish_state": publish_state,
        "contract_version": "0.1.0",
        "source_ids": ["doc_1"],
        "run_id": "run_001",
        "payload": {"subject": "TSMC", "predicate": "supplies_to"},
        "lineage": {"extraction_method": "rule"},
    }
    defaults.update(overrides)
    return PublishedObject(**defaults)


# -- make_record_id tests --------------------------------------------------


class TestMakeRecordId:
    """Deterministic record ID generation."""

    def test_deterministic(self) -> None:
        id1 = make_record_id("mfst_001", "obj_001")
        id2 = make_record_id("mfst_001", "obj_001")
        assert id1 == id2

    def test_prefix(self) -> None:
        rid = make_record_id("mfst_001", "obj_001")
        assert rid.startswith("rm_")

    def test_different_inputs(self) -> None:
        id1 = make_record_id("mfst_001", "obj_001")
        id2 = make_record_id("mfst_001", "obj_002")
        assert id1 != id2

    def test_different_manifests(self) -> None:
        id1 = make_record_id("mfst_001", "obj_001")
        id2 = make_record_id("mfst_002", "obj_001")
        assert id1 != id2


# -- ReadModelBuilder.build_record tests -----------------------------------


class TestBuildRecord:
    """Single record materialization."""

    def test_basic_record(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        obj = _make_object()
        record = builder.build_record(manifest, obj)

        assert isinstance(record, ReadModelRecord)
        assert record.manifest_id == "mfst_001"
        assert record.object_id == "obj_001"
        assert record.object_type == "claim"
        assert record.lane == "narrative"
        assert record.contract_version == "0.1.0"
        assert record.published_at == NOW

    def test_record_id_is_deterministic(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        obj = _make_object()
        r1 = builder.build_record(manifest, obj)
        r2 = builder.build_record(manifest, obj)
        assert r1.record_id == r2.record_id

    def test_carries_payload(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        obj = _make_object(payload={"key": "value"})
        record = builder.build_record(manifest, obj)
        assert record.payload["key"] == "value"

    def test_carries_lineage(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        obj = _make_object(lineage={"source": "test"})
        record = builder.build_record(manifest, obj)
        assert "source" in record.lineage

    def test_carries_source_ids(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        obj = _make_object(source_ids=["doc_1", "doc_2"])
        record = builder.build_record(manifest, obj)
        assert record.source_ids == ["doc_1", "doc_2"]

    def test_carries_validity_window(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 12, 31, tzinfo=UTC)
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        obj = _make_object(valid_from=t1, valid_to=t2)
        record = builder.build_record(manifest, obj)
        assert record.valid_from == t1
        assert record.valid_to == t2

    def test_rejects_non_publishable_type(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        obj = _make_object(object_type="unknown_type")
        # Bypass PublishedObject validation by patching
        object.__setattr__(obj, "object_type", "unknown_type")
        with pytest.raises(ValueError, match="not publishable"):
            builder.build_record(manifest, obj)

    def test_all_publishable_types_accepted(self) -> None:
        from src.contracts.intelligence.ownership import OwnershipPolicy

        builder = ReadModelBuilder()
        manifest = _make_manifest()
        for ot in OwnershipPolicy.PUBLISHABLE_OBJECT_TYPES:
            obj = _make_object(object_type=ot)
            record = builder.build_record(manifest, obj)
            assert record.object_type == ot


# -- ReadModelBuilder.build tests ------------------------------------------


class TestBuild:
    """Batch record materialization."""

    def test_build_multiple(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        objects = [
            _make_object("obj_1", "claim"),
            _make_object("obj_2", "assertion"),
            _make_object("obj_3", "signal"),
        ]
        records = builder.build(manifest, objects)
        assert len(records) == 3
        types = {r.object_type for r in records}
        assert types == {"claim", "assertion", "signal"}

    def test_skips_non_published(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        objects = [
            _make_object("obj_1", publish_state="published"),
            _make_object("obj_2", publish_state="draft"),
            _make_object("obj_3", publish_state="retracted"),
        ]
        records = builder.build(manifest, objects)
        assert len(records) == 1
        assert records[0].object_id == "obj_1"

    def test_include_non_published(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        objects = [
            _make_object("obj_1", publish_state="published"),
            _make_object("obj_2", publish_state="draft"),
        ]
        records = builder.build(manifest, objects, published_only=False)
        assert len(records) == 2

    def test_empty_manifest(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        records = builder.build(manifest, [])
        assert records == []


# -- ReadModelBuilder.build_summary tests ----------------------------------


class TestBuildSummary:
    """Materialization summary for audit logging."""

    def test_summary_structure(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        records = [
            ReadModelRecord(
                record_id="rm_1", manifest_id="mfst_001",
                object_id="obj_1", object_type="claim",
                lane="narrative", contract_version="0.1.0",
            ),
            ReadModelRecord(
                record_id="rm_2", manifest_id="mfst_001",
                object_id="obj_2", object_type="claim",
                lane="narrative", contract_version="0.1.0",
            ),
            ReadModelRecord(
                record_id="rm_3", manifest_id="mfst_001",
                object_id="obj_3", object_type="assertion",
                lane="narrative", contract_version="0.1.0",
            ),
        ]
        summary = builder.build_summary(manifest, records)
        assert summary["manifest_id"] == "mfst_001"
        assert summary["total_records"] == 3
        assert summary["by_object_type"]["claim"] == 2
        assert summary["by_object_type"]["assertion"] == 1

    def test_empty_summary(self) -> None:
        builder = ReadModelBuilder()
        manifest = _make_manifest()
        summary = builder.build_summary(manifest, [])
        assert summary["total_records"] == 0
        assert summary["by_object_type"] == {}


# -- Migration structural tests -------------------------------------------


class TestMigration028:
    """Structural validation of migration 028."""

    @staticmethod
    def _load_sql() -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_read_model_table(self) -> None:
        sql = self._load_sql()
        assert "CREATE TABLE IF NOT EXISTS intel_pub.read_model" in sql

    def test_lane_check(self) -> None:
        sql = self._load_sql()
        for lane in ("narrative", "filing", "structural", "backtest"):
            assert lane in sql

    def test_manifest_index(self) -> None:
        sql = self._load_sql()
        assert "idx_read_model_manifest" in sql

    def test_lane_index(self) -> None:
        sql = self._load_sql()
        assert "idx_read_model_lane" in sql

    def test_type_index(self) -> None:
        sql = self._load_sql()
        assert "idx_read_model_type" in sql

    def test_unique_manifest_object(self) -> None:
        sql = self._load_sql()
        assert "idx_read_model_manifest_object" in sql
        assert "UNIQUE" in sql

    def test_jsonb_columns(self) -> None:
        sql = self._load_sql()
        for col in ("payload", "lineage", "metadata"):
            assert col in sql
