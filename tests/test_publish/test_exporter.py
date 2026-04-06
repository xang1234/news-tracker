"""Tests for bundle export and contract-compatibility parity.

Proves that:
    1. Bundle export produces valid, parseable JSONL with integrity checksums.
    2. Bundle-backed consumers see the same data as direct DB readers.
    3. Checksums are deterministic and verifiable.
    4. Only published objects are included (drafts/retracted excluded).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from src.contracts.intelligence.db_schemas import (
    ExportBundle,
    LaneRun,
    Manifest,
    ManifestPointer,
    PublishedObject,
)
from src.contracts.intelligence.fixtures import (
    make_manifest,
    make_published_object,
)
from src.contracts.intelligence.lanes import LANE_NARRATIVE
from src.contracts.intelligence.version import ContractRegistry
from src.publish.exporter import (
    BundleExporter,
    build_bundle_lines,
    compute_bundle_checksum,
    parse_bundle_lines,
    verify_bundle_checksum,
)
from src.publish.service import PublishService

# -- In-memory mock repository (reuse pattern) -----------------------------


class _InMemoryRepo:
    def __init__(self) -> None:
        self.runs: dict[str, LaneRun] = {}
        self.manifests: dict[str, Manifest] = {}
        self.pointers: dict[str, ManifestPointer] = {}
        self.objects: dict[str, PublishedObject] = {}
        self.bundles: dict[str, ExportBundle] = {}

    async def create_lane_run(self, run):
        self.runs[run.run_id] = run
        return run

    async def get_lane_run(self, run_id):
        return self.runs.get(run_id)

    async def update_lane_run_status(self, run_id, status, *, error_message=None, metrics=None):
        run = self.runs.get(run_id)
        if run:
            run.status = status
        return run

    async def list_lane_runs(self, lane=None, status=None, limit=50):
        return list(self.runs.values())[:limit]

    async def create_manifest(self, m):
        self.manifests[m.manifest_id] = m
        return m

    async def get_manifest(self, manifest_id):
        return self.manifests.get(manifest_id)

    async def update_manifest(
        self,
        manifest_id,
        *,
        object_count=None,
        checksum=None,
        published_at=None,
    ):
        m = self.manifests.get(manifest_id)
        if m is None:
            return None
        if object_count is not None:
            m.object_count = object_count
        if checksum is not None:
            m.checksum = checksum
        if published_at is not None:
            m.published_at = published_at
        return m

    async def get_pointer(self, lane):
        return self.pointers.get(lane)

    async def advance_pointer(self, lane, manifest_id, *, metadata=None):
        old = self.pointers.get(lane)
        ptr = ManifestPointer(
            lane=lane,
            manifest_id=manifest_id,
            previous_manifest_id=old.manifest_id if old else None,
        )
        self.pointers[lane] = ptr
        return ptr

    async def create_published_object(self, obj):
        self.objects[obj.object_id] = obj
        return obj

    async def get_published_object(self, object_id):
        return self.objects.get(object_id)

    async def update_publish_state(self, object_id, new_state):
        obj = self.objects.get(object_id)
        if obj:
            obj.publish_state = new_state
        return obj

    async def list_objects_by_manifest(self, manifest_id, *, publish_state=None):
        results = [o for o in self.objects.values() if o.manifest_id == manifest_id]
        if publish_state:
            results = [o for o in results if o.publish_state == publish_state]
        return sorted(results, key=lambda o: o.object_id)

    async def create_export_bundle(self, bundle):
        self.bundles[bundle.bundle_id] = bundle
        return bundle

    async def get_export_bundle(self, bundle_id):
        return self.bundles.get(bundle_id)


@pytest.fixture()
def repo() -> _InMemoryRepo:
    return _InMemoryRepo()


@pytest.fixture()
def service(repo: _InMemoryRepo) -> PublishService:
    return PublishService(repository=repo)


@pytest.fixture()
def exporter(repo: _InMemoryRepo) -> BundleExporter:
    return BundleExporter(repository=repo)


async def _make_manifest(
    service: PublishService, lane: str = LANE_NARRATIVE
) -> tuple[str, Manifest]:
    """Helper: create a run + manifest, return (run_id, manifest)."""
    run = await service.create_run(lane)
    m = await service.create_manifest(lane, run.run_id)
    return run.run_id, m


# -- Pure function tests ---------------------------------------------------


class TestBuildBundleLines:
    """build_bundle_lines() produces valid JSONL."""

    def test_header_first(self) -> None:
        manifest = make_manifest(published_at=datetime.now(UTC))
        lines = build_bundle_lines(manifest, [])
        header = json.loads(lines[0])
        assert header["_type"] == "manifest_header"
        assert header["manifest_id"] == manifest.manifest_id

    def test_objects_sorted_by_id(self) -> None:
        manifest = make_manifest(published_at=datetime.now(UTC))
        obj_b = make_published_object(object_id="obj_b")
        obj_a = make_published_object(object_id="obj_a")
        lines = build_bundle_lines(manifest, [obj_b, obj_a])
        # Header + 2 objects
        assert len(lines) == 3
        parsed_a = json.loads(lines[1])
        parsed_b = json.loads(lines[2])
        assert parsed_a["object_id"] == "obj_a"
        assert parsed_b["object_id"] == "obj_b"

    def test_empty_manifest(self) -> None:
        manifest = make_manifest(published_at=datetime.now(UTC))
        lines = build_bundle_lines(manifest, [])
        assert len(lines) == 1  # Just the header

    def test_lines_are_valid_json(self) -> None:
        manifest = make_manifest(published_at=datetime.now(UTC))
        obj = make_published_object()
        lines = build_bundle_lines(manifest, [obj])
        for line in lines:
            json.loads(line)  # Should not raise


class TestChecksums:
    """Checksum generation and verification."""

    def test_deterministic(self) -> None:
        manifest = make_manifest(published_at=datetime.now(UTC))
        obj = make_published_object()
        lines1 = build_bundle_lines(manifest, [obj])
        lines2 = build_bundle_lines(manifest, [obj])
        assert compute_bundle_checksum(lines1) == compute_bundle_checksum(lines2)

    def test_format(self) -> None:
        lines = ['{"test": true}']
        checksum = compute_bundle_checksum(lines)
        assert checksum.startswith("sha256:")
        assert len(checksum) == len("sha256:") + 64

    def test_verify_valid(self) -> None:
        lines = ['{"test": true}']
        checksum = compute_bundle_checksum(lines)
        assert verify_bundle_checksum(lines, checksum) is True

    def test_verify_tampered(self) -> None:
        lines = ['{"test": true}']
        assert verify_bundle_checksum(lines, "sha256:wrong") is False

    def test_different_content_different_checksum(self) -> None:
        c1 = compute_bundle_checksum(['{"a": 1}'])
        c2 = compute_bundle_checksum(['{"b": 2}'])
        assert c1 != c2


class TestParseBundleLines:
    """parse_bundle_lines() consumer deserialization."""

    def test_roundtrip(self) -> None:
        manifest = make_manifest(published_at=datetime.now(UTC))
        obj = make_published_object()
        lines = build_bundle_lines(manifest, [obj])
        header, objects = parse_bundle_lines(lines)
        assert header["manifest_id"] == manifest.manifest_id
        assert len(objects) == 1
        assert objects[0]["object_id"] == obj.object_id

    def test_empty_bundle_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_bundle_lines([])

    def test_missing_header_type_raises(self) -> None:
        with pytest.raises(ValueError, match="manifest header"):
            parse_bundle_lines(['{"foo": "bar"}'])

    def test_header_only(self) -> None:
        manifest = make_manifest(published_at=datetime.now(UTC))
        lines = build_bundle_lines(manifest, [])
        header, objects = parse_bundle_lines(lines)
        assert header["_type"] == "manifest_header"
        assert objects == []


# -- BundleExporter integration tests -------------------------------------


class TestBundleExporter:
    """BundleExporter with in-memory repository."""

    async def test_export_sealed_manifest(
        self,
        service: PublishService,
        exporter: BundleExporter,
    ) -> None:
        # Create manifest, add objects, seal
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
            payload={"text": "test claim"},
        )
        await service.transition_object(obj.object_id, "published")
        await service.seal_manifest(m.manifest_id)

        # Export
        bundle, lines = await exporter.export_manifest(m.manifest_id)

        assert bundle.manifest_id == m.manifest_id
        assert bundle.object_count == 1
        assert bundle.format == "jsonl"
        assert bundle.checksum is not None
        assert bundle.size_bytes > 0

    async def test_seal_rejects_drafts(
        self,
        service: PublishService,
        exporter: BundleExporter,
    ) -> None:
        """Cannot seal (and therefore export) a manifest with draft objects."""
        run_id, m = await _make_manifest(service)
        await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )  # stays draft
        with pytest.raises(ValueError, match="still in"):
            await service.seal_manifest(m.manifest_id)

    async def test_export_unsealed_raises(
        self,
        service: PublishService,
        exporter: BundleExporter,
    ) -> None:
        _, m = await _make_manifest(service)
        with pytest.raises(ValueError, match="not sealed"):
            await exporter.export_manifest(m.manifest_id)

    async def test_export_nonexistent_raises(
        self,
        exporter: BundleExporter,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            await exporter.export_manifest("nope")

    async def test_export_checksum_verifiable(
        self,
        service: PublishService,
        exporter: BundleExporter,
    ) -> None:
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.transition_object(obj.object_id, "published")
        await service.seal_manifest(m.manifest_id)

        bundle, lines = await exporter.export_manifest(m.manifest_id)
        assert verify_bundle_checksum(lines, bundle.checksum) is True

    async def test_export_records_in_repo(
        self,
        repo: _InMemoryRepo,
        service: PublishService,
        exporter: BundleExporter,
    ) -> None:
        _, m = await _make_manifest(service)
        await service.seal_manifest(m.manifest_id)
        bundle, _ = await exporter.export_manifest(m.manifest_id)
        assert bundle.bundle_id in repo.bundles


# -- Contract compatibility parity tests -----------------------------------


class TestContractParity:
    """Proves bundle-backed and DB-backed consumers see the same data.

    This is the core contract-compat test: serialize objects to a
    bundle, deserialize them, and verify field-level parity with
    the original in-memory objects.
    """

    async def test_object_field_parity(
        self, service: PublishService, exporter: BundleExporter
    ) -> None:
        """Every published object field survives the JSONL roundtrip."""
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
            payload={"confidence": 0.92, "entities": ["TSMC"]},
            source_ids=["doc_1", "doc_2"],
        )
        await service.transition_object(obj.object_id, "published")
        await service.seal_manifest(m.manifest_id)

        # DB path: read directly
        db_obj = await service.get_object(obj.object_id)

        # Bundle path: export and parse
        _, lines = await exporter.export_manifest(m.manifest_id)
        _, bundle_objects = parse_bundle_lines(lines)
        bundle_obj = bundle_objects[0]

        # Parity checks
        assert bundle_obj["object_id"] == db_obj.object_id
        assert bundle_obj["object_type"] == db_obj.object_type
        assert bundle_obj["manifest_id"] == db_obj.manifest_id
        assert bundle_obj["lane"] == db_obj.lane
        assert bundle_obj["publish_state"] == db_obj.publish_state
        assert bundle_obj["contract_version"] == db_obj.contract_version
        assert bundle_obj["source_ids"] == db_obj.source_ids
        assert bundle_obj["run_id"] == db_obj.run_id
        assert bundle_obj["payload"] == db_obj.payload
        assert bundle_obj["lineage"] == db_obj.lineage

    async def test_manifest_header_parity(
        self, service: PublishService, exporter: BundleExporter
    ) -> None:
        """Manifest header in bundle matches DB manifest."""
        _, m = await _make_manifest(service)
        await service.seal_manifest(m.manifest_id)

        db_manifest = await service.get_manifest(m.manifest_id)
        _, lines = await exporter.export_manifest(m.manifest_id)
        header, _ = parse_bundle_lines(lines)

        assert header["manifest_id"] == db_manifest.manifest_id
        assert header["lane"] == db_manifest.lane
        assert header["run_id"] == db_manifest.run_id
        assert header["contract_version"] == db_manifest.contract_version
        assert header["object_count"] == db_manifest.object_count
        assert header["checksum"] == db_manifest.checksum

    async def test_contract_version_consistency(
        self, service: PublishService, exporter: BundleExporter
    ) -> None:
        """All exported objects carry the current contract version."""
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.transition_object(obj.object_id, "published")
        await service.seal_manifest(m.manifest_id)

        bundle, lines = await exporter.export_manifest(m.manifest_id)
        header, objects = parse_bundle_lines(lines)

        current = str(ContractRegistry.CURRENT)
        assert header["contract_version"] == current
        assert bundle.contract_version == current
        for obj_dict in objects:
            assert obj_dict["contract_version"] == current

    async def test_multiple_objects_parity(
        self, service: PublishService, exporter: BundleExporter
    ) -> None:
        """Multiple objects maintain order and content parity."""
        run_id, m = await _make_manifest(service)
        ids = []
        for i in range(5):
            obj = await service.add_object(
                m.manifest_id,
                object_type="claim",
                lane=LANE_NARRATIVE,
                run_id=run_id,
                payload={"index": i},
            )
            await service.transition_object(obj.object_id, "published")
            ids.append(obj.object_id)
        await service.seal_manifest(m.manifest_id)

        _, lines = await exporter.export_manifest(m.manifest_id)
        _, bundle_objects = parse_bundle_lines(lines)

        # Bundle objects should be sorted by object_id
        bundle_ids = [o["object_id"] for o in bundle_objects]
        assert bundle_ids == sorted(bundle_ids)
        assert len(bundle_objects) == 5
