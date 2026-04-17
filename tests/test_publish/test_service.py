"""Tests for PublishService — lifecycle and pointer semantics.

Uses an in-memory mock repository so that state machine logic and
pointer advancement can be exercised without a database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.contracts.intelligence.db_schemas import (
    LaneRun,
    Manifest,
    ManifestPointer,
    PublishedObject,
)
from src.contracts.intelligence.lanes import LANE_FILING, LANE_NARRATIVE
from src.publish.read_model import ReadModelRecord
from src.publish.service import PublishService

# -- In-memory mock repository ---------------------------------------------


class InMemoryPublishRepository:
    """Minimal in-memory store for testing service logic."""

    def __init__(self) -> None:
        self.runs: dict[str, LaneRun] = {}
        self.manifests: dict[str, Manifest] = {}
        self.pointers: dict[str, ManifestPointer] = {}
        self.objects: dict[str, PublishedObject] = {}
        self.read_model_records: dict[tuple[str, str], ReadModelRecord] = {}

    async def create_lane_run(self, run: LaneRun) -> LaneRun:
        self.runs[run.run_id] = run
        return run

    async def get_lane_run(self, run_id: str) -> LaneRun | None:
        return self.runs.get(run_id)

    async def update_lane_run_status(
        self,
        run_id: str,
        status: str,
        *,
        error_message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> LaneRun | None:
        run = self.runs.get(run_id)
        if run is None:
            return None
        now = datetime.now(UTC)
        run.status = status
        if status == "running" and run.started_at is None:
            run.started_at = now
        if status in ("completed", "failed", "cancelled"):
            run.completed_at = now
        if error_message is not None:
            run.error_message = error_message
        if metrics is not None:
            run.metrics = metrics
        run.updated_at = now
        return run

    async def create_manifest(self, manifest: Manifest) -> Manifest:
        self.manifests[manifest.manifest_id] = manifest
        return manifest

    async def get_manifest(self, manifest_id: str) -> Manifest | None:
        return self.manifests.get(manifest_id)

    async def update_manifest(
        self,
        manifest_id: str,
        *,
        object_count: int | None = None,
        checksum: str | None = None,
        published_at: datetime | None = None,
    ) -> Manifest | None:
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

    async def get_pointer(self, lane: str) -> ManifestPointer | None:
        return self.pointers.get(lane)

    async def advance_pointer(
        self,
        lane: str,
        manifest_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ManifestPointer:
        old = self.pointers.get(lane)
        ptr = ManifestPointer(
            lane=lane,
            manifest_id=manifest_id,
            previous_manifest_id=old.manifest_id if old else None,
            metadata=metadata or {},
        )
        self.pointers[lane] = ptr
        return ptr

    async def create_published_object(self, obj: PublishedObject) -> PublishedObject:
        self.objects[obj.object_id] = obj
        return obj

    async def get_published_object(self, object_id: str) -> PublishedObject | None:
        return self.objects.get(object_id)

    async def update_publish_state(self, object_id: str, new_state: str) -> PublishedObject | None:
        obj = self.objects.get(object_id)
        if obj is None:
            return None
        obj.publish_state = new_state
        obj.updated_at = datetime.now(UTC)
        return obj

    async def list_objects_by_manifest(
        self,
        manifest_id: str,
        *,
        publish_state: str | None = None,
    ) -> list[PublishedObject]:
        results = [o for o in self.objects.values() if o.manifest_id == manifest_id]
        if publish_state is not None:
            results = [o for o in results if o.publish_state == publish_state]
        return sorted(results, key=lambda o: o.created_at)

    async def upsert_records(self, records: list[ReadModelRecord]) -> int:
        for record in records:
            self.read_model_records[(record.manifest_id, record.object_id)] = record
        return len(records)

    async def count_records_for_manifest(self, manifest_id: str) -> int:
        return sum(1 for manifest_key, _ in self.read_model_records if manifest_key == manifest_id)


# -- Fixtures --------------------------------------------------------------


@pytest.fixture()
def repo() -> InMemoryPublishRepository:
    return InMemoryPublishRepository()


@pytest.fixture()
def service(repo: InMemoryPublishRepository) -> PublishService:
    return PublishService(repository=repo, read_model_repository=repo)


# -- Lane run lifecycle tests ----------------------------------------------


class TestLaneRunLifecycle:
    """Full lane run lifecycle: create → start → complete/fail."""

    async def test_create_run(self, service: PublishService) -> None:
        run = await service.create_run(LANE_NARRATIVE)
        assert run.lane == LANE_NARRATIVE
        assert run.status == "pending"
        assert run.run_id.startswith("run_")

    async def test_full_lifecycle_success(self, service: PublishService) -> None:
        run = await service.create_run(LANE_NARRATIVE)
        started = await service.start_run(run.run_id)
        assert started.status == "running"
        assert started.started_at is not None

        completed = await service.complete_run(run.run_id, metrics={"docs_processed": 42})
        assert completed.status == "completed"
        assert completed.completed_at is not None
        assert completed.metrics["docs_processed"] == 42

    async def test_full_lifecycle_failure(self, service: PublishService) -> None:
        run = await service.create_run(LANE_FILING)
        await service.start_run(run.run_id)
        failed = await service.fail_run(run.run_id, "connection timeout")
        assert failed.status == "failed"
        assert failed.error_message == "connection timeout"

    async def test_cancel_pending_run(self, service: PublishService) -> None:
        run = await service.create_run(LANE_NARRATIVE)
        cancelled = await service.cancel_run(run.run_id)
        assert cancelled.status == "cancelled"

    async def test_cannot_start_completed_run(self, service: PublishService) -> None:
        run = await service.create_run(LANE_NARRATIVE)
        await service.start_run(run.run_id)
        await service.complete_run(run.run_id)
        with pytest.raises(ValueError, match="Invalid run transition"):
            await service.start_run(run.run_id)

    async def test_cannot_complete_pending_run(self, service: PublishService) -> None:
        run = await service.create_run(LANE_NARRATIVE)
        with pytest.raises(ValueError, match="Invalid run transition"):
            await service.complete_run(run.run_id)

    async def test_nonexistent_run_raises(self, service: PublishService) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.start_run("run_does_not_exist")

    async def test_invalid_lane_rejected(self, service: PublishService) -> None:
        with pytest.raises(ValueError, match="Unknown lane"):
            await service.create_run("nonexistent_lane")

    async def test_get_run(self, service: PublishService) -> None:
        run = await service.create_run(LANE_NARRATIVE)
        fetched = await service.get_run(run.run_id)
        assert fetched is not None
        assert fetched.run_id == run.run_id

    async def test_get_nonexistent_run(self, service: PublishService) -> None:
        assert await service.get_run("nope") is None


# -- Manifest lifecycle tests ----------------------------------------------


async def _make_run(service: PublishService, lane: str) -> str:
    """Helper: create a lane run and return its run_id."""
    run = await service.create_run(lane)
    return run.run_id


class TestManifestLifecycle:
    """Manifest create → seal workflow."""

    async def test_create_manifest(self, service: PublishService) -> None:
        run_id = await _make_run(service, LANE_NARRATIVE)
        m = await service.create_manifest(LANE_NARRATIVE, run_id)
        assert m.manifest_id.startswith("manifest_")
        assert m.lane == LANE_NARRATIVE
        assert m.run_id == run_id
        assert m.object_count == 0
        assert m.checksum is None

    async def test_seal_manifest(self, service: PublishService) -> None:
        run_id = await _make_run(service, LANE_NARRATIVE)
        m = await service.create_manifest(LANE_NARRATIVE, run_id)
        sealed = await service.seal_manifest(m.manifest_id)
        assert sealed.object_count == 0
        assert sealed.checksum is not None
        assert sealed.published_at is not None

    async def test_seal_manifest_with_objects(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.transition_object(obj.object_id, "published")
        sealed = await service.seal_manifest(m.manifest_id)
        assert sealed.object_count == 1
        assert await service._count_materialized_read_model_records(m.manifest_id) == 1

    async def test_seal_rejects_non_published_objects(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )  # stays draft
        with pytest.raises(ValueError, match="still in"):
            await service.seal_manifest(m.manifest_id)

    async def test_seal_nonexistent_manifest_raises(self, service: PublishService) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.seal_manifest("nope")

    async def test_create_manifest_invalid_lane(self, service: PublishService) -> None:
        with pytest.raises(ValueError, match="Unknown lane"):
            await service.create_manifest("bad_lane", "run_001")

    async def test_create_manifest_run_not_found(self, service: PublishService) -> None:
        with pytest.raises(ValueError, match="Lane run not found"):
            await service.create_manifest(LANE_NARRATIVE, "nonexistent")

    async def test_create_manifest_run_lane_mismatch(self, service: PublishService) -> None:
        run_id = await _make_run(service, LANE_FILING)
        with pytest.raises(ValueError, match="belongs to lane"):
            await service.create_manifest(LANE_NARRATIVE, run_id)


# -- Pointer advancement tests ---------------------------------------------


async def _create_and_seal(
    service: PublishService, lane: str, run_id: str | None = None
) -> Manifest:
    """Helper: create a completed run + sealed manifest for publishing."""
    if run_id is None:
        run = await service.create_run(lane)
        await service.start_run(run.run_id)
        await service.complete_run(run.run_id)
        run_id = run.run_id
    m = await service.create_manifest(lane, run_id)
    return await service.seal_manifest(m.manifest_id)


class TestPointerAdvancement:
    """Atomic pointer semantics."""

    async def test_initial_pointer(self, service: PublishService) -> None:
        m = await _create_and_seal(service, LANE_NARRATIVE)
        ptr = await service.advance_pointer(LANE_NARRATIVE, m.manifest_id)
        assert ptr.lane == LANE_NARRATIVE
        assert ptr.manifest_id == m.manifest_id
        assert ptr.previous_manifest_id is None

    async def test_pointer_tracks_previous(self, service: PublishService) -> None:
        m1 = await _create_and_seal(service, LANE_NARRATIVE)
        m2 = await _create_and_seal(service, LANE_NARRATIVE)

        await service.advance_pointer(LANE_NARRATIVE, m1.manifest_id)
        ptr = await service.advance_pointer(LANE_NARRATIVE, m2.manifest_id)

        assert ptr.manifest_id == m2.manifest_id
        assert ptr.previous_manifest_id == m1.manifest_id

    async def test_pointer_per_lane_isolation(self, service: PublishService) -> None:
        m_narr = await _create_and_seal(service, LANE_NARRATIVE)
        m_file = await _create_and_seal(service, LANE_FILING)

        await service.advance_pointer(LANE_NARRATIVE, m_narr.manifest_id)
        await service.advance_pointer(LANE_FILING, m_file.manifest_id)

        ptr_narr = await service.get_pointer(LANE_NARRATIVE)
        ptr_file = await service.get_pointer(LANE_FILING)

        assert ptr_narr is not None
        assert ptr_file is not None
        assert ptr_narr.manifest_id == m_narr.manifest_id
        assert ptr_file.manifest_id == m_file.manifest_id

    async def test_pointer_nonexistent_manifest_raises(self, service: PublishService) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.advance_pointer(LANE_NARRATIVE, "nope")

    async def test_pointer_lane_mismatch_raises(self, service: PublishService) -> None:
        m = await _create_and_seal(service, LANE_FILING)
        with pytest.raises(ValueError, match="belongs to lane"):
            await service.advance_pointer(LANE_NARRATIVE, m.manifest_id)

    async def test_pointer_unsealed_manifest_rejected(self, service: PublishService) -> None:
        """Cannot publish an unsealed manifest."""
        run_id = await _make_run(service, LANE_NARRATIVE)
        m = await service.create_manifest(LANE_NARRATIVE, run_id)
        with pytest.raises(ValueError, match="not been sealed"):
            await service.advance_pointer(LANE_NARRATIVE, m.manifest_id)

    async def test_pointer_incomplete_run_rejected(self, service: PublishService) -> None:
        """Cannot publish from a run that hasn't completed."""
        run_id = await _make_run(service, LANE_NARRATIVE)
        m = await service.create_manifest(LANE_NARRATIVE, run_id)
        await service.seal_manifest(m.manifest_id)
        with pytest.raises(ValueError, match="not 'completed'"):
            await service.advance_pointer(LANE_NARRATIVE, m.manifest_id)

    async def test_pointer_requires_materialized_read_model(
        self,
        service: PublishService,
        repo: InMemoryPublishRepository,
    ) -> None:
        run = await service.create_run(LANE_NARRATIVE)
        await service.start_run(run.run_id)
        await service.complete_run(run.run_id)
        manifest = await service.create_manifest(LANE_NARRATIVE, run.run_id)
        obj = await service.add_object(
            manifest.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run.run_id,
        )
        await service.transition_object(obj.object_id, "published")
        manifest = await service.seal_manifest(manifest.manifest_id)
        repo.read_model_records.clear()
        with pytest.raises(ValueError, match="materialized read-model record"):
            await service.advance_pointer(LANE_NARRATIVE, manifest.manifest_id)

    async def test_get_pointer_empty(self, service: PublishService) -> None:
        assert await service.get_pointer(LANE_NARRATIVE) is None

    async def test_three_advances_track_history(self, service: PublishService) -> None:
        """Third pointer advance should reference the second manifest."""
        m1 = await _create_and_seal(service, LANE_NARRATIVE)
        m2 = await _create_and_seal(service, LANE_NARRATIVE)
        m3 = await _create_and_seal(service, LANE_NARRATIVE)

        await service.advance_pointer(LANE_NARRATIVE, m1.manifest_id)
        await service.advance_pointer(LANE_NARRATIVE, m2.manifest_id)
        ptr = await service.advance_pointer(LANE_NARRATIVE, m3.manifest_id)

        assert ptr.manifest_id == m3.manifest_id
        assert ptr.previous_manifest_id == m2.manifest_id


# -- Published object state transition tests --------------------------------


async def _make_manifest(
    service: PublishService, lane: str = LANE_NARRATIVE
) -> tuple[str, Manifest]:
    """Helper: create a run + manifest, return (run_id, manifest)."""
    run_id = await _make_run(service, lane)
    m = await service.create_manifest(lane, run_id)
    return run_id, m


class TestPublishedObjectTransitions:
    """Published object lifecycle: draft → review → published/retracted."""

    async def test_add_object(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
            payload={"text": "TSMC capacity expansion"},
            source_ids=["doc_1", "doc_2"],
        )
        assert obj.object_id.startswith("obj_")
        assert obj.publish_state == "draft"
        assert obj.payload["text"] == "TSMC capacity expansion"
        assert obj.source_ids == ["doc_1", "doc_2"]

    async def test_add_object_nonexistent_manifest_raises(self, service: PublishService) -> None:
        with pytest.raises(ValueError, match="Manifest not found"):
            await service.add_object(
                "nope",
                object_type="claim",
                lane=LANE_NARRATIVE,
                run_id="run_001",
            )

    async def test_add_object_lane_mismatch_raises(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service, LANE_FILING)
        with pytest.raises(ValueError, match="does not match manifest lane"):
            await service.add_object(
                m.manifest_id,
                object_type="claim",
                lane=LANE_NARRATIVE,
                run_id=run_id,
            )

    async def test_add_object_run_mismatch_raises(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        other_run_id = await _make_run(service, LANE_NARRATIVE)
        with pytest.raises(ValueError, match="does not match manifest run_id"):
            await service.add_object(
                m.manifest_id,
                object_type="claim",
                lane=LANE_NARRATIVE,
                run_id=other_run_id,
            )

    async def test_add_object_to_sealed_manifest_raises(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        await service.seal_manifest(m.manifest_id)
        with pytest.raises(ValueError, match="sealed manifest"):
            await service.add_object(
                m.manifest_id,
                object_type="claim",
                lane=LANE_NARRATIVE,
                run_id=run_id,
            )

    async def test_transition_object_in_sealed_manifest_raises(
        self, service: PublishService
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
        # Retraction IS allowed post-seal
        _, retracted = await service.transition_object(obj.object_id, "retracted")
        assert retracted.publish_state == "retracted"

    async def test_non_retract_transition_in_sealed_manifest_raises(
        self, service: PublishService
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
        # Non-retract transitions are blocked post-seal
        with pytest.raises(ValueError, match="sealed"):
            await service.transition_object(obj.object_id, "review")

    async def test_add_object_invalid_type_raises(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        with pytest.raises(ValueError, match="not publishable"):
            await service.add_object(
                m.manifest_id,
                object_type="not_a_real_type",
                lane=LANE_NARRATIVE,
                run_id=run_id,
            )

    async def test_draft_to_review_to_published(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        _, reviewed = await service.transition_object(obj.object_id, "review")
        assert reviewed.publish_state == "review"

        _, published = await service.transition_object(obj.object_id, "published")
        assert published.publish_state == "published"

    async def test_draft_direct_to_published(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="assertion",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        _, published = await service.transition_object(obj.object_id, "published")
        assert published.publish_state == "published"

    async def test_published_to_retracted(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.transition_object(obj.object_id, "published")
        _, retracted = await service.transition_object(obj.object_id, "retracted")
        assert retracted.publish_state == "retracted"

    async def test_retracted_is_terminal(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.transition_object(obj.object_id, "retracted")
        with pytest.raises(ValueError, match="Invalid publish transition"):
            await service.transition_object(obj.object_id, "draft")

    async def test_published_cannot_go_to_draft(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.transition_object(obj.object_id, "published")
        with pytest.raises(ValueError, match="Invalid publish transition"):
            await service.transition_object(obj.object_id, "draft")

    async def test_review_back_to_draft(self, service: PublishService) -> None:
        """Review → draft is allowed (revise workflow)."""
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.transition_object(obj.object_id, "review")
        _, revised = await service.transition_object(obj.object_id, "draft")
        assert revised.publish_state == "draft"

    async def test_transition_nonexistent_raises(self, service: PublishService) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.transition_object("nope", "published")

    async def test_list_manifest_objects(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.add_object(
            m.manifest_id,
            object_type="assertion",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        objects = await service.list_manifest_objects(m.manifest_id)
        assert len(objects) == 2

    async def test_list_manifest_objects_filtered(self, service: PublishService) -> None:
        run_id, m = await _make_manifest(service)
        obj = await service.add_object(
            m.manifest_id,
            object_type="claim",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.add_object(
            m.manifest_id,
            object_type="assertion",
            lane=LANE_NARRATIVE,
            run_id=run_id,
        )
        await service.transition_object(obj.object_id, "published")

        published = await service.list_manifest_objects(m.manifest_id, publish_state="published")
        assert len(published) == 1
        assert published[0].object_id == obj.object_id


# -- Checksum utility tests ------------------------------------------------


class TestComputeChecksum:
    """PublishService.compute_checksum() utility."""

    def _make_obj(self, object_id: str, **kwargs) -> PublishedObject:
        defaults = {
            "object_id": object_id,
            "object_type": "claim",
            "manifest_id": "m1",
            "lane": "narrative",
            "publish_state": "published",
            "run_id": "r1",
        }
        defaults.update(kwargs)
        return PublishedObject(**defaults)

    def test_deterministic(self) -> None:
        objs = [self._make_obj("obj_b"), self._make_obj("obj_a")]
        assert PublishService.compute_checksum(objs) == PublishService.compute_checksum(objs)

    def test_order_independent(self) -> None:
        objs1 = [self._make_obj("obj_b"), self._make_obj("obj_a")]
        objs2 = [self._make_obj("obj_a"), self._make_obj("obj_b")]
        assert PublishService.compute_checksum(objs1) == PublishService.compute_checksum(objs2)

    def test_format(self) -> None:
        result = PublishService.compute_checksum([self._make_obj("obj_1")])
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64

    def test_different_ids_different_checksums(self) -> None:
        assert PublishService.compute_checksum(
            [self._make_obj("obj_a")]
        ) != PublishService.compute_checksum([self._make_obj("obj_b")])

    def test_different_payloads_different_checksums(self) -> None:
        obj1 = self._make_obj("obj_a", payload={"text": "version 1"})
        obj2 = self._make_obj("obj_a", payload={"text": "version 2"})
        assert PublishService.compute_checksum([obj1]) != PublishService.compute_checksum([obj2])
