"""Tests for the internal intelligence API routes.

Uses the PublishService with an in-memory mock repository
to exercise API endpoint logic without a database.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.intel import router
from src.contracts.intelligence.db_schemas import (
    LaneRun,
    Manifest,
    ManifestPointer,
    PublishedObject,
)
from src.contracts.intelligence.lanes import LANE_NARRATIVE
from src.contracts.intelligence.version import ContractRegistry
from src.publish.service import PublishService

# -- Test app setup --------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _make_app(service: PublishService) -> FastAPI:
    """Create a test app with the intel router and injected service."""
    app = FastAPI()
    app.include_router(router)

    # Override dependencies
    from src.api.auth import verify_api_key
    from src.api.dependencies import get_publish_service

    app.dependency_overrides[get_publish_service] = lambda: service
    app.dependency_overrides[verify_api_key] = lambda: "test_key"

    return app


# -- In-memory mock (reuse from test_publish) ------------------------------


class _InMemoryRepo:
    """Minimal in-memory store for route testing."""

    def __init__(self) -> None:
        self.runs: dict[str, LaneRun] = {}
        self.manifests: dict[str, Manifest] = {}
        self.pointers: dict[str, ManifestPointer] = {}
        self.objects: dict[str, PublishedObject] = {}

    async def create_lane_run(self, run: LaneRun) -> LaneRun:
        self.runs[run.run_id] = run
        return run

    async def get_lane_run(self, run_id: str) -> LaneRun | None:
        return self.runs.get(run_id)

    async def update_lane_run_status(self, run_id, status, *, error_message=None, metrics=None):
        run = self.runs.get(run_id)
        if run is None:
            return None
        run.status = status
        run.updated_at = _now()
        return run

    async def list_lane_runs(self, lane=None, status=None, limit=50):
        results = list(self.runs.values())
        if lane:
            results = [r for r in results if r.lane == lane]
        if status:
            results = [r for r in results if r.status == status]
        return results[:limit]

    async def create_manifest(self, m: Manifest) -> Manifest:
        self.manifests[m.manifest_id] = m
        return m

    async def get_manifest(self, manifest_id: str) -> Manifest | None:
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
        if m and object_count is not None:
            m.object_count = object_count
        if m and checksum is not None:
            m.checksum = checksum
        if m and published_at is not None:
            m.published_at = published_at
        return m

    async def get_pointer(self, lane: str) -> ManifestPointer | None:
        return self.pointers.get(lane)

    async def advance_pointer(self, lane, manifest_id, *, metadata=None):
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

    async def update_publish_state(self, object_id, new_state):
        obj = self.objects.get(object_id)
        if obj:
            obj.publish_state = new_state
            obj.updated_at = _now()
        return obj

    async def list_objects_by_manifest(self, manifest_id, *, publish_state=None):
        return [o for o in self.objects.values() if o.manifest_id == manifest_id]


@pytest.fixture()
def service() -> PublishService:
    repo = _InMemoryRepo()
    return PublishService(repository=repo)


@pytest.fixture()
def client(service: PublishService) -> TestClient:
    app = _make_app(service)
    return TestClient(app)


# -- Contract info tests ---------------------------------------------------


class TestContractInfo:
    """GET /intel/contract — no auth required."""

    def test_returns_contract_metadata(self, client: TestClient) -> None:
        resp = client.get("/intel/contract")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_version"] == str(ContractRegistry.CURRENT)
        assert "narrative" in data["lanes"]
        assert "claim" in data["publishable_object_types"]

    def test_compatibility_check_valid(self, client: TestClient) -> None:
        resp = client.get(
            "/intel/contract/compatibility",
            params={"version": str(ContractRegistry.CURRENT)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["compatible"] is True

    def test_compatibility_check_invalid(self, client: TestClient) -> None:
        resp = client.get(
            "/intel/contract/compatibility",
            params={"version": "not_a_version"},
        )
        assert resp.status_code == 422


# -- Lane run metadata tests -----------------------------------------------


class TestRunEndpoints:
    """GET /intel/runs endpoints."""

    def test_get_run_not_found(self, client: TestClient) -> None:
        resp = client.get("/intel/runs/nonexistent")
        assert resp.status_code == 404

    def test_get_run_found(self, client: TestClient, service: PublishService) -> None:
        import asyncio

        run = asyncio.get_event_loop().run_until_complete(service.create_run(LANE_NARRATIVE))
        resp = client.get(f"/intel/runs/{run.run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run.run_id
        assert data["lane"] == LANE_NARRATIVE

    def test_list_runs_empty(self, client: TestClient) -> None:
        resp = client.get("/intel/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_runs_invalid_lane(self, client: TestClient) -> None:
        resp = client.get("/intel/runs", params={"lane": "bad_lane"})
        assert resp.status_code == 422


# -- Manifest endpoint tests -----------------------------------------------


class TestManifestEndpoints:
    """GET /intel/manifests endpoints."""

    def test_get_manifest_not_found(self, client: TestClient) -> None:
        resp = client.get("/intel/manifests/nonexistent")
        assert resp.status_code == 404

    def test_get_manifest_found(self, client: TestClient, service: PublishService) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        run = loop.run_until_complete(service.create_run(LANE_NARRATIVE))
        m = loop.run_until_complete(service.create_manifest(LANE_NARRATIVE, run.run_id))
        resp = client.get(f"/intel/manifests/{m.manifest_id}")
        assert resp.status_code == 200
        assert resp.json()["manifest_id"] == m.manifest_id


# -- Pointer endpoint tests ------------------------------------------------


class TestPointerEndpoints:
    """GET /intel/pointers endpoints."""

    def test_get_pointer_empty(self, client: TestClient) -> None:
        resp = client.get("/intel/pointers/narrative")
        assert resp.status_code == 404

    def test_get_pointer_invalid_lane(self, client: TestClient) -> None:
        resp = client.get("/intel/pointers/bad_lane")
        assert resp.status_code == 422

    def test_get_pointer_after_advance(self, client: TestClient, service: PublishService) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        run = loop.run_until_complete(service.create_run(LANE_NARRATIVE))
        loop.run_until_complete(service.start_run(run.run_id))
        loop.run_until_complete(service.complete_run(run.run_id))
        m = loop.run_until_complete(service.create_manifest(LANE_NARRATIVE, run.run_id))
        loop.run_until_complete(service.seal_manifest(m.manifest_id))
        loop.run_until_complete(service.advance_pointer(LANE_NARRATIVE, m.manifest_id))
        resp = client.get("/intel/pointers/narrative")
        assert resp.status_code == 200
        assert resp.json()["manifest_id"] == m.manifest_id


# -- Review endpoint tests -------------------------------------------------


class TestReviewEndpoints:
    """POST /intel/objects/{id}/review endpoint."""

    def test_review_object_not_found(self, client: TestClient) -> None:
        resp = client.post(
            "/intel/objects/nonexistent/review",
            json={"target_state": "published"},
        )
        assert resp.status_code == 404

    def test_review_valid_transition(self, client: TestClient, service: PublishService) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        run = loop.run_until_complete(service.create_run(LANE_NARRATIVE))
        m = loop.run_until_complete(service.create_manifest(LANE_NARRATIVE, run.run_id))
        obj = loop.run_until_complete(
            service.add_object(
                m.manifest_id,
                object_type="claim",
                lane=LANE_NARRATIVE,
                run_id=run.run_id,
            )
        )
        resp = client.post(
            f"/intel/objects/{obj.object_id}/review",
            json={"target_state": "published"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["previous_state"] == "draft"
        assert data["new_state"] == "published"

    def test_review_invalid_transition(self, client: TestClient, service: PublishService) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        run = loop.run_until_complete(service.create_run(LANE_NARRATIVE))
        m = loop.run_until_complete(service.create_manifest(LANE_NARRATIVE, run.run_id))
        obj = loop.run_until_complete(
            service.add_object(
                m.manifest_id,
                object_type="claim",
                lane=LANE_NARRATIVE,
                run_id=run.run_id,
            )
        )
        # Transition to retracted (terminal)
        loop.run_until_complete(service.transition_object(obj.object_id, "retracted"))
        resp = client.post(
            f"/intel/objects/{obj.object_id}/review",
            json={"target_state": "published"},
        )
        assert resp.status_code == 422

    def test_get_object(self, client: TestClient, service: PublishService) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        run = loop.run_until_complete(service.create_run(LANE_NARRATIVE))
        m = loop.run_until_complete(service.create_manifest(LANE_NARRATIVE, run.run_id))
        obj = loop.run_until_complete(
            service.add_object(
                m.manifest_id,
                object_type="claim",
                lane=LANE_NARRATIVE,
                run_id=run.run_id,
                payload={"text": "test claim"},
            )
        )
        resp = client.get(f"/intel/objects/{obj.object_id}")
        assert resp.status_code == 200
        assert resp.json()["payload"]["text"] == "test claim"
