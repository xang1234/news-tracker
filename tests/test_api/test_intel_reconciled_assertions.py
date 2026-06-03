"""Tests for the /intel/assertions/reconciled endpoint.

Surfaces reconciliation results (disputed/corroborated status + support /
contradiction / source-diversity counts) straight from the working
``news_intel.resolved_assertions`` table, so the engine's output is visible
before the publish pipeline materializes it into the read model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.auth import verify_api_key
from src.api.dependencies import get_database
from src.api.routes.intel_surface import router


def _row(assertion_id: str, status: str, *, support: int, contradiction: int) -> dict[str, Any]:
    now = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)
    return {
        "assertion_id": assertion_id,
        "subject_concept_id": "concept_tsmc",
        "predicate": "expands_capacity",
        "object_concept_id": None,
        "confidence": 0.4 if status == "disputed" else 0.8,
        "status": status,
        "valid_from": None,
        "valid_to": None,
        "support_count": support,
        "contradiction_count": contradiction,
        "first_seen_at": now,
        "last_evidence_at": now,
        "source_diversity": 2,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }


class _FakeDB:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.last_sql = ""
        self.last_params: tuple[Any, ...] = ()

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.last_sql = sql
        self.last_params = params
        # Emulate the status filter the repository applies.
        status = next((p for p in params if p in ("disputed", "active")), None)
        if status is None:
            return self._rows
        return [r for r in self._rows if r["status"] == status]


def _client(rows: list[dict[str, Any]]) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_database] = lambda: _FakeDB(rows)
    return TestClient(app)


def test_returns_reconciled_assertions_from_working_table():
    rows = [_row("asrt_disputed", "disputed", support=1, contradiction=1)]
    client = _client(rows)

    resp = client.get("/intel/assertions/reconciled")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    item = body["assertions"][0]
    assert item["assertion_id"] == "asrt_disputed"
    assert item["status"] == "disputed"
    assert item["contradiction_count"] == 1
    assert item["source_diversity"] == 2


def test_status_filter_is_passed_through():
    rows = [
        _row("asrt_disputed", "disputed", support=1, contradiction=1),
        _row("asrt_active", "active", support=3, contradiction=0),
    ]
    client = _client(rows)

    resp = client.get("/intel/assertions/reconciled?status=disputed")

    assert resp.status_code == 200
    body = resp.json()
    assert [a["assertion_id"] for a in body["assertions"]] == ["asrt_disputed"]


def test_reconciled_route_not_shadowed_by_detail_route():
    # "reconciled" must not be captured as an {assertion_id} path param:
    # the detail route would 404; the list route returns an empty list.
    client = _client([])
    resp = client.get("/intel/assertions/reconciled")
    assert resp.status_code == 200
    body = resp.json()
    assert body["assertions"] == []
    assert body["total"] == 0
