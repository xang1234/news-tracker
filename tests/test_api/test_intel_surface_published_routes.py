"""Tests for intel_surface assertion/claim routes using published objects only."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.auth import verify_api_key
from src.api.dependencies import get_database
from src.api.routes.intel_surface import router


def _ts(hour: int) -> datetime:
    return datetime(2026, 4, 16, hour, 0, 0, tzinfo=UTC)


class _FakeDB:
    def __init__(self) -> None:
        self.rows = [
            {
                "object_id": "obj_assert_1",
                "object_type": "assertion",
                "publish_state": "published",
                "contract_version": "1.0.0",
                "lane": "narrative",
                "run_id": "run_1",
                "valid_from": None,
                "valid_to": None,
                "created_at": _ts(12),
                "updated_at": _ts(13),
                "payload": {
                    "assertion_id": "asrt_1",
                    "subject_concept_id": "concept_issuer_nvidia",
                    "predicate": "supplies_to",
                    "object_concept_id": "concept_issuer_msft",
                    "confidence": 0.82,
                    "status": "active",
                    "support_count": 3,
                    "contradiction_count": 1,
                    "source_diversity": 2,
                    "claim_links": [
                        {
                            "claim_id": "claim_1",
                            "link_type": "support",
                            "contribution_weight": 0.9,
                        }
                    ],
                },
            },
            {
                "object_id": "obj_assert_2",
                "object_type": "assertion",
                "publish_state": "published",
                "contract_version": "1.0.0",
                "lane": "narrative",
                "run_id": "run_1",
                "valid_from": None,
                "valid_to": None,
                "created_at": _ts(11),
                "updated_at": _ts(11),
                "payload": {
                    "assertion_id": "asrt_2",
                    "subject_concept_id": "concept_issuer_amd",
                    "predicate": "competes_with",
                    "object_concept_id": "concept_issuer_intel",
                    "confidence": 0.25,
                    "status": "active",
                    "support_count": 1,
                    "contradiction_count": 0,
                    "source_diversity": 1,
                },
            },
            {
                "object_id": "obj_claim_1",
                "object_type": "claim",
                "publish_state": "published",
                "contract_version": "1.0.0",
                "lane": "narrative",
                "run_id": "run_1",
                "valid_from": None,
                "valid_to": None,
                "created_at": _ts(10),
                "updated_at": _ts(10),
                "payload": {
                    "claim_id": "claim_1",
                    "claim_key": "key_1",
                    "source_id": "doc_1",
                    "source_type": "news",
                    "subject_text": "NVIDIA",
                    "predicate": "supplies_to",
                    "object_text": "Microsoft",
                    "confidence": 0.9,
                    "status": "active",
                    "assertion_id": "asrt_1",
                },
            },
            {
                "object_id": "obj_claim_2",
                "object_type": "claim",
                "publish_state": "published",
                "contract_version": "1.0.0",
                "lane": "narrative",
                "run_id": "run_1",
                "valid_from": None,
                "valid_to": None,
                "created_at": _ts(9),
                "updated_at": _ts(9),
                "payload": {
                    "claim_id": "claim_2",
                    "claim_key": "key_2",
                    "source_id": "doc_2",
                    "source_type": "reddit",
                    "subject_text": "NVIDIA",
                    "predicate": "supplies_to",
                    "object_text": "Microsoft",
                    "confidence": 0.7,
                    "status": "active",
                    "assertion_ids": ["asrt_1"],
                },
            },
            {
                "object_id": "obj_claim_3",
                "object_type": "claim",
                "publish_state": "published",
                "contract_version": "1.0.0",
                "lane": "filing",
                "run_id": "run_2",
                "valid_from": None,
                "valid_to": None,
                "created_at": _ts(8),
                "updated_at": _ts(8),
                "payload": {
                    "claim_id": "claim_3",
                    "claim_key": "key_3",
                    "source_id": "doc_3",
                    "source_type": "sec_filing",
                    "subject_text": "Intel",
                    "predicate": "develops_technology",
                    "object_text": "Gaudi",
                    "confidence": 0.6,
                    "status": "active",
                    "assertion_id": "asrt_other",
                },
            },
        ]

    async def fetch(self, query: str, *params):
        if "intel_pub.published_objects" not in query:
            return []

        if "object_type = 'claim'" in query and "payload->>'assertion_id' = $1" not in query:
            rows = [
                r
                for r in self.rows
                if r["object_type"] == "claim" and r["publish_state"] == "published"
            ]
            return sorted(rows, key=lambda r: r["created_at"], reverse=True)

        if not params:
            return []

        object_type = params[0]
        rows = [
            r
            for r in self.rows
            if r["object_type"] == object_type and r["publish_state"] == "published"
        ]

        idx = 1
        if "subject_concept_id" in query and len(params) > idx:
            concept_id = params[idx]
            rows = [
                r
                for r in rows
                if r["payload"].get("subject_concept_id") == concept_id
                or r["payload"].get("object_concept_id") == concept_id
            ]
            idx += 1
        if "payload->>'predicate'" in query and len(params) > idx:
            pred = params[idx]
            rows = [r for r in rows if r["payload"].get("predicate") == pred]
            idx += 1
        if "payload->>'status'" in query and len(params) > idx:
            status = params[idx]
            rows = [r for r in rows if r["payload"].get("status") == status]
            idx += 1
        if "lane =" in query and len(params) > idx:
            lane = params[idx]
            rows = [r for r in rows if r["lane"] == lane]
            idx += 1
        if "payload->>'source_id'" in query and len(params) > idx:
            source_id = params[idx]
            rows = [r for r in rows if r["payload"].get("source_id") == source_id]

        return sorted(rows, key=lambda r: r["created_at"], reverse=True)

    async def fetchrow(self, query: str, *params):
        if "intel_pub.published_objects" not in query:
            return None
        assertion_id = params[0]
        assertions = [
            r
            for r in self.rows
            if r["object_type"] == "assertion"
            and r["publish_state"] == "published"
            and (r["object_id"] == assertion_id or r["payload"].get("assertion_id") == assertion_id)
        ]
        if not assertions:
            return None
        top = sorted(assertions, key=lambda r: r["created_at"], reverse=True)[0]
        if "SELECT payload" in query:
            return {"payload": top["payload"]}
        return top


def _make_client(fake_db: _FakeDB) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_database] = lambda: fake_db
    return TestClient(app)


def test_list_assertions_reads_from_published_objects_only() -> None:
    client = _make_client(_FakeDB())
    resp = client.get("/intel/assertions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert body["assertions"][0]["assertion_id"] == "asrt_1"

    resp = client.get("/intel/assertions?min_confidence=0.8")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["assertions"][0]["assertion_id"] == "asrt_1"


def test_get_assertion_detail_joins_published_claims() -> None:
    client = _make_client(_FakeDB())
    resp = client.get("/intel/assertions/asrt_1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["assertion"]["assertion_id"] == "asrt_1"
    claim_ids = {item["claim_id"] for item in body["claim_links"]}
    assert "claim_1" in claim_ids
    assert "claim_2" in claim_ids
    assert "claim_3" not in claim_ids


def test_list_claims_filters_by_assertion_using_published_payloads() -> None:
    client = _make_client(_FakeDB())
    resp = client.get("/intel/claims?assertion_id=asrt_1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    ids = {c["claim_id"] for c in body["claims"]}
    assert ids == {"claim_1", "claim_2"}
