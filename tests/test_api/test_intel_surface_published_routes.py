"""Tests for intel_surface assertion/claim routes using published objects only."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

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
                "object_id": "obj_assert_1_new",
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
                "object_id": "obj_assert_1_old",
                "object_type": "assertion",
                "publish_state": "published",
                "contract_version": "1.0.0",
                "lane": "narrative",
                "run_id": "run_0",
                "valid_from": None,
                "valid_to": None,
                "created_at": _ts(11),
                "updated_at": _ts(11),
                "payload": {
                    "assertion_id": "asrt_1",
                    "subject_concept_id": "concept_issuer_nvidia",
                    "predicate": "supplies_to",
                    "object_concept_id": "concept_issuer_msft",
                    "confidence": 0.51,
                    "status": "active",
                    "support_count": 2,
                    "contradiction_count": 0,
                    "source_diversity": 1,
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
                "created_at": _ts(10),
                "updated_at": _ts(10),
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
                "object_id": "obj_claim_1_new",
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
                    "claim_id": "claim_1",
                    "claim_key": "key_1",
                    "source_id": "doc_1_new",
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
                "object_id": "obj_claim_1_old",
                "object_type": "claim",
                "publish_state": "published",
                "contract_version": "1.0.0",
                "lane": "narrative",
                "run_id": "run_0",
                "valid_from": None,
                "valid_to": None,
                "created_at": _ts(8),
                "updated_at": _ts(8),
                "payload": {
                    "claim_id": "claim_1",
                    "claim_key": "key_1_old",
                    "source_id": "doc_1_old",
                    "source_type": "news",
                    "subject_text": "NVIDIA",
                    "predicate": "supplies_to",
                    "object_text": "Microsoft",
                    "confidence": 0.3,
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
                "created_at": _ts(7),
                "updated_at": _ts(7),
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
                "created_at": _ts(6),
                "updated_at": _ts(6),
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

    def _published_rows(self, object_type: str | None = None) -> list[dict[str, Any]]:
        rows = [r for r in self.rows if r["publish_state"] == "published"]
        if object_type is not None:
            rows = [r for r in rows if r["object_type"] == object_type]
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)

    @staticmethod
    def _assertion_key(row: dict[str, Any]) -> str:
        payload = row["payload"]
        assertion_id = payload.get("assertion_id")
        return assertion_id if assertion_id else row["object_id"]

    @staticmethod
    def _claim_key(row: dict[str, Any]) -> str:
        payload = row["payload"]
        claim_id = payload.get("claim_id")
        return claim_id if claim_id else row["object_id"]

    @staticmethod
    def _dedupe_latest(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda r: r["created_at"], reverse=True):
            key = key_fn(row)
            if key not in deduped:
                deduped[key] = row
        return sorted(deduped.values(), key=lambda r: r["created_at"], reverse=True)

    @staticmethod
    def _claim_matches_assertion(
        payload: dict[str, Any],
        object_id: str,
        assertion_id: str,
        embedded_claim_ids: set[str],
    ) -> bool:
        if payload.get("assertion_id") == assertion_id:
            return True
        if payload.get("linked_assertion_id") == assertion_id:
            return True
        assertion_ids = payload.get("assertion_ids")
        if isinstance(assertion_ids, list) and assertion_id in assertion_ids:
            return True
        links = payload.get("links")
        if isinstance(links, list):
            for link in links:
                if isinstance(link, dict) and link.get("assertion_id") == assertion_id:
                    return True
        claim_id = payload.get("claim_id") or object_id
        return claim_id in embedded_claim_ids

    def _filtered_assertions(self, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        assert params and params[0] == "assertion"
        idx = 1
        rows = self._published_rows("assertion")

        if "subject_concept_id" in query:
            concept_id = params[idx]
            rows = [
                r
                for r in rows
                if r["payload"].get("subject_concept_id") == concept_id
                or r["payload"].get("object_concept_id") == concept_id
            ]
            idx += 1
        if "payload->>'predicate'" in query:
            pred = params[idx]
            rows = [r for r in rows if r["payload"].get("predicate") == pred]
            idx += 1
        if "payload->>'status'" in query:
            status = params[idx]
            rows = [r for r in rows if r["payload"].get("status") == status]
            idx += 1
        if "jsonb_typeof(payload->'confidence')" in query:
            min_conf = float(params[idx])
            rows = [r for r in rows if float(r["payload"].get("confidence", 0.0)) >= min_conf]

        return self._dedupe_latest(rows, self._assertion_key)

    def _filtered_claims(self, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        assert params and params[0] == "claim"
        idx = 1
        rows = self._published_rows("claim")

        if "lane = $" in query:
            lane = params[idx]
            rows = [r for r in rows if r["lane"] == lane]
            idx += 1
        if "payload->>'source_id'" in query:
            source_id = params[idx]
            rows = [r for r in rows if r["payload"].get("source_id") == source_id]
            idx += 1
        if "payload->>'status'" in query:
            status = params[idx]
            rows = [r for r in rows if r["payload"].get("status") == status]
            idx += 1

        if "payload->>'linked_assertion_id'" in query:
            assertion_id = str(params[idx])
            embedded_claim_ids = set(params[idx + 1])
            rows = [
                r
                for r in rows
                if self._claim_matches_assertion(
                    r["payload"], r["object_id"], assertion_id, embedded_claim_ids
                )
            ]

        return self._dedupe_latest(rows, self._claim_key)

    async def fetch(self, query: str, *params):
        if "intel_pub.published_objects" not in query:
            return []

        if "claim_dedupe_id" in query and "object_type = 'claim'" in query:
            assertion_id = str(params[0])
            embedded_claim_ids = set(params[1])
            rows = self._published_rows("claim")
            rows = [
                r
                for r in rows
                if self._claim_matches_assertion(
                    r["payload"], r["object_id"], assertion_id, embedded_claim_ids
                )
            ]
            return self._dedupe_latest(rows, self._claim_key)

        if "assertion_dedupe_id" in query:
            rows = self._filtered_assertions(query, params)
            if "LIMIT $" in query:
                limit, offset = int(params[-2]), int(params[-1])
                rows = rows[offset : offset + limit]
            return rows

        if "claim_dedupe_id" in query:
            rows = self._filtered_claims(query, params)
            if "LIMIT $" in query:
                limit, offset = int(params[-2]), int(params[-1])
                rows = rows[offset : offset + limit]
            return rows

        return []

    async def fetchrow(self, query: str, *params):
        if "intel_pub.published_objects" not in query:
            return None

        if "SELECT payload" in query and "object_type = 'assertion'" in query:
            assertion_id = params[0]
            assertions = [
                r
                for r in self._published_rows("assertion")
                if r["object_id"] == assertion_id or r["payload"].get("assertion_id") == assertion_id
            ]
            if not assertions:
                return None
            top = sorted(assertions, key=lambda r: r["created_at"], reverse=True)[0]
            return {"payload": top["payload"]}

        if "SELECT *" in query and "object_type = 'assertion'" in query:
            assertion_id = params[0]
            assertions = [
                r
                for r in self._published_rows("assertion")
                if r["object_id"] == assertion_id or r["payload"].get("assertion_id") == assertion_id
            ]
            if not assertions:
                return None
            return sorted(assertions, key=lambda r: r["created_at"], reverse=True)[0]

        if "SELECT count(*) AS cnt" in query and "assertion_dedupe_id" in query:
            return {"cnt": len(self._filtered_assertions(query, params))}

        if "SELECT count(*) AS cnt" in query and "claim_dedupe_id" in query:
            return {"cnt": len(self._filtered_claims(query, params))}

        return None


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
    assert [a["assertion_id"] for a in body["assertions"]] == ["asrt_1", "asrt_2"]

    resp = client.get("/intel/assertions?min_confidence=0.8")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["assertions"][0]["assertion_id"] == "asrt_1"


def test_list_assertions_dedupes_and_paginates() -> None:
    client = _make_client(_FakeDB())
    resp = client.get("/intel/assertions?limit=1&offset=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["assertions"]) == 1
    assert body["assertions"][0]["assertion_id"] == "asrt_2"


def test_get_assertion_detail_joins_published_claims_uses_latest_claim_version() -> None:
    client = _make_client(_FakeDB())
    resp = client.get("/intel/assertions/asrt_1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["assertion"]["assertion_id"] == "asrt_1"
    claim_ids = [item["claim_id"] for item in body["claim_links"]]
    assert claim_ids.count("claim_1") == 1
    assert "claim_2" in claim_ids
    assert "claim_3" not in claim_ids

    claim_1 = next(item for item in body["claim_links"] if item["claim_id"] == "claim_1")
    assert claim_1["claim"]["source_id"] == "doc_1_new"


def test_list_claims_filters_by_assertion_using_published_payloads() -> None:
    client = _make_client(_FakeDB())
    resp = client.get("/intel/claims?assertion_id=asrt_1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    ids = [c["claim_id"] for c in body["claims"]]
    assert sorted(ids) == ["claim_1", "claim_2"]


def test_list_claims_dedupes_and_paginates() -> None:
    client = _make_client(_FakeDB())
    resp = client.get("/intel/claims?limit=1&offset=0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["claims"]) == 1
    assert body["claims"][0]["claim_id"] == "claim_1"
