"""Tests for shared, persisted duplicate suppression."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from src.ingestion.deduplication import SharedDeduplicator


class FakeDatabase:
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield None


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    async def exists(self, key: str) -> bool:
        return key in self.values

    async def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool:
        del ex  # TTL is not needed for unit tests.
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        deleted = 0
        if key in self.values:
            del self.values[key]
            deleted += 1
        if key in self.sets:
            del self.sets[key]
            deleted += 1
        return deleted

    async def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    async def sadd(self, key: str, value: str) -> int:
        members = self.sets.setdefault(key, set())
        before = len(members)
        members.add(value)
        return 1 if len(members) != before else 0

    async def close(self) -> None:
        return None


class FakeRepository:
    def __init__(self) -> None:
        self.documents: dict[str, object] = {}
        self.signatures: dict[str, dict[str, object]] = {}
        self.exact_index: dict[str, str] = {}

    async def insert(self, doc, *, conn=None) -> bool:
        del conn
        inserted = doc.id not in self.documents
        self.documents[doc.id] = doc
        return inserted

    async def get_dedup_signature_by_exact_fingerprint(self, exact_fingerprint: str, *, conn=None):
        del conn
        document_id = self.exact_index.get(exact_fingerprint)
        if document_id is None:
            return None
        return self.signatures[document_id]

    async def fetch_dedup_signatures(self, document_ids: list[str], *, conn=None):
        del conn
        return [self.signatures[document_id] for document_id in document_ids if document_id in self.signatures]

    async def list_dedup_signatures(self, *, conn=None):
        del conn
        return list(self.signatures.values())

    async def upsert_dedup_signature(
        self,
        *,
        document_id: str,
        canonical_document_id: str,
        exact_fingerprint: str,
        minhash_signature: list[str],
        conn=None,
    ) -> None:
        del conn
        row = {
            "document_id": document_id,
            "canonical_document_id": canonical_document_id,
            "exact_fingerprint": exact_fingerprint,
            "minhash_signature": minhash_signature,
        }
        self.signatures[document_id] = row
        self.exact_index[exact_fingerprint] = document_id


@pytest.mark.asyncio
async def test_shared_deduplicator_suppresses_exact_duplicates_across_workers(
    duplicate_documents,
) -> None:
    database = FakeDatabase()
    redis_client = FakeRedis()
    repository = FakeRepository()

    worker_a = SharedDeduplicator(database, redis_client=redis_client, threshold=0.5)
    worker_b = SharedDeduplicator(database, redis_client=redis_client, threshold=0.5)

    original = duplicate_documents[0]
    duplicate = original.model_copy(update={"id": "news_1_copy"})

    first_result = await worker_a.store_if_unique(original, repository=repository)
    second_result = await worker_b.store_if_unique(duplicate, repository=repository)

    assert first_result.is_duplicate is False
    assert second_result.is_duplicate is True
    assert second_result.duplicate_type == "exact"
    assert second_result.canonical_document_id == original.id
    assert len(repository.documents) == 1


@pytest.mark.asyncio
async def test_shared_deduplicator_warms_redis_from_persisted_signatures_for_near_duplicates(
    duplicate_documents,
) -> None:
    database = FakeDatabase()
    repository = FakeRepository()

    first_redis = FakeRedis()
    first_worker = SharedDeduplicator(database, redis_client=first_redis, threshold=0.5)
    original = duplicate_documents[0]
    near_duplicate = duplicate_documents[1]

    first_result = await first_worker.store_if_unique(original, repository=repository)
    assert first_result.is_duplicate is False

    restarted_worker = SharedDeduplicator(database, redis_client=FakeRedis(), threshold=0.5)
    second_result = await restarted_worker.store_if_unique(near_duplicate, repository=repository)

    assert second_result.is_duplicate is True
    assert second_result.duplicate_type == "near"
    assert second_result.canonical_document_id == original.id
