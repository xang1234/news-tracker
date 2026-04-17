"""Compatibility fallback for publish-service read-model materialization."""

from __future__ import annotations

from src.publish.read_model import ReadModelRecord


class InMemoryReadModelRepository:
    """Tiny in-memory read-model store for non-database callers/tests."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ReadModelRecord] = {}

    async def upsert_records(self, records: list[ReadModelRecord]) -> int:
        for record in records:
            self._records[(record.manifest_id, record.object_id)] = record
        return len(records)

    async def count_records_for_manifest(self, manifest_id: str) -> int:
        return sum(1 for manifest_key, _ in self._records if manifest_key == manifest_id)
