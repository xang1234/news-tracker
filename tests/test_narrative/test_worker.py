"""Focused tests for narrative worker edge cases."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.narrative.config import NarrativeConfig
from src.narrative.worker import NarrativeWorker


class _FakeDatabase:
    def __init__(self, conn: AsyncMock) -> None:
        self._conn = conn
        self.execute = AsyncMock()
        self.fetch = AsyncMock()

    @asynccontextmanager
    async def transaction(self):
        yield self._conn


@pytest.mark.asyncio
async def test_process_document_for_theme_skips_duplicate_assignment():
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetchval = AsyncMock(return_value="run_existing")
    conn.fetch = AsyncMock()
    db = _FakeDatabase(conn)
    worker = NarrativeWorker(database=db, config=NarrativeConfig())
    worker._narrative_repo = AsyncMock()

    document = SimpleNamespace(id="doc_1", embedding=[0.1, 0.2, 0.3])

    result = await worker.process_document_for_theme(
        document=document,
        theme_id="theme_1",
        theme_similarity=0.95,
    )

    assert result is None
    conn.fetch.assert_not_awaited()
    worker._narrative_repo.get_recent_buckets.assert_not_called()


@pytest.mark.asyncio
async def test_run_maintenance_does_not_attempt_merge_by_default():
    db = AsyncMock()
    worker = NarrativeWorker(database=db, config=NarrativeConfig())

    await worker._run_maintenance()

    assert db.execute.await_count == 2
    db.fetch.assert_not_called()
