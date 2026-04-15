"""Tests for the schema migration runner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from src.storage.migrations import Migration, apply_migrations


class RecordingDatabase:
    """Small async DB stub for migration-runner tests."""

    def __init__(self, *, legacy_schema: bool) -> None:
        self.legacy_schema = legacy_schema
        self.applied_migrations: set[str] = set()
        self.executed_sql: list[str] = []

    async def execute(self, query: str, *args: object) -> str:
        self.executed_sql.append(query)
        return "OK"

    async def fetch(self, query: str, *args: object) -> list[dict[str, str]]:
        if "SELECT migration_name FROM schema_migrations" in query:
            return [{"migration_name": name} for name in sorted(self.applied_migrations)]
        return []

    async def fetchval(self, query: str, *args: object) -> bool:
        if "table_name = 'documents'" in query:
            return self.legacy_schema
        return False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[RecordingConnection]:
        yield RecordingConnection(self)


class RecordingConnection:
    def __init__(self, database: RecordingDatabase) -> None:
        self.database = database

    async def execute(self, query: str, *args: object) -> str:
        self.database.executed_sql.append(query)
        if "INSERT INTO schema_migrations" in query:
            self.database.applied_migrations.add(str(args[0]))
        return "OK"


def _write_migration(path: Path, sql: str) -> Migration:
    path.write_text(sql, encoding="utf-8")
    return Migration(version=int(path.name.split("_", 1)[0]), name=path.name, path=path)


@pytest.mark.asyncio
async def test_apply_migrations_baselines_legacy_schema(monkeypatch, tmp_path: Path) -> None:
    migration_001 = _write_migration(tmp_path / "001_initial.sql", "SELECT 1;")
    migration_028 = _write_migration(tmp_path / "028_existing.sql", "SELECT 28;")
    migration_029 = _write_migration(tmp_path / "029_reconcile.sql", "SELECT 29;")
    db = RecordingDatabase(legacy_schema=True)

    monkeypatch.setattr(
        "src.storage.migrations.list_migrations",
        lambda: [migration_001, migration_028, migration_029],
    )

    executed = await apply_migrations(db)  # type: ignore[arg-type]

    assert executed == ["029_reconcile.sql"]
    assert "001_initial.sql" in db.applied_migrations
    assert "028_existing.sql" in db.applied_migrations
    assert "029_reconcile.sql" in db.applied_migrations


@pytest.mark.asyncio
async def test_apply_migrations_runs_all_on_fresh_database(monkeypatch, tmp_path: Path) -> None:
    migration_001 = _write_migration(tmp_path / "001_initial.sql", "SELECT 1;")
    migration_002 = _write_migration(tmp_path / "002_followup.sql", "SELECT 2;")
    db = RecordingDatabase(legacy_schema=False)

    monkeypatch.setattr(
        "src.storage.migrations.list_migrations",
        lambda: [migration_001, migration_002],
    )

    executed = await apply_migrations(db)  # type: ignore[arg-type]

    assert executed == ["001_initial.sql", "002_followup.sql"]
    assert db.applied_migrations == {"001_initial.sql", "002_followup.sql"}
