"""Tests for the schema migration runner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from src.storage.migrations import Migration, apply_migrations, list_migrations

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


class RecordingDatabase:
    """Small async DB stub for migration-runner tests."""

    def __init__(
        self,
        *,
        tables: set[str] | None = None,
        columns: dict[str, set[str]] | None = None,
    ) -> None:
        self.tables = set(tables or set())
        self.columns = {
            table: set(table_columns) for table, table_columns in (columns or {}).items()
        }
        self.applied_migrations: set[str] = set()
        self.executed_sql: list[str] = []

    async def execute(self, query: str, *args: object) -> str:
        self.executed_sql.append(query)
        if "CREATE TABLE IF NOT EXISTS schema_migrations" in query:
            self.tables.add("schema_migrations")
        return "OK"

    async def fetch(self, query: str, *args: object) -> list[dict[str, str]]:
        if "SELECT migration_name FROM schema_migrations" in query:
            return [{"migration_name": name} for name in sorted(self.applied_migrations)]
        return []

    async def fetchval(self, query: str, *args: object) -> bool:
        normalized = " ".join(query.split())
        if "FROM information_schema.tables" in normalized:
            if "table_name <> 'schema_migrations'" in normalized:
                return bool(self.tables - {"schema_migrations"})
            table_name = str(args[0])
            return table_name in self.tables
        if "FROM information_schema.columns" in normalized:
            table_name = str(args[0])
            column_name = str(args[1])
            return column_name in self.columns.get(table_name, set())
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


def _migration(name: str) -> Migration:
    return Migration(
        version=int(name.split("_", 1)[0]),
        name=name,
        path=_MIGRATIONS_DIR / name,
    )


def test_list_migrations_orders_initial_schema_before_embedding_upgrade() -> None:
    names = [migration.name for migration in list_migrations()]

    assert names.index("001_initial_schema.sql") < names.index("001_embedding_vector_768.sql")
    assert names.index("001_embedding_vector_768.sql") < names.index(
        "002_add_minilm_embedding.sql",
    )


@pytest.mark.asyncio
async def test_apply_migrations_skips_only_satisfied_legacy_steps(
    monkeypatch,
) -> None:
    migration_001 = _migration("001_initial_schema.sql")
    migration_001_embedding = _migration("001_embedding_vector_768.sql")
    migration_004 = _migration("004_add_themes_and_metrics.sql")
    migration_030 = _migration("030_reconcile_embedding_schema.sql")
    db = RecordingDatabase(
        tables={"documents", "processing_metrics"},
        columns={"documents": set()},
    )

    monkeypatch.setattr(
        "src.storage.migrations.list_migrations",
        lambda: [migration_001, migration_001_embedding, migration_004, migration_030],
    )

    executed = await apply_migrations(db)  # type: ignore[arg-type]

    assert executed == [
        "004_add_themes_and_metrics.sql",
        "030_reconcile_embedding_schema.sql",
    ]
    assert db.applied_migrations == {
        "001_initial_schema.sql",
        "001_embedding_vector_768.sql",
        "004_add_themes_and_metrics.sql",
        "030_reconcile_embedding_schema.sql",
    }


@pytest.mark.asyncio
async def test_apply_migrations_repairs_partial_legacy_schema_without_blanket_baseline(
    monkeypatch,
) -> None:
    migration_001 = _migration("001_initial_schema.sql")
    migration_001_embedding = _migration("001_embedding_vector_768.sql")
    migration_004 = _migration("004_add_themes_and_metrics.sql")
    migration_006 = _migration("006_add_alerts_table.sql")
    db = RecordingDatabase(tables={"documents"})

    monkeypatch.setattr(
        "src.storage.migrations.list_migrations",
        lambda: [migration_001, migration_001_embedding, migration_004, migration_006],
    )

    executed = await apply_migrations(db)  # type: ignore[arg-type]

    assert executed == [
        "001_initial_schema.sql",
        "004_add_themes_and_metrics.sql",
        "006_add_alerts_table.sql",
    ]
    assert "001_embedding_vector_768.sql" in db.applied_migrations
    assert "006_add_alerts_table.sql" in db.applied_migrations


@pytest.mark.asyncio
async def test_apply_migrations_still_skips_destructive_legacy_step_on_rerun(
    monkeypatch,
) -> None:
    migration_001_embedding = _migration("001_embedding_vector_768.sql")
    migration_004 = _migration("004_add_themes_and_metrics.sql")
    migration_030 = _migration("030_reconcile_embedding_schema.sql")
    db = RecordingDatabase(tables={"documents", "processing_metrics"})
    db.applied_migrations.add("001_initial_schema.sql")

    monkeypatch.setattr(
        "src.storage.migrations.list_migrations",
        lambda: [migration_001_embedding, migration_004, migration_030],
    )

    executed = await apply_migrations(db)  # type: ignore[arg-type]

    assert executed == [
        "004_add_themes_and_metrics.sql",
        "030_reconcile_embedding_schema.sql",
    ]
    assert "001_embedding_vector_768.sql" in db.applied_migrations


def test_runtime_reconcile_migration_guards_platform_authority_index() -> None:
    sql = (_MIGRATIONS_DIR / "029_reconcile_runtime_schema.sql").read_text(encoding="utf-8")

    assert "idx_documents_platform_authority" in sql
    assert "information_schema.columns" in sql
    assert "column_name = 'embedding'" in sql


def test_embedding_reconcile_migration_preserves_legacy_conflict_column() -> None:
    sql = (_MIGRATIONS_DIR / "030_reconcile_embedding_schema.sql").read_text(encoding="utf-8")

    assert "RENAME COLUMN embedding TO embedding_conflict_384" in sql
    assert "DROP COLUMN embedding" not in sql
