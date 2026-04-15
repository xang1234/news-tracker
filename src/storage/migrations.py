"""Schema migration runner for News Tracker."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.storage.database import Database

logger = logging.getLogger(__name__)

_MIGRATION_NAME_RE = re.compile(r"^(?P<version>\d+)_.*\.sql$")
_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_SAME_VERSION_PRIORITIES = {
    "001_initial_schema.sql": 0,
    "001_embedding_vector_768.sql": 1,
}


@dataclass(frozen=True)
class Migration:
    """Versioned SQL migration."""

    version: int
    name: str
    path: Path


def list_migrations() -> list[Migration]:
    """List all versioned SQL migrations in deterministic order."""
    migrations: list[Migration] = []
    for path in _MIGRATIONS_DIR.glob("*.sql"):
        match = _MIGRATION_NAME_RE.match(path.name)
        if match is None:
            logger.warning("Skipping unversioned migration file %s", path.name)
            continue
        migrations.append(
            Migration(
                version=int(match.group("version")),
                name=path.name,
                path=path,
            )
        )
    return sorted(migrations, key=_migration_sort_key)


def _migration_sort_key(migration: Migration) -> tuple[int, int, str]:
    return (
        migration.version,
        _SAME_VERSION_PRIORITIES.get(migration.name, 100),
        migration.name,
    )


async def apply_migrations(db: Database) -> list[str]:
    """Apply pending migrations and return the filenames that were executed."""
    await _ensure_schema_migrations_table(db)

    migrations = list_migrations()
    applied = await _get_applied_migrations(db)
    pending = [migration for migration in migrations if migration.name not in applied]
    if await _has_existing_schema(db):
        skipped = await _skip_satisfied_legacy_migrations(db, pending)
        if skipped:
            applied.update(skipped)
            logger.info(
                "Marked satisfied legacy migrations as applied",
                extra={"count": len(skipped)},
            )

    executed: list[str] = []
    for migration in migrations:
        if migration.name in applied:
            continue

        sql = migration.path.read_text(encoding="utf-8")
        async with db.transaction() as conn:
            await conn.execute(sql)
            await conn.execute(
                """
                INSERT INTO schema_migrations (migration_name)
                VALUES ($1)
                ON CONFLICT (migration_name) DO NOTHING
                """,
                migration.name,
            )
        executed.append(migration.name)
        logger.info("Applied migration %s", migration.name)

    return executed


async def _ensure_schema_migrations_table(db: Database) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_name TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


async def _get_applied_migrations(db: Database) -> set[str]:
    rows = await db.fetch("SELECT migration_name FROM schema_migrations")
    return {str(row["migration_name"]) for row in rows}


async def _has_existing_schema(db: Database) -> bool:
    return bool(
        await db.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name <> 'schema_migrations'
            )
            """
        )
    )


async def _skip_satisfied_legacy_migrations(
    db: Database,
    migrations: list[Migration],
) -> set[str]:
    skipped: set[str] = set()

    for migration in migrations:
        if await _legacy_migration_is_satisfied(db, migration.name):
            await _mark_migrations_applied(db, [migration.name])
            skipped.add(migration.name)

    return skipped


async def _legacy_migration_is_satisfied(db: Database, migration_name: str) -> bool:
    if migration_name == "001_initial_schema.sql":
        return await _table_exists(db, "documents") and await _table_exists(
            db,
            "processing_metrics",
        )
    if migration_name == "001_embedding_vector_768.sql":
        return await _table_exists(db, "documents")
    if migration_name == "002_add_minilm_embedding.sql":
        return await _column_exists(db, "documents", "embedding_minilm")
    if migration_name == "003_add_authority_score.sql":
        return await _column_exists(db, "documents", "authority_score")
    if migration_name == "004_add_themes_and_metrics.sql":
        return await _table_exists(db, "themes") and await _table_exists(
            db,
            "theme_metrics",
        )
    return False


async def _table_exists(db: Database, table_name: str) -> bool:
    return bool(
        await db.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = $1
            )
            """,
            table_name,
        )
    )


async def _column_exists(db: Database, table_name: str, column_name: str) -> bool:
    return bool(
        await db.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = $1
                  AND column_name = $2
            )
            """,
            table_name,
            column_name,
        )
    )


async def _mark_migrations_applied(db: Database, migration_names: list[str]) -> None:
    async with db.transaction() as conn:
        for migration_name in migration_names:
            await conn.execute(
                """
                INSERT INTO schema_migrations (migration_name)
                VALUES ($1)
                ON CONFLICT (migration_name) DO NOTHING
                """,
                migration_name,
            )
