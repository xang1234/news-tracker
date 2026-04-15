"""Schema migration runner for News Tracker."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.storage.database import Database

logger = logging.getLogger(__name__)

_MIGRATION_NAME_RE = re.compile(r"^(?P<version>\d+)_.*\.sql$")
_LEGACY_BASELINE_VERSION = 28
_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


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
    return sorted(migrations, key=lambda migration: (migration.version, migration.name))


async def apply_migrations(db: Database) -> list[str]:
    """Apply pending migrations and return the filenames that were executed."""
    await _ensure_schema_migrations_table(db)

    migrations = list_migrations()
    applied = await _get_applied_migrations(db)
    if not applied and await _has_legacy_schema(db):
        legacy_names = [
            migration.name
            for migration in migrations
            if migration.version <= _LEGACY_BASELINE_VERSION
        ]
        if legacy_names:
            await _mark_migrations_applied(db, legacy_names)
            applied.update(legacy_names)
            logger.info(
                "Baselined legacy schema without replaying historical migrations",
                extra={"count": len(legacy_names)},
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


async def _has_legacy_schema(db: Database) -> bool:
    return bool(
        await db.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'documents'
            )
            """
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
