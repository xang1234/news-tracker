"""Tests for SEC identifier security-master migration."""

from pathlib import Path

import pytest

MIGRATION_PATH = Path(__file__).resolve().parents[2] / "migrations" / "034_sec_identifiers.sql"


@pytest.fixture()
def migration_sql() -> str:
    return MIGRATION_PATH.read_text()


class TestMigration034:
    """Migration 034 structural checks."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_adds_sec_identifier_columns(self, migration_sql: str) -> None:
        assert "ALTER TABLE securities ADD COLUMN IF NOT EXISTS sec_cik" in migration_sql
        assert "ALTER TABLE securities ADD COLUMN IF NOT EXISTS issuer_name" in migration_sql
        assert "ALTER TABLE securities ADD COLUMN IF NOT EXISTS former_names" in migration_sql
        assert (
            "ALTER TABLE securities ADD COLUMN IF NOT EXISTS external_identifiers" in migration_sql
        )
        assert "ALTER TABLE securities ADD COLUMN IF NOT EXISTS identifier_lineage" in migration_sql

    def test_indexes_sec_resolution_paths(self, migration_sql: str) -> None:
        assert "idx_securities_sec_cik" in migration_sql
        assert "idx_securities_issuer_name_trgm" in migration_sql
        assert "idx_securities_former_names_gin" in migration_sql

    def test_sec_cik_not_unique_because_multiple_securities_can_share_issuer(
        self,
        migration_sql: str,
    ) -> None:
        assert "UNIQUE INDEX IF NOT EXISTS idx_securities_sec_cik" not in migration_sql
