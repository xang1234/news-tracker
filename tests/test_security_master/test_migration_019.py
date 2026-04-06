"""Tests for migration 019 SQL structure."""

import pathlib

import pytest

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "migrations" / "019_concept_registry.sql"
)


@pytest.fixture()
def migration_sql() -> str:
    return MIGRATION_PATH.read_text()


class TestMigration019:
    """Migration 019 structural checks."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_concepts_table(self, migration_sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS concepts" in migration_sql

    def test_creates_concept_aliases_table(self, migration_sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS concept_aliases" in migration_sql

    def test_creates_issuer_security_map(self, migration_sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS issuer_security_map" in migration_sql

    def test_adds_concept_id_to_securities(self, migration_sql: str) -> None:
        assert "ALTER TABLE securities ADD COLUMN IF NOT EXISTS concept_id" in migration_sql

    def test_concept_type_check_constraint(self, migration_sql: str) -> None:
        assert "'issuer'" in migration_sql
        assert "'security'" in migration_sql
        assert "'technology'" in migration_sql
        assert "'theme'" in migration_sql
        assert "'narrative_frame'" in migration_sql

    def test_uses_if_not_exists(self, migration_sql: str) -> None:
        for line in migration_sql.splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("CREATE TABLE ") and "IF NOT EXISTS" not in stripped:
                pytest.fail(f"CREATE TABLE without IF NOT EXISTS: {line.strip()}")

    def test_trigram_index_on_concepts(self, migration_sql: str) -> None:
        assert "gin_trgm_ops" in migration_sql

    def test_updated_at_trigger(self, migration_sql: str) -> None:
        assert "update_concepts_updated_at" in migration_sql

    def test_foreign_keys(self, migration_sql: str) -> None:
        assert "REFERENCES concepts(concept_id)" in migration_sql
