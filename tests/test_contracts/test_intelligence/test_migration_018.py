"""Tests for migration 018 SQL structure.

Validates that the migration file exists, is well-formed, and creates
the expected schema namespaces and tables.
"""

import pathlib

import pytest

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "migrations"
    / "018_intelligence_schemas.sql"
)


@pytest.fixture()
def migration_sql() -> str:
    """Read the migration SQL."""
    return MIGRATION_PATH.read_text()


class TestMigration018:
    """Migration 018 structural checks."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_schemas(self, migration_sql: str) -> None:
        assert "CREATE SCHEMA IF NOT EXISTS news_intel" in migration_sql
        assert "CREATE SCHEMA IF NOT EXISTS intel_pub" in migration_sql
        assert "CREATE SCHEMA IF NOT EXISTS intel_export" in migration_sql

    def test_creates_lane_runs(self, migration_sql: str) -> None:
        assert "news_intel.lane_runs" in migration_sql

    def test_creates_manifests(self, migration_sql: str) -> None:
        assert "intel_pub.manifests" in migration_sql

    def test_creates_manifest_pointers(self, migration_sql: str) -> None:
        assert "intel_pub.manifest_pointers" in migration_sql

    def test_creates_published_objects(self, migration_sql: str) -> None:
        assert "intel_pub.published_objects" in migration_sql

    def test_creates_export_bundles(self, migration_sql: str) -> None:
        assert "intel_export.export_bundles" in migration_sql

    def test_uses_if_not_exists(self, migration_sql: str) -> None:
        """All CREATE statements use IF NOT EXISTS for idempotency."""
        for line in migration_sql.splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("CREATE TABLE ") and "IF NOT EXISTS" not in stripped:
                pytest.fail(f"CREATE TABLE without IF NOT EXISTS: {line.strip()}")
            if stripped.startswith("CREATE SCHEMA ") and "IF NOT EXISTS" not in stripped:
                pytest.fail(f"CREATE SCHEMA without IF NOT EXISTS: {line.strip()}")

    def test_lane_check_constraints(self, migration_sql: str) -> None:
        """Lane columns have CHECK constraints matching canonical lanes."""
        assert "'narrative'" in migration_sql
        assert "'filing'" in migration_sql
        assert "'structural'" in migration_sql
        assert "'backtest'" in migration_sql

    def test_updated_at_triggers(self, migration_sql: str) -> None:
        """Tables with updated_at columns have auto-update triggers."""
        assert "update_lane_runs_updated_at" in migration_sql
        assert "update_published_objects_updated_at" in migration_sql

    def test_no_existing_table_modifications(self, migration_sql: str) -> None:
        """Migration does not ALTER existing public tables."""
        for line in migration_sql.splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("ALTER TABLE") and "PUBLIC." in stripped:
                pytest.fail(f"Migration modifies existing table: {line.strip()}")
