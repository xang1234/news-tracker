"""Structural checks for migration 032."""

from pathlib import Path

MIGRATION_PATH = Path("migrations/032_read_model_timestamp_repair.sql")


def _sql() -> str:
    return MIGRATION_PATH.read_text(encoding="utf-8")


def test_file_exists() -> None:
    assert MIGRATION_PATH.exists()


def test_adds_updated_at_column() -> None:
    sql = _sql()
    assert "ADD COLUMN IF NOT EXISTS updated_at" in sql


def test_repairs_read_model_timestamps_from_published_objects() -> None:
    sql = _sql()
    assert "UPDATE intel_pub.read_model rm" in sql
    assert "FROM intel_pub.published_objects po" in sql
    assert "created_at = po.created_at" in sql
    assert "updated_at = po.updated_at" in sql
