"""Tests for innovation alias migration."""

from pathlib import Path

MIGRATION_PATH = Path("migrations/040_innovation_alias_metadata.sql")


def test_migration_file_exists() -> None:
    assert MIGRATION_PATH.exists()


def test_adds_confidence_and_review_metadata_columns() -> None:
    sql = MIGRATION_PATH.read_text()

    assert "ALTER TABLE concept_aliases" in sql
    assert "ADD COLUMN IF NOT EXISTS confidence" in sql
    assert "ADD COLUMN IF NOT EXISTS source_attribution" in sql
    assert "ADD COLUMN IF NOT EXISTS review_status" in sql
    assert "ADD COLUMN IF NOT EXISTS review_note" in sql
    assert "ADD COLUMN IF NOT EXISTS metadata" in sql


def test_expands_alias_type_constraint_for_innovation_aliases() -> None:
    sql = MIGRATION_PATH.read_text()

    for alias_type in (
        "subsidiary",
        "acquired_entity",
        "lab",
        "research_institution",
    ):
        assert f"'{alias_type}'" in sql


def test_indexes_review_queue_and_alias_lookup() -> None:
    sql = MIGRATION_PATH.read_text()

    assert "idx_concept_aliases_review_status" in sql
    assert "idx_concept_aliases_alias_confidence" in sql
