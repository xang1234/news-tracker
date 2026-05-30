"""Structural checks for factor registry migration."""

from pathlib import Path

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "migrations" / "033_factor_series.sql"
)


def test_migration_file_exists() -> None:
    assert MIGRATION_PATH.exists()


def test_creates_factor_series_table() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS factor_series" in sql
    assert "provider" in sql
    assert "external_id" in sql
    assert "release_lag_days" in sql
    assert "relevance_tags" in sql
    assert "required_credentials" in sql


def test_creates_factor_observations_table() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS factor_observations" in sql
    assert "observation_date" in sql
    assert "available_at" in sql
    assert "fetched_at" in sql
    assert "revision" in sql
    assert "missing_reason" in sql


def test_observations_have_point_in_time_index() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8")

    assert "idx_factor_observations_as_of" in sql
    assert "available_at DESC" in sql
    assert "fetched_at DESC" in sql


def test_observation_units_must_match_registered_series_units() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8")

    assert "UNIQUE (factor_id, units)" in sql
    assert "FOREIGN KEY (factor_id, units)" in sql
    assert "REFERENCES factor_series(factor_id, units)" in sql


def test_cadence_check_constraint_lists_supported_cadences() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8")

    for cadence in ("daily", "weekly", "monthly", "quarterly", "annual", "irregular"):
        assert f"'{cadence}'" in sql
