"""Migration coverage for research innovation signals."""

from pathlib import Path

MIGRATION = Path("migrations/042_research_signals.sql")


def _sql() -> str:
    return MIGRATION.read_text()


def _compact_sql() -> str:
    return " ".join(_sql().split())


def test_migration_exists() -> None:
    assert MIGRATION.exists()


def test_migration_creates_research_signal_table_with_lineage_columns() -> None:
    sql = _sql()
    compact_sql = _compact_sql()

    assert "CREATE TABLE IF NOT EXISTS innovation_research_signals" in sql
    assert "source" in sql
    assert "record_id" in sql
    assert "issuer_concept_id" in sql
    assert "security_concept_id  TEXT NOT NULL REFERENCES concepts(concept_id)" in sql
    assert "theme_id" in sql
    assert "confidence_reasons JSONB NOT NULL DEFAULT '[]'" in compact_sql
    assert "source_lineage JSONB NOT NULL DEFAULT '{}'" in compact_sql
    assert "metadata JSONB NOT NULL DEFAULT '{}'" in compact_sql


def test_migration_constrains_source_and_confidence() -> None:
    sql = _sql()

    assert "innovation_research_signals_source_check" in sql
    assert "source IN ('openalex', 'arxiv')" in sql
    assert "innovation_research_signals_confidence_check" in sql
    assert "confidence >= 0.0 AND confidence <= 1.0" in sql


def test_migration_indexes_research_lookup_paths() -> None:
    sql = _sql()

    assert "idx_innovation_research_signals_published_date" in sql
    assert "idx_innovation_research_signals_theme_date" in sql
    assert "idx_innovation_research_signals_issuer_date" in sql
    assert "idx_innovation_research_signals_security_date" in sql
    assert "idx_innovation_research_signals_source_date" in sql
