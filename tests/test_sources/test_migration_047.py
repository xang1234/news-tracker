"""Structural checks for pruned source reconciliation migration."""

from pathlib import Path

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "migrations" / ("047_deactivate_pruned_sources.sql")
)

PRUNED_SOURCES = {
    ("rss", "ars-technica"),
    ("rss", "techcrunch"),
    ("rss", "the-verge"),
    ("rss", "toms-hardware"),
    ("twitter", "AMD"),
    ("twitter", "Broadcom"),
    ("twitter", "DeItaone"),
    ("twitter", "MicronTech"),
    ("twitter", "Qualcomm"),
    ("twitter", "SKhynix"),
    ("twitter", "Samsung_SD"),
    ("twitter", "StockMKTNewz"),
    ("twitter", "TechAltar"),
    ("twitter", "intel"),
    ("twitter", "nvidia"),
    ("twitter", "unusual_whales"),
    ("reddit", "AMD_Stock"),
    ("reddit", "intel"),
    ("reddit", "investing"),
    ("reddit", "nvidia"),
    ("reddit", "options"),
    ("reddit", "stockmarket"),
    ("reddit", "stocks"),
    ("reddit", "wallstreetbets"),
}


def _sql() -> str:
    return MIGRATION_PATH.read_text(encoding="utf-8")


def test_migration_file_exists() -> None:
    assert MIGRATION_PATH.exists()


def test_deactivates_all_sources_pruned_from_seed_catalogs() -> None:
    sql = _sql()

    assert "UPDATE sources" in sql
    assert "SET is_active = FALSE" in sql
    assert "updated_at = NOW()" in sql
    for platform, identifier in PRUNED_SOURCES:
        assert f"('{platform}', '{identifier}')" in sql


def test_does_not_delete_source_history() -> None:
    sql = _sql().upper()

    assert "DELETE FROM SOURCES" not in sql
    assert "DROP TABLE" not in sql
