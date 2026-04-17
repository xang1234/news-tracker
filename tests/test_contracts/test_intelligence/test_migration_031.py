"""Structural checks for migration 031."""

from pathlib import Path


MIGRATION_PATH = Path("migrations/031_publish_graph_dedupe_hardening.sql")


def _sql() -> str:
    return MIGRATION_PATH.read_text(encoding="utf-8")


def test_file_exists() -> None:
    assert MIGRATION_PATH.exists()


def test_creates_causal_edge_supports_table() -> None:
    sql = _sql()
    assert "CREATE TABLE IF NOT EXISTS causal_edge_supports" in sql
    assert "support_key" in sql
    assert "origin_kind" in sql


def test_creates_document_dedup_signatures_table() -> None:
    sql = _sql()
    assert "CREATE TABLE IF NOT EXISTS document_dedup_signatures" in sql
    assert "exact_fingerprint" in sql
    assert "minhash_signature" in sql


def test_backfills_active_read_model_rows() -> None:
    sql = _sql()
    assert "INSERT INTO intel_pub.read_model" in sql
    assert "intel_pub.manifest_pointers" in sql
    assert "digest(" in sql
