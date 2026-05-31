"""Tests for SEC structured payload cache persistence."""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.filing.sec_structured import (
    SECStructuredDataRepository,
    SECStructuredPayloadRecord,
)

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "migrations" / "035_sec_structured_cache.sql"
)


def _payload_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": 7,
        "cik": "0000320193",
        "payload_type": "submissions",
        "source_url": "https://data.sec.gov/submissions/CIK0000320193.json",
        "payload_hash": "sha256:abc",
        "payload": {"cik": "0000320193"},
        "accession_numbers": ["0000320193-24-000123"],
        "fetched_at": datetime(2026, 5, 31, tzinfo=UTC),
        "first_seen_at": datetime(2026, 5, 31, tzinfo=UTC),
        "last_seen_at": datetime(2026, 5, 31, tzinfo=UTC),
        "metadata": {"source": "sec"},
        "created_at": datetime(2026, 5, 31, tzinfo=UTC),
        "updated_at": datetime(2026, 5, 31, tzinfo=UTC),
    }
    row.update(overrides)
    return row


def _record() -> SECStructuredPayloadRecord:
    return SECStructuredPayloadRecord(
        cik="320193",
        payload_type="submissions",
        source_url="https://data.sec.gov/submissions/CIK0000320193.json",
        payload_hash="sha256:abc",
        payload={"cik": "0000320193"},
        accession_numbers=["0000320193-24-000123"],
        fetched_at=datetime(2026, 5, 31, tzinfo=UTC),
        metadata={"source": "sec"},
    )


class TestSECStructuredPayloadRecord:
    def test_normalizes_cik(self) -> None:
        record = _record()

        assert record.cik == "0000320193"

    def test_rejects_unknown_payload_type(self) -> None:
        with pytest.raises(ValueError, match="payload_type"):
            SECStructuredPayloadRecord(
                cik="0000320193",
                payload_type="bad",
                source_url="https://data.sec.gov/example.json",
                payload_hash="sha256:abc",
                payload={},
            )


class TestSECStructuredDataRepository:
    @pytest.mark.asyncio
    async def test_upsert_payload_is_idempotent_by_cik_type_and_hash(self) -> None:
        database = AsyncMock()
        database.fetchrow.return_value = _payload_row()
        repository = SECStructuredDataRepository(database)

        persisted = await repository.upsert_payload(_record())

        args = database.fetchrow.call_args[0]
        sql = args[0]
        assert "ON CONFLICT (cik, payload_type, payload_hash)" in sql
        assert "last_seen_at" in sql
        assert args[1] == "0000320193"
        assert args[2] == "submissions"
        assert args[6] == ["0000320193-24-000123"]
        assert persisted.cik == "0000320193"
        assert persisted.id == 7

    @pytest.mark.asyncio
    async def test_get_latest_payload_orders_by_last_seen(self) -> None:
        database = AsyncMock()
        database.fetchrow.return_value = _payload_row()
        repository = SECStructuredDataRepository(database)

        record = await repository.get_latest_payload("320193", "submissions")

        args = database.fetchrow.call_args[0]
        sql = args[0]
        assert "WHERE cik = $1 AND payload_type = $2" in sql
        assert "ORDER BY last_seen_at DESC, fetched_at DESC" in sql
        assert args[1] == "0000320193"
        assert record is not None
        assert record.accession_numbers == ["0000320193-24-000123"]


class TestMigration035:
    @pytest.fixture()
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_sec_structured_payloads_table(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS sec_structured_payloads" in sql

    def test_unique_payload_identity(self, sql: str) -> None:
        assert "UNIQUE (cik, payload_type, payload_hash)" in sql

    def test_accession_lineage_is_indexed(self, sql: str) -> None:
        assert "accession_numbers" in sql
        assert "USING GIN (accession_numbers)" in sql

    def test_payload_is_jsonb(self, sql: str) -> None:
        assert "payload             JSONB NOT NULL" in sql

    def test_updated_at_trigger(self, sql: str) -> None:
        assert "update_sec_structured_payloads_updated_at" in sql
