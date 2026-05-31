"""Persistence for SEC structured payload caches."""

from __future__ import annotations

import json
from typing import Any

from src.filing.sec_structured_models import SECStructuredPayloadRecord
from src.security_master.schemas import normalize_sec_cik
from src.storage.database import Database


def _parse_json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return dict(value)


def _row_to_payload(row: Any) -> SECStructuredPayloadRecord:
    return SECStructuredPayloadRecord(
        id=row["id"],
        cik=row["cik"],
        payload_type=row["payload_type"],
        source_url=row["source_url"],
        payload_hash=row["payload_hash"],
        payload=_parse_json_object(row["payload"]),
        accession_numbers=list(row["accession_numbers"] or []),
        fetched_at=row["fetched_at"],
        first_seen_at=row["first_seen_at"],
        last_seen_at=row["last_seen_at"],
        metadata=_parse_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class SECStructuredDataRepository:
    """Database cache for official SEC submissions and Company Facts payloads."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_payload(
        self,
        record: SECStructuredPayloadRecord,
    ) -> SECStructuredPayloadRecord:
        row = await self._db.fetchrow(
            """
            INSERT INTO sec_structured_payloads (
                cik, payload_type, source_url, payload_hash, payload,
                accession_numbers, fetched_at, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (cik, payload_type, payload_hash) DO UPDATE SET
                source_url = $3,
                payload = $5,
                accession_numbers = $6,
                last_seen_at = GREATEST(sec_structured_payloads.last_seen_at, $7),
                metadata = $8,
                updated_at = NOW()
            RETURNING *
            """,
            record.cik,
            record.payload_type,
            record.source_url,
            record.payload_hash,
            json.dumps(record.payload),
            record.accession_numbers,
            record.fetched_at,
            json.dumps(record.metadata),
        )
        return _row_to_payload(row)

    async def get_latest_payload(
        self,
        cik: str,
        payload_type: str,
    ) -> SECStructuredPayloadRecord | None:
        normalized_cik = normalize_sec_cik(cik)
        if normalized_cik is None:
            raise ValueError("cik must be a non-empty SEC CIK")
        row = await self._db.fetchrow(
            """
            SELECT *
            FROM sec_structured_payloads
            WHERE cik = $1 AND payload_type = $2
            ORDER BY last_seen_at DESC, fetched_at DESC
            LIMIT 1
            """,
            normalized_cik,
            payload_type,
        )
        return _row_to_payload(row) if row else None
