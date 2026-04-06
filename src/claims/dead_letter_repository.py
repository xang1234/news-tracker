"""Repository for claim dead-letter records.

CRUD operations against news_intel.claim_dead_letters with
idempotent upserts keyed by record_id. Supports listing,
filtering, and replay-oriented queries.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.claims.quality import DeadLetterRecord
from src.storage.database import Database

logger = logging.getLogger(__name__)


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value)


def _parse_json_nullable(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return _parse_json(value)


def _row_to_record(row: Any) -> DeadLetterRecord:
    return DeadLetterRecord(
        record_id=row["record_id"],
        lane=row["lane"],
        run_id=row["run_id"],
        source_id=row["source_id"],
        reason=row["reason"],
        error_message=row["error_message"],
        error_detail=_parse_json(row["error_detail"]),
        source_text=row["source_text"],
        claim_snapshot=_parse_json_nullable(row["claim_snapshot"]),
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
    )


class DeadLetterRepository:
    """CRUD operations for claim dead-letter records."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_record(
        self, record: DeadLetterRecord
    ) -> DeadLetterRecord:
        """Insert or update a dead-letter record (idempotent on record_id).

        On conflict, updates error details and metadata but preserves
        the original source context.
        """
        row = await self._db.fetchrow(
            """
            INSERT INTO news_intel.claim_dead_letters (
                record_id, lane, run_id, source_id,
                reason, error_message, error_detail,
                source_text, claim_snapshot, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            )
            ON CONFLICT (record_id) DO UPDATE SET
                error_message = $6,
                error_detail = $7,
                metadata = $10
            RETURNING *
            """,
            record.record_id,
            record.lane,
            record.run_id,
            record.source_id,
            record.reason,
            record.error_message,
            json.dumps(record.error_detail),
            record.source_text,
            json.dumps(record.claim_snapshot) if record.claim_snapshot is not None else None,
            json.dumps(record.metadata),
        )
        return _row_to_record(row)

    async def get_record(
        self, record_id: str
    ) -> DeadLetterRecord | None:
        """Fetch a dead-letter record by ID."""
        row = await self._db.fetchrow(
            "SELECT * FROM news_intel.claim_dead_letters WHERE record_id = $1",
            record_id,
        )
        return _row_to_record(row) if row else None

    async def list_records(
        self,
        *,
        lane: str | None = None,
        run_id: str | None = None,
        reason: str | None = None,
        limit: int = 50,
    ) -> list[DeadLetterRecord]:
        """List dead-letter records with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if lane is not None:
            params.append(lane)
            conditions.append(f"lane = ${len(params)}")
        if run_id is not None:
            params.append(run_id)
            conditions.append(f"run_id = ${len(params)}")
        if reason is not None:
            params.append(reason)
            conditions.append(f"reason = ${len(params)}")

        params.append(limit)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT * FROM news_intel.claim_dead_letters
            {where}
            ORDER BY created_at DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_record(row) for row in rows]

    async def count_by_run(self, run_id: str) -> int:
        """Count dead-letter records for a specific run."""
        row = await self._db.fetchrow(
            "SELECT COUNT(*) AS cnt FROM news_intel.claim_dead_letters "
            "WHERE run_id = $1",
            run_id,
        )
        return row["cnt"] if row else 0

    async def count_by_reason(
        self, reason: str
    ) -> int:
        """Count dead-letter records by reason."""
        row = await self._db.fetchrow(
            "SELECT COUNT(*) AS cnt FROM news_intel.claim_dead_letters "
            "WHERE reason = $1",
            reason,
        )
        return row["cnt"] if row else 0
