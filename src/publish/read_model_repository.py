"""Persistence helpers for the published intelligence read model."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.publish.read_model import ReadModelRecord
from src.storage.database import Database

logger = logging.getLogger(__name__)


class ReadModelRepository:
    """CRUD operations for the consumer-facing published read model."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_records(
        self,
        records: list[ReadModelRecord],
        *,
        conn: Any | None = None,
    ) -> int:
        """Persist read-model rows idempotently."""
        if not records:
            return 0

        query = """
        INSERT INTO intel_pub.read_model (
            record_id, manifest_id, object_id, object_type, lane,
            contract_version, publish_state, source_ids, run_id,
            valid_from, valid_to, payload, lineage, published_at, metadata
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9,
            $10, $11, $12, $13, $14, $15
        )
        ON CONFLICT (manifest_id, object_id) DO UPDATE
        SET record_id = EXCLUDED.record_id,
            object_type = EXCLUDED.object_type,
            lane = EXCLUDED.lane,
            contract_version = EXCLUDED.contract_version,
            publish_state = EXCLUDED.publish_state,
            source_ids = EXCLUDED.source_ids,
            run_id = EXCLUDED.run_id,
            valid_from = EXCLUDED.valid_from,
            valid_to = EXCLUDED.valid_to,
            payload = EXCLUDED.payload,
            lineage = EXCLUDED.lineage,
            published_at = EXCLUDED.published_at,
            metadata = EXCLUDED.metadata
        """

        params = [
            (
                record.record_id,
                record.manifest_id,
                record.object_id,
                record.object_type,
                record.lane,
                record.contract_version,
                record.publish_state,
                record.source_ids,
                record.run_id,
                record.valid_from,
                record.valid_to,
                json.dumps(record.payload),
                json.dumps(record.lineage),
                record.published_at,
                json.dumps(record.metadata),
            )
            for record in records
        ]

        if conn is not None:
            await conn.executemany(query, params)
            return len(records)

        async with self._db.acquire() as acquired:
            await acquired.executemany(query, params)
        return len(records)

    async def count_records_for_manifest(
        self,
        manifest_id: str,
        *,
        conn: Any | None = None,
    ) -> int:
        """Count materialized rows for a manifest."""
        query = "SELECT COUNT(*) FROM intel_pub.read_model WHERE manifest_id = $1"
        if conn is not None:
            value = await conn.fetchval(query, manifest_id)
        else:
            value = await self._db.fetchval(query, manifest_id)
        return int(value or 0)
