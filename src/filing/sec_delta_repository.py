"""Persistence for SEC filing-delta events."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.filing.sec_delta_models import SECFilingDeltaEvent
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
    return {}


def _row_to_event(row: Any) -> SECFilingDeltaEvent:
    return SECFilingDeltaEvent(
        event_id=row["event_id"],
        cik=row["cik"],
        event_type=row["event_type"],
        accession_number=row["accession_number"],
        previous_accession_number=row["previous_accession_number"],
        taxonomy=row["taxonomy"],
        fact_name=row["fact_name"],
        unit=row["unit"],
        period_start=row["period_start"],
        period_end=row["period_end"],
        previous_period_start=row["previous_period_start"],
        previous_period_end=row["previous_period_end"],
        filed_date=row["filed_date"],
        previous_filed_date=row["previous_filed_date"],
        form=row["form"],
        previous_form=row["previous_form"],
        available_at=row["available_at"],
        fetched_at=row["fetched_at"],
        current_value=row["current_value"],
        previous_value=row["previous_value"],
        absolute_delta=row["absolute_delta"],
        relative_delta=row["relative_delta"],
        source_payload_hash=row["source_payload_hash"],
        source_url=row["source_url"],
        metadata=_parse_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class SECFilingDeltaRepository:
    """Repository for idempotent SEC filing-delta event persistence."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_event(self, event: SECFilingDeltaEvent) -> SECFilingDeltaEvent:
        row = await self._db.fetchrow(
            """
            INSERT INTO sec_filing_delta_events (
                event_id, cik, event_type, accession_number,
                previous_accession_number, taxonomy, fact_name, unit,
                period_start, period_end, previous_period_start, previous_period_end,
                filed_date, previous_filed_date, form, previous_form,
                available_at, fetched_at, current_value, previous_value,
                absolute_delta, relative_delta, source_payload_hash, source_url, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25
            )
            ON CONFLICT (event_id) DO UPDATE SET
                current_value = $19,
                previous_value = $20,
                absolute_delta = $21,
                relative_delta = $22,
                source_payload_hash = $23,
                source_url = $24,
                fetched_at = $18,
                metadata = $25,
                updated_at = NOW()
            RETURNING *
            """,
            event.event_id,
            event.cik,
            event.event_type,
            event.accession_number,
            event.previous_accession_number,
            event.taxonomy,
            event.fact_name,
            event.unit,
            event.period_start,
            event.period_end,
            event.previous_period_start,
            event.previous_period_end,
            event.filed_date,
            event.previous_filed_date,
            event.form,
            event.previous_form,
            event.available_at,
            event.fetched_at,
            event.current_value,
            event.previous_value,
            event.absolute_delta,
            event.relative_delta,
            event.source_payload_hash,
            event.source_url,
            json.dumps(event.metadata),
        )
        return _row_to_event(row)

    async def upsert_events(
        self,
        events: list[SECFilingDeltaEvent],
    ) -> list[SECFilingDeltaEvent]:
        return [await self.upsert_event(event) for event in events]

    async def list_events_as_of(
        self,
        cik: str,
        *,
        as_of: datetime,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[SECFilingDeltaEvent]:
        normalized_cik = normalize_sec_cik(cik)
        if normalized_cik is None:
            raise ValueError("cik must be a non-empty SEC CIK")

        params: list[Any] = [normalized_cik, as_of]
        conditions = ["cik = $1", "available_at <= $2"]
        if event_type is not None:
            params.append(event_type)
            conditions.append(f"event_type = ${len(params)}")
        params.append(limit)

        rows = await self._db.fetch(
            f"""
            SELECT *
            FROM sec_filing_delta_events
            WHERE {" AND ".join(conditions)}
            ORDER BY available_at DESC, filed_date DESC, event_id DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_event(row) for row in rows]
