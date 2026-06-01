"""Repository for innovation evidence signals."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from src.innovation.patent_schemas import PatentSignal
from src.storage.database import Database


def _parse_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_signal(row: Any) -> PatentSignal:
    return PatentSignal(
        patent_id=row["patent_id"],
        patent_family_id=row["patent_family_id"],
        event_type=row["event_type"],
        event_date=row["event_date"],
        title=row["title"],
        issuer_concept_id=row["issuer_concept_id"],
        security_concept_id=row["security_concept_id"],
        theme_id=row["theme_id"],
        confidence=float(row["confidence"]),
        confidence_reasons=list(_parse_json(row["confidence_reasons"], [])),
        source_lineage=dict(_parse_json(row["source_lineage"], {})),
        metadata=dict(_parse_json(row["metadata"], {})),
        source_url=row["source_url"],
        fetched_at=row["fetched_at"],
    )


class PatentSignalRepository:
    """Persistence operations for patent-derived innovation signals."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_signals(self, signals: list[PatentSignal]) -> list[PatentSignal]:
        """Insert or update patent signals idempotently."""
        written: list[PatentSignal] = []
        for signal in signals:
            row = await self._db.fetchrow(
                """
                INSERT INTO innovation_patent_signals (
                    patent_id, patent_family_id, event_type, event_date, title,
                    issuer_concept_id, security_concept_id, theme_id,
                    confidence, source_url, confidence_reasons, source_lineage,
                    metadata, fetched_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7,
                    $8, $9, $10, $11, $12, $13, $14
                )
                ON CONFLICT (
                    patent_id, event_type, issuer_concept_id,
                    security_concept_id, theme_id
                ) DO UPDATE SET
                    patent_family_id = $2,
                    event_date = $4,
                    title = $5,
                    confidence = $9,
                    source_url = $10,
                    confidence_reasons = $11,
                    source_lineage = $12,
                    metadata = $13,
                    fetched_at = $14,
                    updated_at = NOW()
                RETURNING *
                """,
                signal.patent_id,
                signal.patent_family_id,
                signal.event_type,
                signal.event_date,
                signal.title,
                signal.issuer_concept_id,
                signal.security_concept_id,
                signal.theme_id,
                signal.confidence,
                signal.source_url,
                json.dumps(signal.confidence_reasons),
                json.dumps(signal.source_lineage),
                json.dumps(signal.metadata),
                signal.fetched_at,
            )
            written.append(_row_to_signal(row))
        return written

    async def list_signals(
        self,
        *,
        theme_id: str | None = None,
        issuer_concept_id: str | None = None,
        security_concept_id: str | None = None,
        start: date | None = None,
        end: date | None = None,
        limit: int = 100,
    ) -> list[PatentSignal]:
        """List patent signals for explanation and downstream lane assembly."""
        conditions: list[str] = []
        params: list[Any] = []
        if theme_id is not None:
            params.append(theme_id)
            conditions.append(f"theme_id = ${len(params)}")
        if issuer_concept_id is not None:
            params.append(issuer_concept_id)
            conditions.append(f"issuer_concept_id = ${len(params)}")
        if security_concept_id is not None:
            params.append(security_concept_id)
            conditions.append(f"security_concept_id = ${len(params)}")
        if start is not None:
            params.append(start)
            conditions.append(f"event_date >= ${len(params)}")
        if end is not None:
            params.append(end)
            conditions.append(f"event_date <= ${len(params)}")

        params.append(limit)
        limit_idx = len(params)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT *
            FROM innovation_patent_signals
            {where}
            ORDER BY event_date DESC, confidence DESC, patent_id
            LIMIT ${limit_idx}
            """,
            *params,
        )
        return [_row_to_signal(row) for row in rows]
