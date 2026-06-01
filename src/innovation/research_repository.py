"""Repository for research-derived innovation evidence."""

from __future__ import annotations

import json
from typing import Any

from src.innovation.research_schemas import ResearchSignal
from src.storage.database import Database


def _parse_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_signal(row: Any) -> ResearchSignal:
    return ResearchSignal(
        source=row["source"],
        record_id=row["record_id"],
        published_date=row["published_date"],
        title=row["title"],
        issuer_concept_id=row["issuer_concept_id"],
        security_concept_id=row["security_concept_id"],
        theme_id=row["theme_id"],
        confidence=float(row["confidence"]),
        confidence_reasons=list(_parse_json(row["confidence_reasons"], [])),
        source_lineage=dict(_parse_json(row["source_lineage"], {})),
        metadata=dict(_parse_json(row["metadata"], {})),
        url=row["url"],
        fetched_at=row["fetched_at"],
    )


class ResearchSignalRepository:
    """Persistence operations for research innovation signals."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_signals(self, signals: list[ResearchSignal]) -> list[ResearchSignal]:
        written: list[ResearchSignal] = []
        for signal in signals:
            row = await self._db.fetchrow(
                """
                INSERT INTO innovation_research_signals (
                    source, record_id, published_date, title,
                    issuer_concept_id, security_concept_id, theme_id,
                    confidence, url, fetched_at, confidence_reasons,
                    source_lineage, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7,
                    $8, $9, $10, $11, $12, $13
                )
                ON CONFLICT (
                    source, record_id, issuer_concept_id,
                    security_concept_id, theme_id
                ) DO UPDATE SET
                    published_date = $3,
                    title = $4,
                    confidence = $8,
                    url = $9,
                    fetched_at = $10,
                    confidence_reasons = $11,
                    source_lineage = $12,
                    metadata = $13,
                    updated_at = NOW()
                RETURNING *
                """,
                signal.source,
                signal.record_id,
                signal.published_date,
                signal.title,
                signal.issuer_concept_id,
                signal.security_concept_id,
                signal.theme_id,
                signal.confidence,
                signal.url,
                signal.fetched_at,
                json.dumps(signal.confidence_reasons),
                json.dumps(signal.source_lineage),
                json.dumps(signal.metadata),
            )
            written.append(_row_to_signal(row))
        return written

    async def list_signals(
        self,
        *,
        source: str | None = None,
        theme_id: str | None = None,
        issuer_concept_id: str | None = None,
        limit: int = 100,
    ) -> list[ResearchSignal]:
        conditions: list[str] = []
        params: list[Any] = []
        if source is not None:
            params.append(source)
            conditions.append(f"source = ${len(params)}")
        if theme_id is not None:
            params.append(theme_id)
            conditions.append(f"theme_id = ${len(params)}")
        if issuer_concept_id is not None:
            params.append(issuer_concept_id)
            conditions.append(f"issuer_concept_id = ${len(params)}")

        params.append(limit)
        limit_idx = len(params)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT *
            FROM innovation_research_signals
            {where}
            ORDER BY published_date DESC, confidence DESC, record_id
            LIMIT ${limit_idx}
            """,
            *params,
        )
        return [_row_to_signal(row) for row in rows]
