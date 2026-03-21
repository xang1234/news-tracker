"""Repository helpers for narrative momentum."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.narrative.schemas import NarrativeRun, NarrativeRunBucket, NarrativeSignalState
from src.storage.database import Database


class NarrativeRepository:
    """Repository for narrative runs, buckets, and signal state."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def get_by_id(self, run_id: str) -> NarrativeRun | None:
        row = await self._db.fetchrow(
            "SELECT * FROM narrative_runs WHERE run_id = $1",
            run_id,
        )
        return _row_to_run(row) if row else None

    async def get_candidate_runs(
        self,
        theme_id: str,
        statuses: list[str] | None = None,
        limit: int = 5,
    ) -> list[NarrativeRun]:
        statuses = statuses or ["active", "cooling"]
        rows = await self._db.fetch(
            """
            SELECT * FROM narrative_runs
            WHERE theme_id = $1
              AND status = ANY($2::text[])
            ORDER BY last_document_at DESC
            LIMIT $3
            """,
            theme_id,
            statuses,
            limit,
        )
        return [_row_to_run(row) for row in rows]

    async def list_theme_runs(
        self,
        theme_id: str,
        status: str | None = None,
        limit: int = 20,
    ) -> list[NarrativeRun]:
        if status is None:
            rows = await self._db.fetch(
                """
                SELECT * FROM narrative_runs
                WHERE theme_id = $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                theme_id,
                limit,
            )
        else:
            rows = await self._db.fetch(
                """
                SELECT * FROM narrative_runs
                WHERE theme_id = $1 AND status = $2
                ORDER BY updated_at DESC
                LIMIT $3
                """,
                theme_id,
                status,
                limit,
            )
        return [_row_to_run(row) for row in rows]

    async def list_global_momentum(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = await self._db.fetch(
            """
            SELECT nr.*, t.name AS theme_name
            FROM narrative_runs nr
            JOIN themes t ON t.theme_id = nr.theme_id
            WHERE nr.status IN ('active', 'cooling')
            ORDER BY nr.conviction_score DESC, nr.current_rate_per_hour DESC, nr.updated_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [
            {
                "run": _row_to_run(row),
                "theme_name": row["theme_name"],
            }
            for row in rows
        ]

    async def get_run_documents(
        self,
        run_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = await self._db.fetch(
            """
            SELECT d.id, d.platform, d.title, d.content, d.url, d.author_name,
                   d.tickers, d.authority_score, d.sentiment, d.timestamp,
                   nrd.similarity, nrd.assigned_at
            FROM narrative_run_documents nrd
            JOIN documents d ON d.id = nrd.document_id
            WHERE nrd.run_id = $1
            ORDER BY nrd.assigned_at DESC
            LIMIT $2
            """,
            run_id,
            limit,
        )
        items: list[dict[str, Any]] = []
        for row in rows:
            sentiment = row["sentiment"]
            if isinstance(sentiment, str):
                sentiment = json.loads(sentiment)
            items.append({
                "document_id": row["id"],
                "platform": row["platform"],
                "title": row["title"],
                "content_preview": (row["content"] or "")[:300] or None,
                "url": row["url"],
                "author_name": row["author_name"],
                "tickers": row["tickers"] or [],
                "authority_score": row["authority_score"],
                "sentiment": sentiment,
                "timestamp": row["timestamp"],
                "similarity": row["similarity"],
                "assigned_at": row["assigned_at"],
            })
        return items

    async def get_recent_buckets(
        self,
        run_id: str,
        limit: int = 72,
    ) -> list[NarrativeRunBucket]:
        rows = await self._db.fetch(
            """
            SELECT * FROM narrative_run_buckets
            WHERE run_id = $1
            ORDER BY bucket_start DESC
            LIMIT $2
            """,
            run_id,
            limit,
        )
        buckets = [_row_to_bucket(row) for row in rows]
        return list(reversed(buckets))

    async def get_signal_states(self, run_id: str) -> dict[str, NarrativeSignalState]:
        rows = await self._db.fetch(
            """
            SELECT * FROM narrative_signal_state
            WHERE run_id = $1
            """,
            run_id,
        )
        return {
            row["trigger_type"]: _row_to_signal_state(row)
            for row in rows
        }

    async def get_recent_alerts(
        self,
        run_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        rows = await self._db.fetch(
            """
            SELECT alert_id, trigger_type, severity, title, message,
                   conviction_score, created_at, trigger_data
            FROM alerts
            WHERE subject_type = 'narrative_run'
              AND subject_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            run_id,
            limit,
        )
        items: list[dict[str, Any]] = []
        for row in rows:
            data = row["trigger_data"]
            if isinstance(data, str):
                data = json.loads(data)
            items.append({
                "alert_id": row["alert_id"],
                "trigger_type": row["trigger_type"],
                "severity": row["severity"],
                "title": row["title"],
                "message": row["message"],
                "conviction_score": row["conviction_score"],
                "created_at": row["created_at"],
                "trigger_data": data,
            })
        return items


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
    if isinstance(value, str):
        stripped = value.strip("[]")
        if not stripped:
            return np.zeros(768, dtype=np.float32)
        return np.array([float(x) for x in stripped.split(",")], dtype=np.float32)
    return np.zeros(768, dtype=np.float32)


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _row_to_run(row: Any) -> NarrativeRun:
    return NarrativeRun(
        run_id=row["run_id"],
        theme_id=row["theme_id"],
        status=row["status"],
        centroid=_parse_vector(row["centroid"]),
        label=row["label"],
        started_at=row["started_at"],
        last_document_at=row["last_document_at"],
        closed_at=row.get("closed_at"),
        doc_count=row["doc_count"],
        platform_first_seen=_parse_json(row.get("platform_first_seen")),
        ticker_counts={k: int(v) for k, v in _parse_json(row.get("ticker_counts")).items()},
        avg_sentiment=float(row.get("avg_sentiment") or 0.0),
        avg_authority=float(row.get("avg_authority") or 0.0),
        platform_count=int(row.get("platform_count") or 0),
        current_rate_per_hour=float(row.get("current_rate_per_hour") or 0.0),
        current_acceleration=float(row.get("current_acceleration") or 0.0),
        conviction_score=float(row.get("conviction_score") or 0.0),
        last_signal_at=row.get("last_signal_at"),
        metadata=_parse_json(row.get("metadata")),
        created_at=row.get("created_at") or row["started_at"],
        updated_at=row.get("updated_at") or row["last_document_at"],
    )


def _row_to_bucket(row: Any) -> NarrativeRunBucket:
    return NarrativeRunBucket(
        run_id=row["run_id"],
        bucket_start=row["bucket_start"],
        doc_count=int(row.get("doc_count") or 0),
        platform_counts={k: int(v) for k, v in _parse_json(row.get("platform_counts")).items()},
        ticker_counts={k: int(v) for k, v in _parse_json(row.get("ticker_counts")).items()},
        sentiment_sum=float(row.get("sentiment_sum") or 0.0),
        sentiment_weight=float(row.get("sentiment_weight") or 0.0),
        sentiment_confidence_sum=float(row.get("sentiment_confidence_sum") or 0.0),
        sentiment_doc_count=int(row.get("sentiment_doc_count") or 0),
        authority_sum=float(row.get("authority_sum") or 0.0),
        high_authority_sentiment_sum=float(row.get("high_authority_sentiment_sum") or 0.0),
        high_authority_weight=float(row.get("high_authority_weight") or 0.0),
        high_authority_doc_count=int(row.get("high_authority_doc_count") or 0),
        low_authority_sentiment_sum=float(row.get("low_authority_sentiment_sum") or 0.0),
        low_authority_weight=float(row.get("low_authority_weight") or 0.0),
        low_authority_doc_count=int(row.get("low_authority_doc_count") or 0),
    )


def _row_to_signal_state(row: Any) -> NarrativeSignalState:
    return NarrativeSignalState(
        run_id=row["run_id"],
        trigger_type=row["trigger_type"],
        state=row["state"],
        last_score=float(row.get("last_score") or 0.0),
        last_alert_at=row.get("last_alert_at"),
        last_transition_at=row["last_transition_at"],
        cooldown_until=row.get("cooldown_until"),
        metadata=_parse_json(row.get("metadata")),
    )
