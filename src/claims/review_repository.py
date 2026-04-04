"""Repository for the claim review queue.

CRUD operations against news_intel.review_tasks with idempotent
upserts keyed by task_id. Supports listing, filtering, state
transitions, and resolution recording.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.claims.review import (
    VALID_RESOLUTIONS,
    ReviewTask,
    validate_review_transition,
)
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


def _parse_text_array(value: Any) -> list[str]:
    """Parse a TEXT[] column value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _row_to_task(row: Any) -> ReviewTask:
    return ReviewTask(
        task_id=row["task_id"],
        task_type=row["task_type"],
        trigger_reason=row["trigger_reason"],
        status=row["status"],
        claim_ids=_parse_text_array(row["claim_ids"]),
        concept_ids=_parse_text_array(row["concept_ids"]),
        priority=row["priority"],
        assigned_to=row["assigned_to"],
        resolution=row["resolution"],
        resolution_notes=row["resolution_notes"],
        payload=_parse_json(row["payload"]),
        lineage=_parse_json(row["lineage"]),
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class ReviewRepository:
    """CRUD operations for review tasks."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_task(self, task: ReviewTask) -> ReviewTask:
        """Insert or update a review task (idempotent on task_id).

        On conflict (same task_id), updates priority and metadata
        but preserves the original trigger and lineage.
        """
        row = await self._db.fetchrow(
            """
            INSERT INTO news_intel.review_tasks (
                task_id, task_type, trigger_reason, status,
                claim_ids, concept_ids, priority,
                assigned_to, resolution, resolution_notes,
                payload, lineage, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7,
                $8, $9, $10, $11, $12, $13
            )
            ON CONFLICT (task_id) DO UPDATE SET
                priority = LEAST(news_intel.review_tasks.priority, $7),
                metadata = $13
            RETURNING *
            """,
            task.task_id,
            task.task_type,
            task.trigger_reason,
            task.status,
            task.claim_ids,
            task.concept_ids,
            task.priority,
            task.assigned_to,
            task.resolution,
            task.resolution_notes,
            json.dumps(task.payload),
            json.dumps(task.lineage),
            json.dumps(task.metadata),
        )
        return _row_to_task(row)

    async def get_task(self, task_id: str) -> ReviewTask | None:
        """Fetch a review task by ID."""
        row = await self._db.fetchrow(
            "SELECT * FROM news_intel.review_tasks WHERE task_id = $1",
            task_id,
        )
        return _row_to_task(row) if row else None

    async def list_tasks(
        self,
        *,
        task_type: str | None = None,
        status: str | None = None,
        trigger_reason: str | None = None,
        concept_id: str | None = None,
        claim_id: str | None = None,
        assigned_to: str | None = None,
        limit: int = 50,
    ) -> list[ReviewTask]:
        """List review tasks with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if task_type is not None:
            params.append(task_type)
            conditions.append(f"task_type = ${len(params)}")
        if status is not None:
            params.append(status)
            conditions.append(f"status = ${len(params)}")
        if trigger_reason is not None:
            params.append(trigger_reason)
            conditions.append(f"trigger_reason = ${len(params)}")
        if concept_id is not None:
            params.append(concept_id)
            conditions.append(f"${len(params)} = ANY(concept_ids)")
        if claim_id is not None:
            params.append(claim_id)
            conditions.append(f"${len(params)} = ANY(claim_ids)")
        if assigned_to is not None:
            params.append(assigned_to)
            conditions.append(f"assigned_to = ${len(params)}")

        params.append(limit)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT * FROM news_intel.review_tasks
            {where}
            ORDER BY priority ASC, created_at ASC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_task(row) for row in rows]

    async def transition_task(
        self,
        task_id: str,
        target_status: str,
        *,
        resolution: str | None = None,
        resolution_notes: str | None = None,
        assigned_to: str | None = None,
    ) -> tuple[str, ReviewTask]:
        """Transition a review task's status.

        Returns (previous_status, updated_task).
        Raises ValueError on invalid transition or missing task.
        """
        if resolution is not None and resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid resolution {resolution!r}. "
                f"Must be one of {sorted(VALID_RESOLUTIONS)}"
            )

        current = await self.get_task(task_id)
        if current is None:
            raise ValueError(f"Review task not found: {task_id}")

        validate_review_transition(current.status, target_status)

        # Resolved tasks must have a resolution
        if target_status == "resolved" and resolution is None:
            raise ValueError(
                "Resolution is required when transitioning to 'resolved'"
            )

        row = await self._db.fetchrow(
            """
            UPDATE news_intel.review_tasks
            SET status = $2,
                resolution = COALESCE($3, resolution),
                resolution_notes = COALESCE($4, resolution_notes),
                assigned_to = COALESCE($5, assigned_to)
            WHERE task_id = $1
            RETURNING *
            """,
            task_id,
            target_status,
            resolution,
            resolution_notes,
            assigned_to,
        )
        return current.status, _row_to_task(row)

    async def count_pending(
        self,
        *,
        task_type: str | None = None,
    ) -> int:
        """Count pending review tasks, optionally by type."""
        if task_type is not None:
            row = await self._db.fetchrow(
                """
                SELECT COUNT(*) AS cnt FROM news_intel.review_tasks
                WHERE status = 'pending' AND task_type = $1
                """,
                task_type,
            )
        else:
            row = await self._db.fetchrow(
                "SELECT COUNT(*) AS cnt FROM news_intel.review_tasks "
                "WHERE status = 'pending'"
            )
        return row["cnt"] if row else 0

    async def get_tasks_for_claim(
        self, claim_id: str
    ) -> list[ReviewTask]:
        """Get all review tasks linked to a specific claim."""
        rows = await self._db.fetch(
            """
            SELECT * FROM news_intel.review_tasks
            WHERE $1 = ANY(claim_ids)
            ORDER BY priority ASC, created_at ASC
            """,
            claim_id,
        )
        return [_row_to_task(row) for row in rows]

    async def get_tasks_for_concept(
        self, concept_id: str
    ) -> list[ReviewTask]:
        """Get all review tasks linked to a specific concept."""
        rows = await self._db.fetch(
            """
            SELECT * FROM news_intel.review_tasks
            WHERE $1 = ANY(concept_ids)
            ORDER BY priority ASC, created_at ASC
            """,
            concept_id,
        )
        return [_row_to_task(row) for row in rows]
