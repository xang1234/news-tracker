"""Feedback repository for CRUD operations and aggregation queries.

Follows the AlertRepository pattern with asyncpg, providing
storage, retrieval, and statistics for Feedback records.
"""

import logging
from typing import Any

from src.feedback.schemas import Feedback
from src.storage.database import Database

logger = logging.getLogger(__name__)


class FeedbackRepository:
    """Repository for feedback persistence and querying.

    Provides create, list_by_entity, and get_stats operations for
    Feedback records stored in the ``feedback`` table.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create(self, feedback: Feedback) -> Feedback:
        """Insert a new feedback record.

        Args:
            feedback: Feedback to persist.

        Returns:
            The created Feedback with DB-assigned defaults.
        """
        sql = """
            INSERT INTO feedback (
                feedback_id, entity_type, entity_id, rating,
                quality_label, comment, user_id, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
        """
        row = await self._db.fetchrow(
            sql,
            feedback.feedback_id,
            feedback.entity_type,
            feedback.entity_id,
            feedback.rating,
            feedback.quality_label,
            feedback.comment,
            feedback.user_id,
            feedback.created_at,
        )
        return _row_to_feedback(row)

    async def list_by_entity(
        self,
        entity_type: str,
        entity_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Feedback]:
        """Get feedback for a specific entity.

        Args:
            entity_type: Entity type filter (theme, alert, document).
            entity_id: Entity identifier.
            limit: Maximum records to return.
            offset: Offset for pagination.

        Returns:
            List of feedback ordered by created_at descending.
        """
        sql = """
            SELECT * FROM feedback
            WHERE entity_type = $1 AND entity_id = $2
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
        """
        rows = await self._db.fetch(sql, entity_type, entity_id, limit, offset)
        return [_row_to_feedback(row) for row in rows]

    async def get_stats(
        self,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated feedback statistics.

        Groups by entity_type and computes average rating,
        count, and distribution of quality labels.

        Args:
            entity_type: Optional filter to a single entity type.

        Returns:
            List of stat dicts with entity_type, count, avg_rating,
            and label_distribution.
        """
        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1

        if entity_type is not None:
            conditions.append(f"entity_type = ${param_idx}")
            params.append(entity_type)
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT
                entity_type,
                COUNT(*) AS total_count,
                ROUND(AVG(rating)::numeric, 2) AS avg_rating,
                COUNT(*) FILTER (WHERE quality_label = 'useful') AS label_useful,
                COUNT(*) FILTER (WHERE quality_label = 'noise') AS label_noise,
                COUNT(*) FILTER (WHERE quality_label = 'too_late') AS label_too_late,
                COUNT(*) FILTER (WHERE quality_label = 'wrong_direction') AS label_wrong_direction
            FROM feedback
            {where_clause}
            GROUP BY entity_type
            ORDER BY entity_type
        """
        rows = await self._db.fetch(sql, *params)
        return [_row_to_stats(row) for row in rows]


def _row_to_feedback(row: Any) -> Feedback:
    """Convert an asyncpg Record to a Feedback."""
    return Feedback(
        feedback_id=row["feedback_id"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        rating=row["rating"],
        quality_label=row.get("quality_label"),
        comment=row.get("comment"),
        user_id=row.get("user_id"),
        created_at=row["created_at"],
    )


def _row_to_stats(row: Any) -> dict[str, Any]:
    """Convert an asyncpg Record to a stats dict."""
    return {
        "entity_type": row["entity_type"],
        "total_count": row["total_count"],
        "avg_rating": float(row["avg_rating"]) if row["avg_rating"] else 0.0,
        "label_distribution": {
            "useful": row["label_useful"],
            "noise": row["label_noise"],
            "too_late": row["label_too_late"],
            "wrong_direction": row["label_wrong_direction"],
        },
    }
