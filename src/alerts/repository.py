"""Alert repository for CRUD operations and rate limit queries.

Follows the ThemeRepository pattern with asyncpg, providing
storage, retrieval, and counting operations for Alert records.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.alerts.schemas import Alert
from src.storage.database import Database

logger = logging.getLogger(__name__)


class AlertRepository:
    """Repository for alert persistence and querying.

    Provides create, read, count, and acknowledge operations for
    Alert records stored in the ``alerts`` table.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create(self, alert: Alert) -> Alert:
        """Insert a new alert.

        Args:
            alert: Alert to persist.

        Returns:
            The created Alert with DB-assigned defaults.
        """
        sql = """
            INSERT INTO alerts (
                alert_id, theme_id, trigger_type, severity,
                title, message, trigger_data, acknowledged, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
        """
        row = await self._db.fetchrow(
            sql,
            alert.alert_id,
            alert.theme_id,
            alert.trigger_type,
            alert.severity,
            alert.title,
            alert.message,
            json.dumps(alert.trigger_data),
            alert.acknowledged,
            alert.created_at,
        )
        return _row_to_alert(row)

    async def create_batch(self, alerts: list[Alert]) -> list[Alert]:
        """Insert multiple alerts with per-alert error handling.

        Args:
            alerts: Alerts to persist.

        Returns:
            List of successfully created alerts.
        """
        created: list[Alert] = []
        for alert in alerts:
            try:
                result = await self.create(alert)
                created.append(result)
            except Exception as e:
                logger.error(
                    "Failed to persist alert %s: %s", alert.alert_id, e,
                )
        return created

    async def get_by_id(self, alert_id: str) -> Alert | None:
        """Get an alert by ID.

        Args:
            alert_id: Alert identifier.

        Returns:
            Alert or None if not found.
        """
        sql = "SELECT * FROM alerts WHERE alert_id = $1"
        row = await self._db.fetchrow(sql, alert_id)
        if row is None:
            return None
        return _row_to_alert(row)

    async def get_recent(
        self,
        *,
        severity: str | None = None,
        trigger_type: str | None = None,
        theme_id: str | None = None,
        acknowledged: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Alert]:
        """Get recent alerts with optional filtering.

        Uses dynamic SQL builder with incremental param_idx
        (matches DocumentRepository pattern).

        Args:
            severity: Filter by severity level.
            trigger_type: Filter by trigger type.
            theme_id: Filter by theme.
            acknowledged: Filter by acknowledgement status.
            limit: Maximum alerts to return.
            offset: Offset for pagination.

        Returns:
            List of alerts ordered by created_at descending.
        """
        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1

        if severity is not None:
            conditions.append(f"severity = ${param_idx}")
            params.append(severity)
            param_idx += 1

        if trigger_type is not None:
            conditions.append(f"trigger_type = ${param_idx}")
            params.append(trigger_type)
            param_idx += 1

        if theme_id is not None:
            conditions.append(f"theme_id = ${param_idx}")
            params.append(theme_id)
            param_idx += 1

        if acknowledged is not None:
            conditions.append(f"acknowledged = ${param_idx}")
            params.append(acknowledged)
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT * FROM alerts
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        rows = await self._db.fetch(sql, *params)
        return [_row_to_alert(row) for row in rows]

    async def count_today_by_severity(self, severity: str) -> int:
        """Count alerts created today for a specific severity.

        Used for rate limiting — counts from midnight UTC to now.

        Args:
            severity: Severity level to count.

        Returns:
            Number of alerts created today with the given severity.
        """
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        sql = """
            SELECT COUNT(*) FROM alerts
            WHERE severity = $1 AND created_at >= $2
        """
        count = await self._db.fetchval(sql, severity, today_start)
        return count or 0

    async def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged.

        Args:
            alert_id: Alert to acknowledge.

        Returns:
            True if updated, False if alert not found.
        """
        sql = """
            UPDATE alerts SET acknowledged = TRUE
            WHERE alert_id = $1 AND acknowledged = FALSE
            RETURNING alert_id
        """
        result = await self._db.fetchval(sql, alert_id)
        return result is not None


def _row_to_alert(row: Any) -> Alert:
    """Convert an asyncpg Record to an Alert."""
    trigger_data = row.get("trigger_data", {})
    if isinstance(trigger_data, str):
        trigger_data = json.loads(trigger_data)

    return Alert(
        alert_id=row["alert_id"],
        theme_id=row["theme_id"],
        trigger_type=row["trigger_type"],
        severity=row["severity"],
        title=row["title"],
        message=row["message"],
        trigger_data=trigger_data,
        acknowledged=row.get("acknowledged", False),
        created_at=row["created_at"],
    )
