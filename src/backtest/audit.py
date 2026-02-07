"""Backtest run audit logging.

Tracks the lifecycle of backtest executions: running → completed | failed.
Stores parameters, results, and error messages for reproducibility.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.storage.database import Database

logger = logging.getLogger(__name__)


@dataclass
class BacktestRun:
    """A single backtest execution record."""

    run_id: str
    model_version_id: str
    date_range_start: Any  # date
    date_range_end: Any  # date
    parameters: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] | None = None
    status: str = "running"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime | None = None
    error_message: str | None = None


class BacktestRunRepository:
    """CRUD operations for backtest run audit records."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create(self, run: BacktestRun) -> BacktestRun:
        """Insert a new backtest run record.

        Args:
            run: BacktestRun to persist.

        Returns:
            The persisted BacktestRun with DB timestamps.
        """
        sql = """
            INSERT INTO backtest_runs (
                run_id, model_version_id, date_range_start, date_range_end,
                parameters, status
            ) VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
        """
        row = await self._db.fetchrow(
            sql,
            run.run_id,
            run.model_version_id,
            run.date_range_start,
            run.date_range_end,
            json.dumps(run.parameters),
            run.status,
        )
        return _row_to_run(row)

    async def mark_completed(
        self,
        run_id: str,
        results: dict[str, Any],
    ) -> BacktestRun:
        """Mark a backtest run as completed with results.

        Args:
            run_id: Run identifier.
            results: Backtest results dictionary.

        Returns:
            The updated BacktestRun.

        Raises:
            ValueError: If run not found.
        """
        sql = """
            UPDATE backtest_runs
            SET status = 'completed',
                results = $2,
                completed_at = NOW()
            WHERE run_id = $1
            RETURNING *
        """
        row = await self._db.fetchrow(sql, run_id, json.dumps(results))
        if row is None:
            raise ValueError(f"BacktestRun {run_id!r} not found")
        return _row_to_run(row)

    async def mark_failed(
        self,
        run_id: str,
        error: str,
    ) -> BacktestRun:
        """Mark a backtest run as failed with an error message.

        Args:
            run_id: Run identifier.
            error: Error description.

        Returns:
            The updated BacktestRun.

        Raises:
            ValueError: If run not found.
        """
        sql = """
            UPDATE backtest_runs
            SET status = 'failed',
                error_message = $2,
                completed_at = NOW()
            WHERE run_id = $1
            RETURNING *
        """
        row = await self._db.fetchrow(sql, run_id, error)
        if row is None:
            raise ValueError(f"BacktestRun {run_id!r} not found")
        return _row_to_run(row)

    async def get_by_id(self, run_id: str) -> BacktestRun | None:
        """Get a backtest run by ID.

        Args:
            run_id: Run identifier.

        Returns:
            BacktestRun or None if not found.
        """
        sql = "SELECT * FROM backtest_runs WHERE run_id = $1"
        row = await self._db.fetchrow(sql, run_id)
        if row is None:
            return None
        return _row_to_run(row)

    async def list_runs(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[BacktestRun]:
        """List backtest runs, optionally filtered by status.

        Args:
            status: Optional status filter ('running', 'completed', 'failed').
            limit: Maximum runs to return.

        Returns:
            List of BacktestRun records ordered by created_at descending.
        """
        if status:
            sql = """
                SELECT * FROM backtest_runs
                WHERE status = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            rows = await self._db.fetch(sql, status, limit)
        else:
            sql = """
                SELECT * FROM backtest_runs
                ORDER BY created_at DESC
                LIMIT $1
            """
            rows = await self._db.fetch(sql, limit)

        return [_row_to_run(row) for row in rows]


def _row_to_run(row: Any) -> BacktestRun:
    """Convert an asyncpg Record to a BacktestRun."""
    parameters = row.get("parameters", {})
    if isinstance(parameters, str):
        parameters = json.loads(parameters)

    results = row.get("results")
    if isinstance(results, str):
        results = json.loads(results)

    return BacktestRun(
        run_id=row["run_id"],
        model_version_id=row["model_version_id"],
        date_range_start=row["date_range_start"],
        date_range_end=row["date_range_end"],
        parameters=parameters,
        results=results,
        status=row["status"],
        created_at=row["created_at"],
        completed_at=row.get("completed_at"),
        error_message=row.get("error_message"),
    )
