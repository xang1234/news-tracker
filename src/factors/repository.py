"""Database repository for factor series and point-in-time observations."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

from src.factors.schemas import FactorObservation, FactorSeries
from src.storage.database import Database
from src.storage.migrations import apply_migrations

logger = logging.getLogger(__name__)

_UPSERT_SERIES_SQL = """
INSERT INTO factor_series (
    factor_id, provider, external_id, name, description, units, cadence,
    release_lag_days, relevance_tags, required_credentials, source_url, is_active, metadata
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
ON CONFLICT (factor_id) DO UPDATE SET
    provider = EXCLUDED.provider,
    external_id = EXCLUDED.external_id,
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    units = EXCLUDED.units,
    cadence = EXCLUDED.cadence,
    release_lag_days = EXCLUDED.release_lag_days,
    relevance_tags = EXCLUDED.relevance_tags,
    required_credentials = EXCLUDED.required_credentials,
    source_url = EXCLUDED.source_url,
    is_active = EXCLUDED.is_active,
    metadata = EXCLUDED.metadata,
    updated_at = NOW()
RETURNING *
"""

_UPSERT_OBSERVATION_SQL = """
INSERT INTO factor_observations (
    factor_id, observation_date, value, units, available_at, fetched_at,
    revision, missing_reason, metadata
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (factor_id, observation_date, available_at, revision) DO UPDATE SET
    value = EXCLUDED.value,
    units = EXCLUDED.units,
    fetched_at = EXCLUDED.fetched_at,
    missing_reason = EXCLUDED.missing_reason,
    metadata = EXCLUDED.metadata
RETURNING *
"""


class FactorRepository:
    """CRUD operations for factor registry and observation tables."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create_table(self) -> None:
        """Backward-compatible schema helper for factor tables."""
        await apply_migrations(self._db)
        logger.info("Factor schema ensured via migrations")

    async def upsert_series(self, series: FactorSeries) -> FactorSeries:
        """Insert or update a factor registry entry."""
        row = await self._db.fetchrow(
            _UPSERT_SERIES_SQL,
            *_series_upsert_args(series),
        )
        return _record_to_series(row)

    async def get_series(self, factor_id: str) -> FactorSeries | None:
        """Fetch one factor registry entry."""
        row = await self._db.fetchrow(
            "SELECT * FROM factor_series WHERE factor_id = $1",
            factor_id,
        )
        return _record_to_series(row) if row else None

    async def list_series(
        self,
        *,
        active_only: bool = False,
        relevance_tag: str | None = None,
        provider: str | None = None,
    ) -> list[FactorSeries]:
        """List factor registry entries with lightweight filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if active_only:
            conditions.append("is_active = TRUE")
        if relevance_tag is not None:
            params.append(relevance_tag)
            conditions.append(f"${len(params)} = ANY(relevance_tags)")
        if provider is not None:
            params.append(provider)
            conditions.append(f"provider = ${len(params)}")

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT * FROM factor_series{where_clause}
            ORDER BY provider, external_id
            """,
            *params,
        )
        return [_record_to_series(row) for row in rows]

    async def upsert_observation(self, observation: FactorObservation) -> FactorObservation:
        """Insert or update one point-in-time factor observation."""
        row = await self._db.fetchrow(
            _UPSERT_OBSERVATION_SQL,
            *_observation_upsert_args(observation),
        )
        return _record_to_observation(row)

    async def upsert_observations(
        self,
        observations: list[FactorObservation],
    ) -> list[FactorObservation]:
        """Insert or update point-in-time observations as one atomic write."""
        if not observations:
            return []

        async with self._db.transaction() as connection:
            return [
                _record_to_observation(
                    await connection.fetchrow(
                        _UPSERT_OBSERVATION_SQL,
                        *_observation_upsert_args(observation),
                    )
                )
                for observation in observations
            ]

    async def upsert_series_with_observations(
        self,
        series: FactorSeries,
        observations: list[FactorObservation],
    ) -> tuple[FactorSeries, list[FactorObservation]]:
        """Insert or update a registry entry and its observations atomically."""
        async with self._db.transaction() as connection:
            series_row = await connection.fetchrow(
                _UPSERT_SERIES_SQL,
                *_series_upsert_args(series),
            )
            observation_rows = [
                await connection.fetchrow(
                    _UPSERT_OBSERVATION_SQL,
                    *_observation_upsert_args(observation),
                )
                for observation in observations
            ]
        return _record_to_series(series_row), [
            _record_to_observation(row) for row in observation_rows
        ]

    async def get_observations_as_of(
        self,
        factor_id: str,
        *,
        start: date,
        end: date,
        as_of: datetime,
    ) -> list[FactorObservation]:
        """Fetch latest observations available as of a point in time."""
        rows = await self._db.fetch(
            """
            SELECT DISTINCT ON (observation_date) *
            FROM factor_observations
            WHERE factor_id = $1
              AND observation_date >= $2
              AND observation_date <= $3
              AND available_at <= $4
            ORDER BY observation_date ASC, available_at DESC, fetched_at DESC
            """,
            factor_id,
            start,
            end,
            as_of,
        )
        return [_record_to_observation(row) for row in rows]

    async def get_latest_observations_as_of(
        self,
        factor_ids: list[str],
        *,
        start: date,
        end: date,
        as_of: datetime,
        per_factor: int = 2,
    ) -> dict[str, list[FactorObservation]]:
        """Fetch the latest visible observations for multiple factors in one query."""
        if not factor_ids:
            return {}

        rows = await self._db.fetch(
            """
            WITH latest_per_observation AS (
                SELECT DISTINCT ON (factor_id, observation_date) *
                FROM factor_observations
                WHERE factor_id = ANY($1::text[])
                  AND observation_date >= $2
                  AND observation_date <= $3
                  AND available_at <= $4
                ORDER BY factor_id, observation_date DESC, available_at DESC, fetched_at DESC
            ),
            ranked AS (
                SELECT
                    latest_per_observation.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY factor_id
                        ORDER BY observation_date DESC
                    ) AS factor_rank
                FROM latest_per_observation
            )
            SELECT *
            FROM ranked
            WHERE factor_rank <= $5
            ORDER BY factor_id, observation_date ASC
            """,
            factor_ids,
            start,
            end,
            as_of,
            per_factor,
        )

        grouped: dict[str, list[FactorObservation]] = {factor_id: [] for factor_id in factor_ids}
        for row in rows:
            grouped.setdefault(row["factor_id"], []).append(_record_to_observation(row))
        return grouped


def _series_upsert_args(series: FactorSeries) -> tuple[Any, ...]:
    return (
        series.factor_id,
        series.provider,
        series.external_id,
        series.name,
        series.description,
        series.units,
        series.cadence,
        series.release_lag_days,
        series.relevance_tags,
        series.required_credentials,
        series.source_url,
        series.is_active,
        json.dumps(series.metadata),
    )


def _observation_upsert_args(observation: FactorObservation) -> tuple[Any, ...]:
    return (
        observation.factor_id,
        observation.observation_date,
        observation.value,
        observation.units,
        observation.available_at,
        observation.fetched_at,
        observation.revision,
        observation.missing_reason,
        json.dumps(observation.metadata),
    )


def _parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return dict(value or {})


def _record_to_series(record: Any) -> FactorSeries:
    return FactorSeries(
        factor_id=record["factor_id"],
        provider=record["provider"],
        external_id=record["external_id"],
        name=record["name"],
        description=record["description"],
        units=record["units"],
        cadence=record["cadence"],
        release_lag_days=record["release_lag_days"],
        relevance_tags=list(record["relevance_tags"] or []),
        required_credentials=list(record["required_credentials"] or []),
        source_url=record["source_url"],
        is_active=record["is_active"],
        metadata=_parse_metadata(record["metadata"]),
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


def _record_to_observation(record: Any) -> FactorObservation:
    return FactorObservation(
        factor_id=record["factor_id"],
        observation_date=record["observation_date"],
        value=record["value"],
        units=record["units"],
        available_at=record["available_at"],
        fetched_at=record["fetched_at"],
        revision=record["revision"],
        missing_reason=record["missing_reason"],
        metadata=_parse_metadata(record["metadata"]),
    )
