"""Repository for intelligence layer publish operations.

CRUD operations against the news_intel, intel_pub, and intel_export
schema tables. All methods are async and use positional SQL parameters.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from src.contracts.intelligence.db_schemas import (
    ExportBundle,
    LaneRun,
    Manifest,
    ManifestPointer,
    PublishedObject,
)
from src.storage.database import Database

logger = logging.getLogger(__name__)


# -- Row converters --------------------------------------------------------


def _parse_json(value: Any) -> dict[str, Any]:
    """Parse a JSONB column value into a dict.

    asyncpg returns JSONB as str (no codec registered), but this
    also handles dict (if a codec is registered later) and None.
    """
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value)


def _row_to_lane_run(row: Any) -> LaneRun:
    return LaneRun(
        run_id=row["run_id"],
        lane=row["lane"],
        status=row["status"],
        contract_version=row["contract_version"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        error_message=row["error_message"],
        config_snapshot=_parse_json(row["config_snapshot"]),
        metrics=_parse_json(row["metrics"]),
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_manifest(row: Any) -> Manifest:
    return Manifest(
        manifest_id=row["manifest_id"],
        lane=row["lane"],
        run_id=row["run_id"],
        contract_version=row["contract_version"],
        published_at=row["published_at"],
        object_count=row["object_count"],
        checksum=row["checksum"],
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
    )


def _row_to_manifest_pointer(row: Any) -> ManifestPointer:
    return ManifestPointer(
        lane=row["lane"],
        manifest_id=row["manifest_id"],
        activated_at=row["activated_at"],
        previous_manifest_id=row["previous_manifest_id"],
        metadata=_parse_json(row["metadata"]),
    )


def _row_to_published_object(row: Any) -> PublishedObject:
    return PublishedObject(
        object_id=row["object_id"],
        object_type=row["object_type"],
        manifest_id=row["manifest_id"],
        lane=row["lane"],
        publish_state=row["publish_state"],
        contract_version=row["contract_version"],
        source_ids=list(row["source_ids"] or []),
        run_id=row["run_id"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        payload=_parse_json(row["payload"]),
        lineage=_parse_json(row["lineage"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_export_bundle(row: Any) -> ExportBundle:
    return ExportBundle(
        bundle_id=row["bundle_id"],
        manifest_id=row["manifest_id"],
        lane=row["lane"],
        contract_version=row["contract_version"],
        format=row["format"],
        object_count=row["object_count"],
        size_bytes=row["size_bytes"],
        checksum=row["checksum"],
        exported_at=row["exported_at"],
        exported_by=row["exported_by"],
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
    )


class PublishRepository:
    """CRUD operations for intelligence layer publish tables."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # -- Lane runs ---------------------------------------------------------

    async def create_lane_run(self, run: LaneRun) -> LaneRun:
        """Insert a new lane run."""
        row = await self._db.fetchrow(
            """
            INSERT INTO news_intel.lane_runs (
                run_id, lane, status, contract_version,
                started_at, completed_at, error_message,
                config_snapshot, metrics, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
            """,
            run.run_id,
            run.lane,
            run.status,
            run.contract_version,
            run.started_at,
            run.completed_at,
            run.error_message,
            json.dumps(run.config_snapshot),
            json.dumps(run.metrics),
            json.dumps(run.metadata),
        )
        return _row_to_lane_run(row)

    async def get_lane_run(self, run_id: str) -> LaneRun | None:
        """Fetch a lane run by ID."""
        row = await self._db.fetchrow(
            "SELECT * FROM news_intel.lane_runs WHERE run_id = $1",
            run_id,
        )
        return _row_to_lane_run(row) if row else None

    async def update_lane_run_status(
        self,
        run_id: str,
        status: str,
        *,
        error_message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> LaneRun | None:
        """Update a lane run's status and optional fields."""
        row = await self._db.fetchrow(
            """
            UPDATE news_intel.lane_runs
            SET status = $2,
                started_at = CASE WHEN $2 = 'running' AND started_at IS NULL
                                  THEN NOW() ELSE started_at END,
                completed_at = CASE WHEN $2 IN ('completed', 'failed', 'cancelled')
                                    THEN NOW() ELSE completed_at END,
                error_message = COALESCE($3, error_message),
                metrics = COALESCE($4, metrics)
            WHERE run_id = $1
            RETURNING *
            """,
            run_id,
            status,
            error_message,
            json.dumps(metrics) if metrics is not None else None,
        )
        return _row_to_lane_run(row) if row else None

    async def list_lane_runs(
        self,
        lane: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[LaneRun]:
        """List lane runs with optional filters."""
        conditions = []
        params: list[Any] = []
        if lane is not None:
            params.append(lane)
            conditions.append(f"lane = ${len(params)}")
        if status is not None:
            params.append(status)
            conditions.append(f"status = ${len(params)}")
        params.append(limit)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT * FROM news_intel.lane_runs
            {where}
            ORDER BY created_at DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_lane_run(row) for row in rows]

    # -- Manifests ---------------------------------------------------------

    async def create_manifest(self, manifest: Manifest) -> Manifest:
        """Insert a new manifest."""
        row = await self._db.fetchrow(
            """
            INSERT INTO intel_pub.manifests (
                manifest_id, lane, run_id, contract_version,
                published_at, object_count, checksum, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
            """,
            manifest.manifest_id,
            manifest.lane,
            manifest.run_id,
            manifest.contract_version,
            manifest.published_at,
            manifest.object_count,
            manifest.checksum,
            json.dumps(manifest.metadata),
        )
        return _row_to_manifest(row)

    async def get_manifest(self, manifest_id: str) -> Manifest | None:
        """Fetch a manifest by ID."""
        row = await self._db.fetchrow(
            "SELECT * FROM intel_pub.manifests WHERE manifest_id = $1",
            manifest_id,
        )
        return _row_to_manifest(row) if row else None

    async def update_manifest(
        self,
        manifest_id: str,
        *,
        object_count: int | None = None,
        checksum: str | None = None,
        published_at: datetime | None = None,
    ) -> Manifest | None:
        """Update a manifest's object count, checksum, and/or published_at."""
        row = await self._db.fetchrow(
            """
            UPDATE intel_pub.manifests
            SET object_count = COALESCE($2, object_count),
                checksum = COALESCE($3, checksum),
                published_at = COALESCE($4, published_at)
            WHERE manifest_id = $1
            RETURNING *
            """,
            manifest_id,
            object_count,
            checksum,
            published_at,
        )
        return _row_to_manifest(row) if row else None

    # -- Manifest pointers -------------------------------------------------

    async def get_pointer(self, lane: str) -> ManifestPointer | None:
        """Get the current manifest pointer for a lane."""
        row = await self._db.fetchrow(
            "SELECT * FROM intel_pub.manifest_pointers WHERE lane = $1",
            lane,
        )
        return _row_to_manifest_pointer(row) if row else None

    async def advance_pointer(
        self,
        lane: str,
        manifest_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ManifestPointer:
        """Atomically advance the manifest pointer for a lane.

        Uses INSERT ... ON CONFLICT to handle both initial pointer
        creation and subsequent advances in a single statement.
        """
        row = await self._db.fetchrow(
            """
            INSERT INTO intel_pub.manifest_pointers (
                lane, manifest_id, activated_at, previous_manifest_id, metadata
            )
            VALUES ($1, $2, NOW(), NULL, $3)
            ON CONFLICT (lane) DO UPDATE
            SET previous_manifest_id = intel_pub.manifest_pointers.manifest_id,
                manifest_id = $2,
                activated_at = NOW(),
                metadata = $3
            RETURNING *
            """,
            lane,
            manifest_id,
            json.dumps(metadata or {}),
        )
        return _row_to_manifest_pointer(row)

    # -- Published objects -------------------------------------------------

    async def create_published_object(self, obj: PublishedObject) -> PublishedObject:
        """Insert a new published object."""
        row = await self._db.fetchrow(
            """
            INSERT INTO intel_pub.published_objects (
                object_id, object_type, manifest_id, lane,
                publish_state, contract_version,
                source_ids, run_id, valid_from, valid_to,
                payload, lineage
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING *
            """,
            obj.object_id,
            obj.object_type,
            obj.manifest_id,
            obj.lane,
            obj.publish_state,
            obj.contract_version,
            obj.source_ids,
            obj.run_id,
            obj.valid_from,
            obj.valid_to,
            json.dumps(obj.payload),
            json.dumps(obj.lineage),
        )
        return _row_to_published_object(row)

    async def get_published_object(self, object_id: str) -> PublishedObject | None:
        """Fetch a published object by ID."""
        row = await self._db.fetchrow(
            "SELECT * FROM intel_pub.published_objects WHERE object_id = $1",
            object_id,
        )
        return _row_to_published_object(row) if row else None

    async def update_publish_state(self, object_id: str, new_state: str) -> PublishedObject | None:
        """Update the publish state of an object."""
        row = await self._db.fetchrow(
            """
            UPDATE intel_pub.published_objects
            SET publish_state = $2
            WHERE object_id = $1
            RETURNING *
            """,
            object_id,
            new_state,
        )
        return _row_to_published_object(row) if row else None

    async def list_objects_by_manifest(
        self,
        manifest_id: str,
        *,
        publish_state: str | None = None,
    ) -> list[PublishedObject]:
        """List published objects within a manifest."""
        if publish_state is not None:
            rows = await self._db.fetch(
                """
                SELECT * FROM intel_pub.published_objects
                WHERE manifest_id = $1 AND publish_state = $2
                ORDER BY created_at
                """,
                manifest_id,
                publish_state,
            )
        else:
            rows = await self._db.fetch(
                """
                SELECT * FROM intel_pub.published_objects
                WHERE manifest_id = $1
                ORDER BY created_at
                """,
                manifest_id,
            )
        return [_row_to_published_object(row) for row in rows]

    # -- Export bundles -----------------------------------------------------

    async def create_export_bundle(self, bundle: ExportBundle) -> ExportBundle:
        """Insert a new export bundle record."""
        row = await self._db.fetchrow(
            """
            INSERT INTO intel_export.export_bundles (
                bundle_id, manifest_id, lane, contract_version,
                format, object_count, size_bytes, checksum,
                exported_at, exported_by, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING *
            """,
            bundle.bundle_id,
            bundle.manifest_id,
            bundle.lane,
            bundle.contract_version,
            bundle.format,
            bundle.object_count,
            bundle.size_bytes,
            bundle.checksum,
            bundle.exported_at,
            bundle.exported_by,
            json.dumps(bundle.metadata),
        )
        return _row_to_export_bundle(row)

    async def get_export_bundle(self, bundle_id: str) -> ExportBundle | None:
        """Fetch an export bundle by ID."""
        row = await self._db.fetchrow(
            "SELECT * FROM intel_export.export_bundles WHERE bundle_id = $1",
            bundle_id,
        )
        return _row_to_export_bundle(row) if row else None
