"""Model version tracking for reproducible backtesting.

Captures snapshots of embedding and clustering configuration so that
backtest results can be tied to the exact model parameters used.
Version IDs are deterministic: mv_{sha256(config_json)[:12]}.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.storage.database import Database

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """A snapshot of model configuration at a point in time."""

    version_id: str
    embedding_model: str
    clustering_config: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    description: str | None = None


def compute_version_id(config: dict[str, Any]) -> str:
    """Generate a deterministic version ID from config.

    Args:
        config: Configuration dictionary to hash.

    Returns:
        Version ID in format mv_{sha256[:12]}.
    """
    config_json = json.dumps(config, sort_keys=True, default=str)
    digest = hashlib.sha256(config_json.encode()).hexdigest()[:12]
    return f"mv_{digest}"


class ModelVersionRepository:
    """CRUD operations for model version records."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create(self, version: ModelVersion) -> ModelVersion:
        """Insert or update a model version (idempotent upsert).

        Args:
            version: ModelVersion to persist.

        Returns:
            The persisted ModelVersion with DB timestamps.
        """
        sql = """
            INSERT INTO model_versions (
                version_id, embedding_model, clustering_config,
                config_snapshot, description
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (version_id) DO UPDATE SET
                embedding_model = EXCLUDED.embedding_model,
                clustering_config = EXCLUDED.clustering_config,
                config_snapshot = EXCLUDED.config_snapshot,
                description = EXCLUDED.description
            RETURNING *
        """
        row = await self._db.fetchrow(
            sql,
            version.version_id,
            version.embedding_model,
            json.dumps(version.clustering_config),
            json.dumps(version.config_snapshot),
            version.description,
        )
        return _row_to_model_version(row)

    async def get_by_id(self, version_id: str) -> ModelVersion | None:
        """Get a model version by ID.

        Args:
            version_id: Version identifier.

        Returns:
            ModelVersion or None if not found.
        """
        sql = "SELECT * FROM model_versions WHERE version_id = $1"
        row = await self._db.fetchrow(sql, version_id)
        if row is None:
            return None
        return _row_to_model_version(row)

    async def get_latest(self) -> ModelVersion | None:
        """Get the most recently created model version.

        Returns:
            Latest ModelVersion or None if no versions exist.
        """
        sql = "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT 1"
        row = await self._db.fetchrow(sql)
        if row is None:
            return None
        return _row_to_model_version(row)

    async def list_versions(self, limit: int = 50) -> list[ModelVersion]:
        """List model versions ordered by creation time descending.

        Args:
            limit: Maximum versions to return.

        Returns:
            List of ModelVersion records.
        """
        sql = """
            SELECT * FROM model_versions
            ORDER BY created_at DESC
            LIMIT $1
        """
        rows = await self._db.fetch(sql, limit)
        return [_row_to_model_version(row) for row in rows]


def create_version_from_settings(
    embedding_model: str,
    clustering_config: dict[str, Any],
    full_settings: dict[str, Any] | None = None,
    description: str | None = None,
) -> ModelVersion:
    """Factory: create a ModelVersion from current settings.

    Builds a deterministic version_id from the combined embedding +
    clustering config, following the theme ID pattern (hash-based).

    Args:
        embedding_model: Name of the embedding model.
        clustering_config: Clustering algorithm parameters.
        full_settings: Optional full settings snapshot.
        description: Optional human-readable description.

    Returns:
        ModelVersion with a deterministic version_id.
    """
    config_for_hash = {
        "embedding_model": embedding_model,
        "clustering_config": clustering_config,
    }
    version_id = compute_version_id(config_for_hash)

    return ModelVersion(
        version_id=version_id,
        embedding_model=embedding_model,
        clustering_config=clustering_config,
        config_snapshot=full_settings or {},
        description=description,
    )


def _row_to_model_version(row: Any) -> ModelVersion:
    """Convert an asyncpg Record to a ModelVersion."""
    clustering_config = row.get("clustering_config", {})
    if isinstance(clustering_config, str):
        clustering_config = json.loads(clustering_config)

    config_snapshot = row.get("config_snapshot", {})
    if isinstance(config_snapshot, str):
        config_snapshot = json.loads(config_snapshot)

    return ModelVersion(
        version_id=row["version_id"],
        embedding_model=row["embedding_model"],
        clustering_config=clustering_config,
        config_snapshot=config_snapshot,
        created_at=row["created_at"],
        description=row.get("description"),
    )
