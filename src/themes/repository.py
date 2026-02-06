"""Theme repository for CRUD operations on the themes table.

Follows the DocumentRepository pattern with asyncpg, providing
high-level operations for Theme persistence with pgvector centroid
storage and JSONB field handling.
"""

import json
import logging
from typing import Any

import numpy as np

from src.storage.database import Database
from src.themes.schemas import VALID_LIFECYCLE_STAGES, Theme

logger = logging.getLogger(__name__)

# Fields allowed in generic update(). Excludes centroid (use update_centroid),
# theme_id (immutable), and created_at/updated_at (DB-managed).
_UPDATABLE_FIELDS = frozenset({
    "name",
    "description",
    "top_keywords",
    "top_tickers",
    "top_entities",
    "document_count",
    "lifecycle_stage",
    "metadata",
})


class ThemeRepository:
    """
    Repository for theme storage and retrieval.

    Provides CRUD operations for Theme records backed by the themes
    PostgreSQL table with pgvector centroid embeddings and JSONB fields.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    # ── Create ──────────────────────────────────────────────

    async def create(self, theme: Theme) -> Theme:
        """
        Insert a new theme.

        Args:
            theme: Theme to persist.

        Returns:
            The created Theme with DB-assigned timestamps.
        """
        sql = """
            INSERT INTO themes (
                theme_id, name, description, centroid,
                top_keywords, top_tickers, top_entities,
                document_count, lifecycle_stage, metadata
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7,
                $8, $9, $10
            )
            RETURNING *
        """
        row = await self._db.fetchrow(
            sql,
            theme.theme_id,
            theme.name,
            theme.description,
            _centroid_to_pgvector(theme.centroid),
            theme.top_keywords,
            theme.top_tickers,
            json.dumps(theme.top_entities),
            theme.document_count,
            theme.lifecycle_stage,
            json.dumps(theme.metadata),
        )
        return _row_to_theme(row)

    # ── Read ────────────────────────────────────────────────

    async def get_by_id(self, theme_id: str) -> Theme | None:
        """
        Get a theme by ID.

        Args:
            theme_id: Theme identifier.

        Returns:
            Theme or None if not found.
        """
        sql = "SELECT * FROM themes WHERE theme_id = $1"
        row = await self._db.fetchrow(sql, theme_id)
        if row is None:
            return None
        return _row_to_theme(row)

    async def get_all(
        self,
        lifecycle_stages: list[str] | None = None,
        limit: int = 100,
    ) -> list[Theme]:
        """
        Get all themes, optionally filtered by lifecycle stage.

        Args:
            lifecycle_stages: Optional list of stages to filter by.
            limit: Maximum number of themes to return.

        Returns:
            List of themes ordered by updated_at descending.
        """
        if lifecycle_stages:
            sql = """
                SELECT * FROM themes
                WHERE lifecycle_stage = ANY($1)
                ORDER BY updated_at DESC
                LIMIT $2
            """
            rows = await self._db.fetch(sql, lifecycle_stages, limit)
        else:
            sql = """
                SELECT * FROM themes
                ORDER BY updated_at DESC
                LIMIT $1
            """
            rows = await self._db.fetch(sql, limit)

        return [_row_to_theme(row) for row in rows]

    # ── Update ──────────────────────────────────────────────

    async def update(
        self,
        theme_id: str,
        updates: dict[str, Any],
    ) -> Theme:
        """
        Update specific fields of a theme.

        Args:
            theme_id: Theme to update.
            updates: Dict mapping field names to new values.

        Returns:
            The updated Theme.

        Raises:
            ValueError: If updates is empty, contains invalid fields,
                        has a bad lifecycle_stage, or theme not found.
        """
        if not updates:
            raise ValueError("No updates provided")

        invalid = set(updates) - _UPDATABLE_FIELDS
        if invalid:
            raise ValueError(
                f"Invalid fields for update: {sorted(invalid)}. "
                f"Allowed: {sorted(_UPDATABLE_FIELDS)}"
            )

        if "lifecycle_stage" in updates:
            stage = updates["lifecycle_stage"]
            if stage not in VALID_LIFECYCLE_STAGES:
                raise ValueError(
                    f"Invalid lifecycle_stage {stage!r}. "
                    f"Must be one of: {sorted(VALID_LIFECYCLE_STAGES)}"
                )

        # Build dynamic SET clause
        set_parts: list[str] = []
        params: list[Any] = []
        idx = 1

        for field_name, value in updates.items():
            # JSONB fields need json.dumps
            if field_name in ("top_entities", "metadata"):
                params.append(json.dumps(value))
            else:
                params.append(value)
            set_parts.append(f"{field_name} = ${idx}")
            idx += 1

        # theme_id is the last param
        params.append(theme_id)
        where_param = f"${idx}"

        sql = f"""
            UPDATE themes
            SET {', '.join(set_parts)}
            WHERE theme_id = {where_param}
            RETURNING *
        """
        row = await self._db.fetchrow(sql, *params)

        if row is None:
            raise ValueError(f"Theme {theme_id!r} not found")

        return _row_to_theme(row)

    async def update_centroid(
        self,
        theme_id: str,
        centroid: np.ndarray,
    ) -> None:
        """
        Update a theme's centroid embedding (fast path).

        Dedicated method for the performance-critical centroid update
        during incremental transform() — fixed SQL, no JSONB overhead.

        Args:
            theme_id: Theme to update.
            centroid: New 768-dim centroid vector.

        Raises:
            ValueError: If theme not found.
        """
        sql = """
            UPDATE themes
            SET centroid = $2
            WHERE theme_id = $1
        """
        result = await self._db.execute(sql, theme_id, _centroid_to_pgvector(centroid))

        # asyncpg execute returns status string like "UPDATE 1" or "UPDATE 0"
        if result.endswith(" 0"):
            raise ValueError(f"Theme {theme_id!r} not found")

    # ── Delete ──────────────────────────────────────────────

    async def delete(self, theme_id: str) -> bool:
        """
        Delete a theme.

        Args:
            theme_id: Theme to delete.

        Returns:
            True if deleted, False if not found.
        """
        sql = "DELETE FROM themes WHERE theme_id = $1 RETURNING theme_id"
        result = await self._db.fetchval(sql, theme_id)
        return result is not None


# ── Helpers (module-level for testability) ──────────────────


def _parse_centroid(value: Any) -> np.ndarray:
    """Parse a pgvector string or list into a numpy array.

    Args:
        value: pgvector string like "[0.1,0.2,...]" or a list of floats.

    Returns:
        Float32 numpy array.

    Raises:
        ValueError: If value is an unexpected type.
    """
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
    if isinstance(value, str):
        return np.array(
            [float(x) for x in value.strip("[]").split(",")],
            dtype=np.float32,
        )
    raise ValueError(f"Cannot parse centroid from {type(value).__name__}")


def _centroid_to_pgvector(centroid: np.ndarray) -> str:
    """Convert a numpy array to pgvector string format.

    Args:
        centroid: Numpy array of floats.

    Returns:
        String like "[0.1,0.2,...]".
    """
    return f"[{','.join(str(float(x)) for x in centroid)}]"


def _row_to_theme(row: Any) -> Theme:
    """Convert an asyncpg Record to a Theme.

    Handles pgvector string → ndarray, JSONB string → dict/list,
    and TEXT[] → list passthrough.
    """
    centroid = _parse_centroid(row["centroid"])

    # JSONB fields may come back as strings or already-parsed
    top_entities = row.get("top_entities", [])
    if isinstance(top_entities, str):
        top_entities = json.loads(top_entities)

    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return Theme(
        theme_id=row["theme_id"],
        name=row["name"],
        centroid=centroid,
        top_keywords=list(row.get("top_keywords", [])),
        top_tickers=list(row.get("top_tickers", [])),
        lifecycle_stage=row.get("lifecycle_stage", "emerging"),
        document_count=row.get("document_count", 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        description=row.get("description"),
        top_entities=top_entities,
        metadata=metadata,
    )
