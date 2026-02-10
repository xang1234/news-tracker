"""Database repository for the sources table."""

import logging

from src.sources.schemas import Source
from src.storage.database import Database

logger = logging.getLogger(__name__)

# SQL for table creation (mirrors migration 014)
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    platform     TEXT NOT NULL,
    identifier   TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    description  TEXT NOT NULL DEFAULT '',
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (platform, identifier)
);

CREATE INDEX IF NOT EXISTS idx_sources_platform
    ON sources(platform);
CREATE INDEX IF NOT EXISTS idx_sources_platform_active
    ON sources(platform, is_active) WHERE is_active = TRUE;
"""

_UPSERT_SQL = """
INSERT INTO sources (platform, identifier, display_name, description, is_active, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (platform, identifier) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    is_active = EXCLUDED.is_active,
    metadata = EXCLUDED.metadata,
    updated_at = NOW()
RETURNING platform, identifier
"""

_BULK_UPSERT_SQL = """
INSERT INTO sources (platform, identifier, display_name, description, is_active, metadata)
SELECT * FROM unnest(
    $1::text[], $2::text[], $3::text[], $4::text[], $5::boolean[], $6::jsonb[]
)
ON CONFLICT (platform, identifier) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    is_active = EXCLUDED.is_active,
    metadata = EXCLUDED.metadata,
    updated_at = NOW()
"""


def _record_to_source(record) -> Source:
    """Convert an asyncpg Record to a Source dataclass."""
    return Source(
        platform=record["platform"],
        identifier=record["identifier"],
        display_name=record["display_name"],
        description=record["description"],
        is_active=record["is_active"],
        metadata=dict(record["metadata"]) if record["metadata"] else {},
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


class SourcesRepository:
    """CRUD operations for the sources table."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create_table(self) -> None:
        """Create the sources table and indexes (idempotent)."""
        await self._db.execute(_CREATE_TABLE_SQL)
        logger.info("Sources table ensured")

    async def upsert(self, source: Source) -> None:
        """Insert or update a single source."""
        import json

        await self._db.fetch(
            _UPSERT_SQL,
            source.platform,
            source.identifier,
            source.display_name,
            source.description,
            source.is_active,
            json.dumps(source.metadata),
        )

    async def bulk_upsert(self, sources: list[Source]) -> int:
        """Insert or update multiple sources in one statement.

        Returns the number of sources processed.
        """
        if not sources:
            return 0

        import json

        platforms = [s.platform for s in sources]
        identifiers = [s.identifier for s in sources]
        display_names = [s.display_name for s in sources]
        descriptions = [s.description for s in sources]
        actives = [s.is_active for s in sources]
        metadatas = [json.dumps(s.metadata) for s in sources]

        await self._db.execute(
            _BULK_UPSERT_SQL,
            platforms, identifiers, display_names, descriptions, actives, metadatas,
        )
        logger.info("Bulk upserted %d sources", len(sources))
        return len(sources)

    async def get_by_key(
        self, platform: str, identifier: str
    ) -> Source | None:
        """Fetch a single source by platform and identifier."""
        row = await self._db.fetchrow(
            "SELECT * FROM sources WHERE platform = $1 AND identifier = $2",
            platform, identifier,
        )
        return _record_to_source(row) if row else None

    async def list_sources(
        self,
        platform: str | None = None,
        search: str | None = None,
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Source], int]:
        """Paginated list with filters. Returns (sources, total)."""
        conditions: list[str] = []
        params: list = []
        idx = 1

        if active_only:
            conditions.append("is_active = TRUE")

        if platform:
            conditions.append(f"platform = ${idx}")
            params.append(platform)
            idx += 1

        if search:
            conditions.append(
                f"(identifier ILIKE ${idx} OR display_name ILIKE ${idx})"
            )
            params.append(f"%{search}%")
            idx += 1

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM sources{where_clause}"
        total = await self._db.fetchval(count_sql, *params)

        data_sql = f"""
            SELECT * FROM sources{where_clause}
            ORDER BY platform, identifier
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        params.extend([limit, offset])
        rows = await self._db.fetch(data_sql, *params)

        return [_record_to_source(r) for r in rows], total or 0

    async def get_active_by_platform(self, platform: str) -> list[Source]:
        """Fetch all active sources for a given platform."""
        rows = await self._db.fetch(
            "SELECT * FROM sources WHERE platform = $1 AND is_active = TRUE ORDER BY identifier",
            platform,
        )
        return [_record_to_source(r) for r in rows]

    async def deactivate(self, platform: str, identifier: str) -> bool:
        """Soft-deactivate a source. Returns True if a row was updated."""
        result = await self._db.execute(
            """
            UPDATE sources SET is_active = FALSE, updated_at = NOW()
            WHERE platform = $1 AND identifier = $2 AND is_active = TRUE
            """,
            platform, identifier,
        )
        return result.endswith("1")

    async def count(self) -> int:
        """Count total sources in the table."""
        return await self._db.fetchval("SELECT COUNT(*) FROM sources")
