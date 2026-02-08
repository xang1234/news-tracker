"""Authority repository for CRUD operations on authority profiles.

Follows the FeedbackRepository pattern with asyncpg, providing
upsert, retrieval, and batch operations for AuthorityProfile records.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.authority.schemas import AuthorityProfile
from src.storage.database import Database

logger = logging.getLogger(__name__)


class AuthorityRepository:
    """Repository for authority profile persistence and querying."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def get(self, author_id: str, platform: str) -> AuthorityProfile | None:
        """Retrieve an authority profile by composite key.

        Args:
            author_id: Author identifier.
            platform: Source platform.

        Returns:
            AuthorityProfile if found, None otherwise.
        """
        sql = """
            SELECT * FROM authority_profiles
            WHERE author_id = $1 AND platform = $2
        """
        row = await self._db.fetchrow(sql, author_id, platform)
        if row is None:
            return None
        return _row_to_profile(row)

    async def upsert(self, profile: AuthorityProfile) -> AuthorityProfile:
        """Insert or update an authority profile.

        Uses ON CONFLICT to handle idempotent upserts on the
        (author_id, platform) composite primary key.

        Args:
            profile: Profile to persist.

        Returns:
            The persisted profile with DB-assigned defaults.
        """
        sql = """
            INSERT INTO authority_profiles (
                author_id, platform, tier, base_weight,
                total_calls, correct_calls, first_seen,
                last_good_call, topic_scores, centrality_score,
                updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11)
            ON CONFLICT (author_id, platform) DO UPDATE SET
                tier = EXCLUDED.tier,
                base_weight = EXCLUDED.base_weight,
                total_calls = EXCLUDED.total_calls,
                correct_calls = EXCLUDED.correct_calls,
                last_good_call = EXCLUDED.last_good_call,
                topic_scores = EXCLUDED.topic_scores,
                centrality_score = EXCLUDED.centrality_score,
                updated_at = EXCLUDED.updated_at
            RETURNING *
        """
        row = await self._db.fetchrow(
            sql,
            profile.author_id,
            profile.platform,
            profile.tier,
            profile.base_weight,
            profile.total_calls,
            profile.correct_calls,
            profile.first_seen,
            profile.last_good_call,
            json.dumps(profile.topic_scores),
            profile.centrality_score,
            profile.updated_at,
        )
        return _row_to_profile(row)

    async def list_by_platform(
        self,
        platform: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuthorityProfile]:
        """List authority profiles for a platform.

        Args:
            platform: Platform to filter by.
            limit: Max records.
            offset: Pagination offset.

        Returns:
            List of profiles ordered by updated_at descending.
        """
        sql = """
            SELECT * FROM authority_profiles
            WHERE platform = $1
            ORDER BY updated_at DESC
            LIMIT $2 OFFSET $3
        """
        rows = await self._db.fetch(sql, platform, limit, offset)
        return [_row_to_profile(row) for row in rows]

    async def get_batch(
        self,
        keys: list[tuple[str, str]],
    ) -> dict[tuple[str, str], AuthorityProfile]:
        """Retrieve multiple profiles by (author_id, platform) pairs.

        Args:
            keys: List of (author_id, platform) tuples.

        Returns:
            Dict mapping (author_id, platform) to profile.
        """
        if not keys:
            return {}

        # Build VALUES list for efficient batch lookup
        author_ids = [k[0] for k in keys]
        platforms = [k[1] for k in keys]

        sql = """
            SELECT ap.* FROM authority_profiles ap
            INNER JOIN unnest($1::text[], $2::text[]) AS lookup(aid, plat)
                ON ap.author_id = lookup.aid AND ap.platform = lookup.plat
        """
        rows = await self._db.fetch(sql, author_ids, platforms)
        result = {}
        for row in rows:
            profile = _row_to_profile(row)
            result[(profile.author_id, profile.platform)] = profile
        return result

    async def count(self, platform: str | None = None) -> int:
        """Count authority profiles.

        Args:
            platform: Optional platform filter.

        Returns:
            Number of profiles.
        """
        if platform is not None:
            sql = "SELECT COUNT(*) FROM authority_profiles WHERE platform = $1"
            return await self._db.fetchval(sql, platform)
        sql = "SELECT COUNT(*) FROM authority_profiles"
        return await self._db.fetchval(sql)


def _row_to_profile(row: Any) -> AuthorityProfile:
    """Convert an asyncpg Record to an AuthorityProfile."""
    topic_scores = row["topic_scores"]
    if isinstance(topic_scores, str):
        topic_scores = json.loads(topic_scores)

    return AuthorityProfile(
        author_id=row["author_id"],
        platform=row["platform"],
        tier=row["tier"],
        base_weight=row["base_weight"],
        total_calls=row["total_calls"],
        correct_calls=row["correct_calls"],
        first_seen=row["first_seen"],
        last_good_call=row.get("last_good_call"),
        topic_scores=topic_scores or {},
        centrality_score=row["centrality_score"],
        updated_at=row["updated_at"],
    )
