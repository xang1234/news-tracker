"""Sources service with caching and seed support."""

import json
import logging
import time
from pathlib import Path

from src.sources.config import SourcesConfig
from src.sources.repository import SourcesRepository
from src.sources.schemas import Source
from src.storage.database import Database

logger = logging.getLogger(__name__)

_SEED_FILE = Path(__file__).parent / "data" / "seed_sources.json"


def _parse_seed_entry(entry: dict) -> Source:
    """Convert a JSON seed entry to a Source dataclass."""
    return Source(
        platform=entry["platform"],
        identifier=entry["identifier"],
        display_name=entry.get("display_name", ""),
        description=entry.get("description", ""),
        is_active=entry.get("is_active", True),
        metadata=entry.get("metadata", {}),
    )


class SourcesService:
    """Cached access to sources with seed support.

    Wraps SourcesRepository with TTL-based in-memory caching
    so that hot-path lookups (get_twitter_sources, etc.) avoid
    DB round-trips on every call.
    """

    def __init__(
        self,
        database: Database,
        config: SourcesConfig | None = None,
    ) -> None:
        self._config = config or SourcesConfig()
        self._repo = SourcesRepository(database)

        # Per-platform TTL cache state
        self._twitter_cache: list[str] | None = None
        self._twitter_cached_at: float = 0.0
        self._reddit_cache: list[str] | None = None
        self._reddit_cached_at: float = 0.0
        self._substack_cache: list[tuple[str, str, str]] | None = None
        self._substack_cached_at: float = 0.0

    @property
    def repository(self) -> SourcesRepository:
        """Access the underlying repository for direct DB operations."""
        return self._repo

    # ── Cached accessors ────────────────────────────────────────

    async def get_twitter_sources(self) -> list[str]:
        """Get active Twitter identifiers (cached)."""
        now = time.monotonic()
        ttl = self._config.cache_ttl_seconds
        if self._twitter_cache is not None and (now - self._twitter_cached_at) < ttl:
            return self._twitter_cache

        sources = await self._repo.get_active_by_platform("twitter")
        identifiers = [s.identifier for s in sources]
        self._twitter_cache = identifiers
        self._twitter_cached_at = now
        return identifiers

    async def get_reddit_sources(self) -> list[str]:
        """Get active Reddit subreddit names (cached)."""
        now = time.monotonic()
        ttl = self._config.cache_ttl_seconds
        if self._reddit_cache is not None and (now - self._reddit_cached_at) < ttl:
            return self._reddit_cache

        sources = await self._repo.get_active_by_platform("reddit")
        identifiers = [s.identifier for s in sources]
        self._reddit_cache = identifiers
        self._reddit_cached_at = now
        return identifiers

    async def get_substack_sources(self) -> list[tuple[str, str, str]]:
        """Get active Substack publications as (slug, display_name, description) tuples (cached)."""
        now = time.monotonic()
        ttl = self._config.cache_ttl_seconds
        if self._substack_cache is not None and (now - self._substack_cached_at) < ttl:
            return self._substack_cache

        sources = await self._repo.get_active_by_platform("substack")
        tuples = [(s.identifier, s.display_name, s.description) for s in sources]
        self._substack_cache = tuples
        self._substack_cached_at = now
        return tuples

    def invalidate_cache(self) -> None:
        """Force-clear all caches so next access hits the DB."""
        self._twitter_cache = None
        self._reddit_cache = None
        self._substack_cache = None
        self._twitter_cached_at = 0.0
        self._reddit_cached_at = 0.0
        self._substack_cached_at = 0.0

    # ── Seed ────────────────────────────────────────────────────

    async def seed_from_json(self, path: Path | None = None) -> int:
        """Load sources from a JSON file into the database.

        Returns the number of sources upserted.
        """
        seed_path = path or _SEED_FILE
        with open(seed_path) as f:
            entries = json.load(f)

        sources = [_parse_seed_entry(e) for e in entries]
        count = await self._repo.bulk_upsert(sources)
        self.invalidate_cache()
        logger.info("Seeded %d sources from %s", count, seed_path)
        return count

    async def ensure_seeded(self) -> None:
        """Seed from default JSON if the table is empty and seed_on_init is True."""
        if not self._config.seed_on_init:
            return

        existing = await self._repo.count()
        if existing > 0:
            logger.debug("Sources table has %d rows, skipping seed", existing)
            return

        logger.info("Sources table empty, seeding from default JSON")
        await self.seed_from_json()
