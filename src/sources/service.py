"""Sources service with caching and seed support."""

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.feeds import FEEDS, Feed
from src.sources.config import SourcesConfig
from src.sources.repository import SourcesRepository
from src.sources.schemas import RssSourceHealth, Source
from src.storage.database import Database

logger = logging.getLogger(__name__)

_SEED_FILE = Path(__file__).parent / "data" / "seed_sources.json"


def _parse_seed_entry(entry: dict[str, Any]) -> Source:
    """Convert a JSON seed entry to a Source dataclass."""
    return Source(
        platform=entry["platform"],
        identifier=entry["identifier"],
        display_name=entry.get("display_name", ""),
        description=entry.get("description", ""),
        is_active=entry.get("is_active", True),
        metadata=entry.get("metadata", {}),
    )


def _feed_to_source(feed: Feed) -> Source:
    """Represent a static RSS catalog entry as a sources-table row."""
    return Source(
        platform="rss",
        identifier=feed.slug,
        display_name=feed.name,
        description=f"{feed.name} RSS/Atom feed",
        is_active=feed.enabled,
        metadata={
            "url": feed.url,
            "category": feed.category,
            "authority": feed.authority,
            "full_text": feed.full_text,
        },
    )


def _source_to_rss_feed(source: Source) -> Feed:
    """Convert an RSS source row to FeedAdapter configuration."""
    metadata = source.metadata or {}
    url = _required_metadata_str(source, metadata, "url")
    category = _required_metadata_str(source, metadata, "category")
    authority = _optional_metadata_str(metadata, "authority", default="standard")
    return Feed(
        slug=source.identifier,
        name=source.display_name or source.identifier,
        url=url,
        category=category,
        authority=authority,
        full_text=_metadata_bool(metadata.get("full_text"), default=False),
        enabled=source.is_active,
    )


def _required_metadata_str(source: Source, metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"RSS source {source.identifier} metadata.{key} must be a non-empty string"
        )
    return value.strip()


def _optional_metadata_str(metadata: dict[str, Any], key: str, *, default: str) -> str:
    value = metadata.get(key, default)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _metadata_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _metadata_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return default
    return default


def _metadata_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metadata_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _rss_operator_status(
    *,
    source: Source,
    health: dict[str, Any],
    now: datetime,
    stale_after_seconds: int,
) -> str:
    if not source.is_active:
        return "inactive"
    health_status = str(health.get("status") or "")
    if health_status in {"error", "parse_failure"}:
        return "failing"
    last_fetch_at = _parse_iso_datetime(health.get("last_fetch_at"))
    if last_fetch_at is None:
        return "never_fetched"
    if (now - last_fetch_at).total_seconds() > stale_after_seconds:
        return "stale"
    return "active"


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
        self._rss_cache: list[Feed] | None = None
        self._rss_cached_at: float = 0.0

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

    async def get_rss_feeds(self) -> list[Feed]:
        """Get active RSS/Atom feed records as FeedAdapter configs (cached)."""
        now = time.monotonic()
        ttl = self._config.cache_ttl_seconds
        if self._rss_cache is not None and (now - self._rss_cached_at) < ttl:
            return self._rss_cache

        sources = await self._repo.get_active_by_platform("rss")
        feeds = [_source_to_rss_feed(source) for source in sources]
        self._rss_cache = feeds
        self._rss_cached_at = now
        return feeds

    async def get_rss_source_health(
        self,
        *,
        stale_after_seconds: int = 24 * 60 * 60,
        now: datetime | None = None,
    ) -> list[RssSourceHealth]:
        """Summarize RSS source health for operator surfaces."""
        current_time = now or datetime.now(UTC)
        sources, _ = await self._repo.list_sources(platform="rss", limit=500)
        summaries: list[RssSourceHealth] = []
        for source in sources:
            metadata = source.metadata or {}
            health = _metadata_dict(metadata.get("health"))
            recent_document_count = _metadata_int(health.get("recent_document_count"))
            last_fetch_at = _metadata_str(health.get("last_fetch_at"))
            last_success_at = _metadata_str(health.get("last_success_at"))
            last_error_at = _metadata_str(health.get("last_error_at"))
            summaries.append(
                RssSourceHealth(
                    slug=source.identifier,
                    name=source.display_name or source.identifier,
                    url=str(metadata.get("url") or ""),
                    category=str(metadata.get("category") or ""),
                    is_active=source.is_active,
                    status=_rss_operator_status(
                        source=source,
                        health=health,
                        now=current_time,
                        stale_after_seconds=stale_after_seconds,
                    ),
                    is_producing=recent_document_count > 0,
                    recent_document_count=recent_document_count,
                    last_fetch_at=last_fetch_at,
                    last_success_at=last_success_at,
                    last_error_at=last_error_at,
                    last_error=str(health.get("last_error") or health.get("error") or ""),
                    health_status=str(health.get("status") or ""),
                )
            )
        return summaries

    async def record_rss_feed_health(self, feed_slug: str, health: dict[str, object]) -> bool:
        """Persist the latest RSS fetch health snapshot into source metadata."""
        updated = await self._repo.patch_metadata("rss", feed_slug, {"health": health})
        if updated:
            self.invalidate_cache()
        return updated

    def invalidate_cache(self) -> None:
        """Force-clear all caches so next access hits the DB."""
        self._twitter_cache = None
        self._reddit_cache = None
        self._substack_cache = None
        self._rss_cache = None
        self._twitter_cached_at = 0.0
        self._reddit_cached_at = 0.0
        self._substack_cached_at = 0.0
        self._rss_cached_at = 0.0

    # ── Seed ────────────────────────────────────────────────────

    async def seed_from_json(self, path: Path | None = None) -> int:
        """Load sources from a JSON file into the database.

        The default seed also materializes the static RSS catalog as
        platform="rss" rows so operators can manage feeds through Sources.
        Returns the number of sources upserted.
        """
        seed_path = path or _SEED_FILE
        with open(seed_path) as f:
            entries = json.load(f)

        sources = [_parse_seed_entry(e) for e in entries]
        if path is None:
            sources.extend(_feed_to_source(feed) for feed in FEEDS)
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
