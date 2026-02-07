"""Security master service with caching and seed support."""

import json
import logging
import time
from pathlib import Path

from src.security_master.config import SecurityMasterConfig
from src.security_master.repository import SecurityMasterRepository
from src.security_master.schemas import Security
from src.storage.database import Database

logger = logging.getLogger(__name__)

_SEED_FILE = Path(__file__).parent / "data" / "seed_securities.json"


def _parse_seed_entry(entry: dict) -> Security:
    """Convert a JSON seed entry to a Security dataclass."""
    return Security(
        ticker=entry["ticker"],
        exchange=entry.get("exchange", "US"),
        name=entry.get("name", ""),
        aliases=entry.get("aliases", []),
        sector=entry.get("sector", ""),
        country=entry.get("country", "US"),
        currency=entry.get("currency", "USD"),
        figi=entry.get("figi"),
        is_active=entry.get("is_active", True),
    )


class SecurityMasterService:
    """Cached access to the security master with seed support.

    Wraps SecurityMasterRepository with TTL-based in-memory caching
    so that hot-path lookups (get_all_tickers, get_company_map) avoid
    DB round-trips on every call.
    """

    def __init__(
        self,
        database: Database,
        config: SecurityMasterConfig | None = None,
    ) -> None:
        self._config = config or SecurityMasterConfig()
        self._repo = SecurityMasterRepository(database)

        # TTL cache state
        self._tickers_cache: set[str] | None = None
        self._tickers_cached_at: float = 0.0
        self._company_cache: dict[str, str] | None = None
        self._company_cached_at: float = 0.0

    @property
    def repository(self) -> SecurityMasterRepository:
        """Access the underlying repository for direct DB operations."""
        return self._repo

    # ── Cached accessors ────────────────────────────────────────

    async def get_all_tickers(self) -> set[str]:
        """Get all active ticker symbols (cached)."""
        now = time.monotonic()
        ttl = self._config.cache_ttl_seconds
        if self._tickers_cache is not None and (now - self._tickers_cached_at) < ttl:
            return self._tickers_cache

        tickers = await self._repo.get_all_active_tickers()
        self._tickers_cache = tickers
        self._tickers_cached_at = now
        return tickers

    async def get_company_map(self) -> dict[str, str]:
        """Get company-name-to-ticker mapping (cached)."""
        now = time.monotonic()
        ttl = self._config.cache_ttl_seconds
        if self._company_cache is not None and (now - self._company_cached_at) < ttl:
            return self._company_cache

        mapping = await self._repo.get_company_to_ticker_map()
        self._company_cache = mapping
        self._company_cached_at = now
        return mapping

    def invalidate_cache(self) -> None:
        """Force-clear both caches so next access hits the DB."""
        self._tickers_cache = None
        self._company_cache = None
        self._tickers_cached_at = 0.0
        self._company_cached_at = 0.0

    # ── Fuzzy search ────────────────────────────────────────────

    async def fuzzy_search(
        self, query: str, limit: int = 10
    ) -> list[Security]:
        """Search securities by name using pg_trgm similarity."""
        return await self._repo.search_by_name(
            query, limit=limit, threshold=self._config.fuzzy_threshold
        )

    # ── Seed ────────────────────────────────────────────────────

    async def seed_from_json(self, path: Path | None = None) -> int:
        """Load securities from a JSON file into the database.

        Returns the number of securities upserted.
        """
        seed_path = path or _SEED_FILE
        with open(seed_path) as f:
            entries = json.load(f)

        securities = [_parse_seed_entry(e) for e in entries]
        count = await self._repo.bulk_upsert(securities)
        self.invalidate_cache()
        logger.info("Seeded %d securities from %s", count, seed_path)
        return count

    async def ensure_seeded(self) -> None:
        """Seed from default JSON if the table is empty and seed_on_init is True."""
        if not self._config.seed_on_init:
            return

        existing = await self._repo.count()
        if existing > 0:
            logger.debug("Securities table has %d rows, skipping seed", existing)
            return

        logger.info("Securities table empty, seeding from default JSON")
        await self.seed_from_json()
