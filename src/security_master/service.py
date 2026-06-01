"""Security master service with caching and seed support."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.security_master.config import SecurityMasterConfig
from src.security_master.nasdaq_trader import (
    NASDAQ_TRADER_EXTERNAL_KEY,
    SECURITY_MASTER_US_EXCHANGE,
    NasdaqTraderReconciliationResult,
    NasdaqTraderSymbolDirectory,
    build_nasdaq_trader_reconciliation,
    fetch_nasdaq_trader_symbol_directories,
    parse_nasdaq_trader_symbol_directories,
)
from src.security_master.repository import SecurityMasterRepository
from src.security_master.schemas import Security, SecurityIdentifierLineage, normalize_sec_cik
from src.storage.database import Database

logger = logging.getLogger(__name__)

_SEED_FILE = Path(__file__).parent / "data" / "seed_securities.json"


def _parse_seed_entry(entry: dict[str, Any]) -> Security:
    """Convert a JSON seed entry to a Security dataclass."""
    sec_cik = normalize_sec_cik(entry.get("sec_cik"))
    external_identifiers = dict(entry.get("external_identifiers", {}))
    identifier_lineage = []
    for record in entry.get("identifier_lineage", []):
        normalized_record = dict(record)
        if (
            normalized_record.get("identifier_type") == "sec_cik"
            and normalized_record.get("value") is not None
        ):
            normalized_record["value"] = normalize_sec_cik(normalized_record["value"]) or ""
        identifier_lineage.append(normalized_record)
    if sec_cik:
        external_identifiers.setdefault("sec_ticker", entry["ticker"])
        if not any(record.get("identifier_type") == "sec_cik" for record in identifier_lineage):
            identifier_lineage.append(
                {
                    "identifier_type": "sec_cik",
                    "value": sec_cik,
                    "source": "sec_company_tickers",
                }
            )
    return Security(
        ticker=entry["ticker"],
        exchange=entry.get("exchange", "US"),
        name=entry.get("name", ""),
        aliases=entry.get("aliases", []),
        sector=entry.get("sector", ""),
        country=entry.get("country", "US"),
        currency=entry.get("currency", "USD"),
        figi=entry.get("figi"),
        sec_cik=sec_cik,
        issuer_name=entry.get("issuer_name", ""),
        former_names=entry.get("former_names", []),
        external_identifiers=external_identifiers,
        identifier_lineage=[
            SecurityIdentifierLineage.from_raw(record) for record in identifier_lineage
        ],
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

    async def fuzzy_search(self, query: str, limit: int = 10) -> list[Security]:
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

    async def ingest_nasdaq_trader_symbol_directory(
        self,
        nasdaq_listed_text: str,
        other_listed_text: str,
        *,
        observed_at: datetime | None = None,
    ) -> NasdaqTraderReconciliationResult:
        """Parse and reconcile Nasdaq Trader symbol-directory files."""
        directory = parse_nasdaq_trader_symbol_directories(
            nasdaq_listed_text,
            other_listed_text,
        )
        result = await self._reconcile_nasdaq_trader_directory(directory, observed_at=observed_at)
        logger.info(
            "Ingested Nasdaq Trader symbol directory",
            extra={
                "current_record_count": result.current_record_count,
                "deactivated_missing_count": result.deactivated_missing_count,
            },
        )
        return result

    async def refresh_nasdaq_trader_symbol_directory(self) -> NasdaqTraderReconciliationResult:
        """Fetch official Nasdaq Trader symbol files and reconcile them."""
        directory = await fetch_nasdaq_trader_symbol_directories()
        return await self._reconcile_nasdaq_trader_directory(directory)

    async def _reconcile_nasdaq_trader_directory(
        self,
        directory: NasdaqTraderSymbolDirectory,
        *,
        observed_at: datetime | None = None,
    ) -> NasdaqTraderReconciliationResult:
        current_keys = {
            (record.symbol, SECURITY_MASTER_US_EXCHANGE) for record in directory.records
        }
        existing_by_key = await self._repo.get_by_keys(current_keys)
        previously_sourced = await self._repo.list_by_external_identifier(
            NASDAQ_TRADER_EXTERNAL_KEY,
        )
        result = build_nasdaq_trader_reconciliation(
            directory,
            existing_by_key=existing_by_key,
            previously_sourced_by_key={
                (security.ticker, security.exchange): security for security in previously_sourced
            },
            observed_at=observed_at,
        )
        await self._repo.bulk_upsert(list(result.securities))
        self.invalidate_cache()
        return result

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
