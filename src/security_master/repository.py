"""Database repository for the securities table."""

import logging

from src.security_master.schemas import Security
from src.storage.database import Database

logger = logging.getLogger(__name__)

# SQL for table creation (mirrors migration 008)
_CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS securities (
    ticker      TEXT NOT NULL,
    exchange    TEXT NOT NULL DEFAULT 'US',
    name        TEXT NOT NULL DEFAULT '',
    aliases     TEXT[] NOT NULL DEFAULT '{}',
    sector      TEXT NOT NULL DEFAULT '',
    country     TEXT NOT NULL DEFAULT 'US',
    currency    TEXT NOT NULL DEFAULT 'USD',
    figi        TEXT,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, exchange)
);

CREATE INDEX IF NOT EXISTS idx_securities_ticker
    ON securities(ticker);
CREATE INDEX IF NOT EXISTS idx_securities_name_trgm
    ON securities USING GIN (name gin_trgm_ops);
CREATE UNIQUE INDEX IF NOT EXISTS idx_securities_figi
    ON securities(figi) WHERE figi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_securities_active
    ON securities(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_securities_sector
    ON securities(sector);
"""

_UPSERT_SQL = """
INSERT INTO securities (ticker, exchange, name, aliases, sector, country, currency, figi, is_active)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (ticker, exchange) DO UPDATE SET
    name = EXCLUDED.name,
    aliases = EXCLUDED.aliases,
    sector = EXCLUDED.sector,
    country = EXCLUDED.country,
    currency = EXCLUDED.currency,
    figi = EXCLUDED.figi,
    is_active = EXCLUDED.is_active,
    updated_at = NOW()
RETURNING ticker, exchange
"""

_BULK_UPSERT_SQL = """
INSERT INTO securities (ticker, exchange, name, aliases, sector, country, currency, figi, is_active)
SELECT * FROM unnest(
    $1::text[], $2::text[], $3::text[], $4::text[][], $5::text[],
    $6::text[], $7::text[], $8::text[], $9::boolean[]
)
ON CONFLICT (ticker, exchange) DO UPDATE SET
    name = EXCLUDED.name,
    aliases = EXCLUDED.aliases,
    sector = EXCLUDED.sector,
    country = EXCLUDED.country,
    currency = EXCLUDED.currency,
    figi = EXCLUDED.figi,
    is_active = EXCLUDED.is_active,
    updated_at = NOW()
"""


def _record_to_security(record) -> Security:
    """Convert an asyncpg Record to a Security dataclass."""
    return Security(
        ticker=record["ticker"],
        exchange=record["exchange"],
        name=record["name"],
        aliases=list(record["aliases"]) if record["aliases"] else [],
        sector=record["sector"],
        country=record["country"],
        currency=record["currency"],
        figi=record["figi"],
        is_active=record["is_active"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


class SecurityMasterRepository:
    """CRUD operations for the securities table."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create_table(self) -> None:
        """Create the securities table and indexes (idempotent)."""
        await self._db.execute(_CREATE_TABLE_SQL)
        logger.info("Securities table ensured")

    async def upsert(self, security: Security) -> None:
        """Insert or update a single security."""
        await self._db.fetch(
            _UPSERT_SQL,
            security.ticker,
            security.exchange,
            security.name,
            security.aliases,
            security.sector,
            security.country,
            security.currency,
            security.figi,
            security.is_active,
        )

    async def bulk_upsert(self, securities: list[Security]) -> int:
        """Insert or update multiple securities in one statement.

        Returns the number of securities processed.
        """
        if not securities:
            return 0

        tickers = [s.ticker for s in securities]
        exchanges = [s.exchange for s in securities]
        names = [s.name for s in securities]
        aliases = [s.aliases for s in securities]
        sectors = [s.sector for s in securities]
        countries = [s.country for s in securities]
        currencies = [s.currency for s in securities]
        figis = [s.figi for s in securities]
        actives = [s.is_active for s in securities]

        await self._db.execute(
            _BULK_UPSERT_SQL,
            tickers, exchanges, names, aliases, sectors,
            countries, currencies, figis, actives,
        )
        logger.info("Bulk upserted %d securities", len(securities))
        return len(securities)

    async def get_by_ticker(
        self, ticker: str, exchange: str = "US"
    ) -> Security | None:
        """Fetch a single security by ticker and exchange."""
        row = await self._db.fetchrow(
            "SELECT * FROM securities WHERE ticker = $1 AND exchange = $2",
            ticker, exchange,
        )
        return _record_to_security(row) if row else None

    async def get_all_active(self) -> list[Security]:
        """Fetch all active securities."""
        rows = await self._db.fetch(
            "SELECT * FROM securities WHERE is_active = TRUE ORDER BY ticker"
        )
        return [_record_to_security(r) for r in rows]

    async def get_all_active_tickers(self) -> set[str]:
        """Fetch just the ticker symbols for all active securities."""
        rows = await self._db.fetch(
            "SELECT ticker FROM securities WHERE is_active = TRUE"
        )
        return {r["ticker"] for r in rows}

    async def get_company_to_ticker_map(self) -> dict[str, str]:
        """Build a company-name-to-ticker lookup from active securities.

        Includes the security name and all aliases (lowercased).
        """
        rows = await self._db.fetch(
            "SELECT ticker, name, aliases FROM securities WHERE is_active = TRUE"
        )
        mapping: dict[str, str] = {}
        for row in rows:
            ticker = row["ticker"]
            if row["name"]:
                mapping[row["name"].lower()] = ticker
            for alias in row["aliases"] or []:
                mapping[alias.lower()] = ticker
        return mapping

    async def search_by_name(
        self, query: str, limit: int = 10, threshold: float = 0.3
    ) -> list[Security]:
        """Fuzzy search securities by name using pg_trgm similarity."""
        rows = await self._db.fetch(
            """
            SELECT *, similarity(name, $1) AS sim
            FROM securities
            WHERE is_active = TRUE AND similarity(name, $1) >= $2
            ORDER BY sim DESC
            LIMIT $3
            """,
            query, threshold, limit,
        )
        return [_record_to_security(r) for r in rows]

    async def deactivate(self, ticker: str, exchange: str = "US") -> bool:
        """Soft-deactivate a security. Returns True if a row was updated."""
        result = await self._db.execute(
            """
            UPDATE securities SET is_active = FALSE, updated_at = NOW()
            WHERE ticker = $1 AND exchange = $2 AND is_active = TRUE
            """,
            ticker, exchange,
        )
        return result.endswith("1")

    async def count(self) -> int:
        """Count total securities in the table."""
        return await self._db.fetchval("SELECT COUNT(*) FROM securities")
