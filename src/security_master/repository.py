"""Database repository for the securities table."""

import json
import logging
from collections.abc import Iterable
from typing import Any

from src.security_master.schemas import Security, normalize_sec_cik
from src.storage.database import Database
from src.storage.migrations import apply_migrations

logger = logging.getLogger(__name__)

_UPSERT_SQL = """
INSERT INTO securities (
    ticker, exchange, name, aliases, sector, country, currency, figi,
    sec_cik, issuer_name, former_names, external_identifiers,
    identifier_lineage, is_active
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (ticker, exchange) DO UPDATE SET
    name = EXCLUDED.name,
    aliases = EXCLUDED.aliases,
    sector = EXCLUDED.sector,
    country = EXCLUDED.country,
    currency = EXCLUDED.currency,
    figi = EXCLUDED.figi,
    sec_cik = EXCLUDED.sec_cik,
    issuer_name = EXCLUDED.issuer_name,
    former_names = EXCLUDED.former_names,
    external_identifiers = EXCLUDED.external_identifiers,
    identifier_lineage = EXCLUDED.identifier_lineage,
    is_active = EXCLUDED.is_active,
    updated_at = NOW()
RETURNING ticker, exchange
"""

_BULK_UPSERT_SQL = """
INSERT INTO securities (
    ticker, exchange, name, aliases, sector, country, currency, figi,
    sec_cik, issuer_name, former_names, external_identifiers,
    identifier_lineage, is_active
)
SELECT * FROM unnest(
    $1::text[], $2::text[], $3::text[], $4::text[][], $5::text[],
    $6::text[], $7::text[], $8::text[], $9::text[], $10::text[],
    $11::text[][], $12::jsonb[], $13::jsonb[], $14::boolean[]
)
ON CONFLICT (ticker, exchange) DO UPDATE SET
    name = EXCLUDED.name,
    aliases = EXCLUDED.aliases,
    sector = EXCLUDED.sector,
    country = EXCLUDED.country,
    currency = EXCLUDED.currency,
    figi = EXCLUDED.figi,
    sec_cik = EXCLUDED.sec_cik,
    issuer_name = EXCLUDED.issuer_name,
    former_names = EXCLUDED.former_names,
    external_identifiers = EXCLUDED.external_identifiers,
    identifier_lineage = EXCLUDED.identifier_lineage,
    is_active = EXCLUDED.is_active,
    updated_at = NOW()
"""


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    try:
        return record[key]
    except (KeyError, IndexError):
        return default


def _parse_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, str):
        return json.loads(value)
    return value


def _security_identifier_lineage_json(security: Security) -> str:
    return json.dumps([record.to_dict() for record in security.identifier_lineage])


def _security_upsert_args(security: Security) -> tuple[Any, ...]:
    return (
        security.ticker,
        security.exchange,
        security.name,
        security.aliases,
        security.sector,
        security.country,
        security.currency,
        security.figi,
        security.sec_cik,
        security.issuer_name,
        security.former_names,
        json.dumps(security.external_identifiers),
        _security_identifier_lineage_json(security),
        security.is_active,
    )


def _bulk_upsert_args(securities: list[Security]) -> tuple[list[Any], ...]:
    return (
        [security.ticker for security in securities],
        [security.exchange for security in securities],
        [security.name for security in securities],
        [security.aliases for security in securities],
        [security.sector for security in securities],
        [security.country for security in securities],
        [security.currency for security in securities],
        [security.figi for security in securities],
        [security.sec_cik for security in securities],
        [security.issuer_name for security in securities],
        [security.former_names for security in securities],
        [json.dumps(security.external_identifiers) for security in securities],
        [_security_identifier_lineage_json(security) for security in securities],
        [security.is_active for security in securities],
    )


def _record_to_security(record: Any) -> Security:
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
        sec_cik=_record_value(record, "sec_cik"),
        issuer_name=_record_value(record, "issuer_name", ""),
        former_names=list(_record_value(record, "former_names", []) or []),
        external_identifiers=dict(_parse_json(_record_value(record, "external_identifiers"), {})),
        identifier_lineage=list(_parse_json(_record_value(record, "identifier_lineage"), [])),
        is_active=record["is_active"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


class SecurityMasterRepository:
    """CRUD operations for the securities table."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def create_table(self) -> None:
        """Backward-compatible schema helper for securities."""
        await apply_migrations(self._db)
        logger.info("Securities schema ensured via migrations")

    async def upsert(self, security: Security) -> None:
        """Insert or update a single security."""
        await self._db.fetch(_UPSERT_SQL, *_security_upsert_args(security))

    async def bulk_upsert(self, securities: list[Security]) -> int:
        """Insert or update multiple securities in one statement.

        Returns the number of securities processed.
        """
        if not securities:
            return 0

        await self._db.execute(_BULK_UPSERT_SQL, *_bulk_upsert_args(securities))
        logger.info("Bulk upserted %d securities", len(securities))
        return len(securities)

    async def get_by_ticker(self, ticker: str, exchange: str = "US") -> Security | None:
        """Fetch a single security by ticker and exchange."""
        row = await self._db.fetchrow(
            "SELECT * FROM securities WHERE ticker = $1 AND exchange = $2",
            ticker,
            exchange,
        )
        return _record_to_security(row) if row else None

    async def get_by_keys(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> dict[tuple[str, str], Security]:
        """Fetch securities for a batch of (ticker, exchange) composite keys."""
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            return {}

        tickers = [ticker for ticker, _exchange in unique_keys]
        exchanges = [exchange for _ticker, exchange in unique_keys]
        rows = await self._db.fetch(
            """
            SELECT securities.*
            FROM securities
            JOIN unnest($1::text[], $2::text[]) AS requested(ticker, exchange)
              ON securities.ticker = requested.ticker
             AND securities.exchange = requested.exchange
            ORDER BY securities.ticker, securities.exchange
            """,
            tickers,
            exchanges,
        )
        securities = [_record_to_security(row) for row in rows]
        return {(security.ticker, security.exchange): security for security in securities}

    async def list_by_external_identifier(self, identifier_key: str) -> list[Security]:
        """List securities that carry a top-level external identifier key."""
        rows = await self._db.fetch(
            """
            SELECT *
            FROM securities
            WHERE external_identifiers ? $1
            ORDER BY ticker, exchange
            """,
            identifier_key,
        )
        return [_record_to_security(row) for row in rows]

    async def list_securities(
        self,
        search: str | None = None,
        active_only: bool = False,
        exchange: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Security], int]:
        """Paginated list with filters. Returns (securities, total)."""
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if active_only:
            conditions.append("is_active = TRUE")

        if exchange:
            conditions.append(f"exchange = ${idx}")
            params.append(exchange)
            idx += 1

        if search:
            conditions.append(
                f"""(
                    ticker ILIKE ${idx}
                    OR sec_cik ILIKE ${idx}
                    OR name ILIKE ${idx}
                    OR issuer_name ILIKE ${idx}
                    OR EXISTS (
                        SELECT 1 FROM unnest(aliases) AS alias_value
                        WHERE alias_value ILIKE ${idx}
                    )
                    OR EXISTS (
                        SELECT 1 FROM unnest(former_names) AS former_name_value
                        WHERE former_name_value ILIKE ${idx}
                    )
                )"""
            )
            params.append(f"%{search}%")
            idx += 1

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM securities{where_clause}"
        total = await self._db.fetchval(count_sql, *params)

        data_sql = f"""
            SELECT * FROM securities{where_clause}
            ORDER BY ticker, exchange
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        params.extend([limit, offset])
        rows = await self._db.fetch(data_sql, *params)

        return [_record_to_security(r) for r in rows], total or 0

    async def get_all_active(self) -> list[Security]:
        """Fetch all active securities."""
        rows = await self._db.fetch(
            "SELECT * FROM securities WHERE is_active = TRUE ORDER BY ticker"
        )
        return [_record_to_security(r) for r in rows]

    async def get_all_active_tickers(self) -> set[str]:
        """Fetch just the ticker symbols for all active securities."""
        rows = await self._db.fetch("SELECT ticker FROM securities WHERE is_active = TRUE")
        return {r["ticker"] for r in rows}

    async def get_by_sec_cik(
        self,
        sec_cik: str,
        *,
        active_only: bool = True,
    ) -> list[Security]:
        """Fetch securities tied to an SEC CIK.

        A CIK identifies an issuer, not necessarily a single listed
        instrument, so this returns a deterministic list instead of a
        single row.
        """
        normalized_cik = normalize_sec_cik(sec_cik)
        conditions = ["sec_cik = $1"]
        if active_only:
            conditions.append("is_active = TRUE")
        rows = await self._db.fetch(
            f"""
            SELECT * FROM securities
            WHERE {" AND ".join(conditions)}
            ORDER BY is_active DESC, ticker, exchange
            """,
            normalized_cik,
        )
        return [_record_to_security(row) for row in rows]

    async def resolve_sec_identifier(
        self,
        identifier: str,
        *,
        active_only: bool = True,
        limit: int = 10,
    ) -> list[Security]:
        """Resolve a ticker, CIK, current name, alias, or former issuer name.

        The result is ordered deterministically so ambiguous names prefer
        active current securities while still allowing callers to inspect
        inactive/renamed rows when ``active_only`` is false.
        """
        raw_identifier = identifier.strip()
        if not raw_identifier:
            return []

        ticker = raw_identifier.upper()
        try:
            normalized_cik = normalize_sec_cik(raw_identifier)
        except ValueError:
            normalized_cik = None

        active_clause = "AND is_active = TRUE" if active_only else ""
        rows = await self._db.fetch(
            f"""
            SELECT * FROM securities
            WHERE (
                ticker = $1
                OR sec_cik = $3
                OR lower(name) = lower($2)
                OR lower(issuer_name) = lower($2)
                OR EXISTS (
                    SELECT 1
                    FROM unnest(aliases || former_names) AS alias_value
                    WHERE lower(alias_value) = lower($2)
                )
            )
            {active_clause}
            ORDER BY
                CASE
                    WHEN ticker = $1 AND is_active THEN 0
                    WHEN ticker = $1 THEN 1
                    WHEN sec_cik = $3 AND is_active THEN 2
                    WHEN lower(issuer_name) = lower($2) AND is_active THEN 3
                    WHEN EXISTS (
                        SELECT 1
                        FROM unnest(former_names) AS former_name
                        WHERE lower(former_name) = lower($2)
                    ) AND is_active THEN 4
                    WHEN is_active THEN 5
                    ELSE 6
                END,
                ticker,
                exchange
            LIMIT $4
            """,
            ticker,
            raw_identifier,
            normalized_cik,
            limit,
        )
        return [_record_to_security(row) for row in rows]

    async def get_company_to_ticker_map(self) -> dict[str, str]:
        """Build a company-name-to-ticker lookup from active securities.

        Includes the security name, SEC issuer name, former names, and aliases.
        """
        rows = await self._db.fetch(
            """
            SELECT ticker, name, aliases, issuer_name, former_names
            FROM securities
            WHERE is_active = TRUE
            """
        )
        mapping: dict[str, str] = {}
        for row in rows:
            ticker = row["ticker"]
            if row["name"]:
                mapping[row["name"].lower()] = ticker
            issuer_name = _record_value(row, "issuer_name", "")
            if issuer_name:
                mapping[issuer_name.lower()] = ticker
            for alias in row["aliases"] or []:
                mapping[alias.lower()] = ticker
            for former_name in _record_value(row, "former_names", []) or []:
                mapping[former_name.lower()] = ticker
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
            query,
            threshold,
            limit,
        )
        return [_record_to_security(r) for r in rows]

    async def deactivate(self, ticker: str, exchange: str = "US") -> bool:
        """Soft-deactivate a security. Returns True if a row was updated."""
        result = await self._db.execute(
            """
            UPDATE securities SET is_active = FALSE, updated_at = NOW()
            WHERE ticker = $1 AND exchange = $2 AND is_active = TRUE
            """,
            ticker,
            exchange,
        )
        return result.endswith("1")

    async def count(self) -> int:
        """Count total securities in the table."""
        return int(await self._db.fetchval("SELECT COUNT(*) FROM securities"))
