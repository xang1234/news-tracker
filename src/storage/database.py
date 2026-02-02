"""
PostgreSQL database connection management.

Uses asyncpg for high-performance async database operations.
Provides connection pooling and transaction management.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

import asyncpg

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class Database:
    """
    Async PostgreSQL database connection manager.

    Uses asyncpg connection pool for efficient connection reuse.
    Provides transaction context managers and health checks.

    Usage:
        db = Database()
        await db.connect()

        async with db.transaction() as conn:
            await conn.execute("INSERT INTO ...")

        await db.close()
    """

    def __init__(
        self,
        database_url: str | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
    ):
        """
        Initialize database connection manager.

        Args:
            database_url: PostgreSQL connection URL
            min_size: Minimum pool size
            max_size: Maximum pool size
        """
        settings = get_settings()

        self._database_url = database_url or str(settings.database_url)
        self._min_size = min_size or settings.db_pool_min_size
        self._max_size = max_size or settings.db_pool_max_size

        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """
        Establish database connection pool.

        Creates a connection pool with the configured size limits.
        Also runs initialization queries (e.g., enabling pgvector).
        """
        try:
            self._pool = await asyncpg.create_pool(
                self._database_url,
                min_size=self._min_size,
                max_size=self._max_size,
                command_timeout=60,
            )

            # Initialize database (enable extensions, etc.)
            async with self._pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            logger.info(
                f"Database connected (pool: {self._min_size}-{self._max_size})"
            )

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection closed")

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def pool(self) -> asyncpg.Pool:
        """Get connection pool, raising if not connected."""
        if self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """
        Acquire a connection from the pool.

        Usage:
            async with db.acquire() as conn:
                await conn.execute("...")
        """
        async with self.pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """
        Start a transaction.

        Usage:
            async with db.transaction() as conn:
                await conn.execute("INSERT INTO ...")
                await conn.execute("UPDATE ...")
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """
        Execute a query without returning results.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Status string from PostgreSQL
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """
        Execute a query and fetch all results.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of records
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """
        Execute a query and fetch one result.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single record or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """
        Execute a query and fetch a single value.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single value
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def health_check(self) -> bool:
        """
        Check if database is healthy.

        Returns:
            True if database is accessible
        """
        try:
            result = await self.fetchval("SELECT 1")
            return result == 1
        except Exception:
            return False


# Global database instance
_database: Database | None = None


async def get_database() -> Database:
    """
    Get global database instance.

    Creates and connects if not already connected.

    Returns:
        Connected Database instance
    """
    global _database

    if _database is None:
        _database = Database()
        await _database.connect()

    return _database


async def close_database() -> None:
    """Close global database connection."""
    global _database

    if _database is not None:
        await _database.close()
        _database = None
