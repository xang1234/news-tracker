"""
Document repository for CRUD operations.

Provides high-level database operations for NormalizedDocument storage.
Uses asyncpg for efficient batch operations and supports pgvector
for future embedding storage.
"""

import json
import logging
from datetime import datetime
from typing import Any

import asyncpg

from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform
from src.storage.database import Database

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository for document storage and retrieval.

    Provides CRUD operations for NormalizedDocument with:
    - Efficient batch inserts
    - Upsert with conflict resolution
    - Query by platform, ticker, date range
    - Vector similarity search (when embeddings available)

    Tables:
        - documents: Main document storage
        - document_tickers: Many-to-many relationship
    """

    def __init__(self, database: Database):
        """
        Initialize repository.

        Args:
            database: Connected Database instance
        """
        self._db = database

    async def create_tables(self) -> None:
        """
        Create database tables if they don't exist.

        Creates the document schema including indexes.
        """
        create_sql = """
        -- Enable required extensions
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS pg_trgm;

        -- Main documents table
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            url TEXT,
            timestamp TIMESTAMPTZ NOT NULL,
            fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            author_id TEXT NOT NULL,
            author_name TEXT NOT NULL,
            author_followers INTEGER,
            author_verified BOOLEAN DEFAULT FALSE,
            content TEXT NOT NULL,
            content_type TEXT NOT NULL DEFAULT 'post',
            title TEXT,
            engagement JSONB NOT NULL DEFAULT '{}',
            tickers TEXT[] NOT NULL DEFAULT '{}',
            urls_mentioned TEXT[] NOT NULL DEFAULT '{}',
            spam_score REAL NOT NULL DEFAULT 0.0,
            bot_probability REAL NOT NULL DEFAULT 0.0,
            embedding vector(384),
            sentiment JSONB,
            theme_ids TEXT[] NOT NULL DEFAULT '{}',
            raw_data JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_documents_platform
            ON documents(platform);
        CREATE INDEX IF NOT EXISTS idx_documents_timestamp
            ON documents(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_documents_author_id
            ON documents(author_id);
        CREATE INDEX IF NOT EXISTS idx_documents_tickers
            ON documents USING GIN(tickers);
        CREATE INDEX IF NOT EXISTS idx_documents_created_at
            ON documents(created_at DESC);

        -- Full-text search index on content
        CREATE INDEX IF NOT EXISTS idx_documents_content_search
            ON documents USING GIN(to_tsvector('english', content));

        -- Partial index for high-quality documents (low spam)
        CREATE INDEX IF NOT EXISTS idx_documents_quality
            ON documents(platform, timestamp DESC)
            WHERE spam_score < 0.5;

        -- Processing metrics table (for observability)
        CREATE TABLE IF NOT EXISTS processing_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            dimensions JSONB NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
            ON processing_metrics(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_metrics_name
            ON processing_metrics(metric_name);
        """

        await self._db.execute(create_sql)
        logger.info("Database tables created/verified")

    async def insert(self, doc: NormalizedDocument) -> bool:
        """
        Insert a document into the database.

        Uses upsert semantics - updates if document already exists.

        Args:
            doc: Document to insert

        Returns:
            True if inserted, False if updated
        """
        sql = """
        INSERT INTO documents (
            id, platform, url, timestamp, fetched_at,
            author_id, author_name, author_followers, author_verified,
            content, content_type, title,
            engagement, tickers, urls_mentioned,
            spam_score, bot_probability,
            embedding, sentiment, theme_ids, raw_data
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9,
            $10, $11, $12,
            $13, $14, $15,
            $16, $17,
            $18, $19, $20, $21
        )
        ON CONFLICT (id) DO UPDATE SET
            engagement = EXCLUDED.engagement,
            spam_score = EXCLUDED.spam_score,
            bot_probability = EXCLUDED.bot_probability,
            updated_at = NOW()
        RETURNING (xmax = 0) AS inserted
        """

        result = await self._db.fetchval(
            sql,
            doc.id,
            doc.platform.value if isinstance(doc.platform, Platform) else doc.platform,
            doc.url,
            doc.timestamp,
            doc.fetched_at,
            doc.author_id,
            doc.author_name,
            doc.author_followers,
            doc.author_verified,
            doc.content,
            doc.content_type,
            doc.title,
            json.dumps(doc.engagement.model_dump()),
            doc.tickers_mentioned,
            doc.urls_mentioned,
            doc.spam_score,
            doc.bot_probability,
            doc.embedding,
            json.dumps(doc.sentiment) if doc.sentiment else None,
            doc.theme_ids,
            json.dumps(doc.raw_data) if doc.raw_data else None,
        )

        return bool(result)

    async def insert_batch(
        self,
        docs: list[NormalizedDocument],
    ) -> tuple[int, int]:
        """
        Insert multiple documents efficiently.

        Uses executemany for batch insertion with upsert semantics.

        Args:
            docs: Documents to insert

        Returns:
            Tuple of (inserted_count, updated_count)
        """
        if not docs:
            return 0, 0

        sql = """
        INSERT INTO documents (
            id, platform, url, timestamp, fetched_at,
            author_id, author_name, author_followers, author_verified,
            content, content_type, title,
            engagement, tickers, urls_mentioned,
            spam_score, bot_probability,
            raw_data
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9,
            $10, $11, $12,
            $13, $14, $15,
            $16, $17,
            $18
        )
        ON CONFLICT (id) DO UPDATE SET
            engagement = EXCLUDED.engagement,
            spam_score = EXCLUDED.spam_score,
            bot_probability = EXCLUDED.bot_probability,
            updated_at = NOW()
        """

        # Prepare batch data
        batch_data = [
            (
                doc.id,
                doc.platform.value if isinstance(doc.platform, Platform) else doc.platform,
                doc.url,
                doc.timestamp,
                doc.fetched_at,
                doc.author_id,
                doc.author_name,
                doc.author_followers,
                doc.author_verified,
                doc.content,
                doc.content_type,
                doc.title,
                json.dumps(doc.engagement.model_dump()),
                doc.tickers_mentioned,
                doc.urls_mentioned,
                doc.spam_score,
                doc.bot_probability,
                json.dumps(doc.raw_data) if doc.raw_data else None,
            )
            for doc in docs
        ]

        async with self._db.acquire() as conn:
            await conn.executemany(sql, batch_data)

        logger.info(f"Batch inserted {len(docs)} documents")
        return len(docs), 0  # Can't easily distinguish inserts vs updates in batch

    async def get_by_id(self, doc_id: str) -> NormalizedDocument | None:
        """
        Get document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document or None if not found
        """
        sql = "SELECT * FROM documents WHERE id = $1"
        row = await self._db.fetchrow(sql, doc_id)

        if row is None:
            return None

        return self._row_to_document(row)

    async def exists(self, doc_id: str) -> bool:
        """
        Check if document exists.

        Args:
            doc_id: Document ID

        Returns:
            True if document exists
        """
        sql = "SELECT EXISTS(SELECT 1 FROM documents WHERE id = $1)"
        return await self._db.fetchval(sql, doc_id)

    async def get_by_platform(
        self,
        platform: Platform,
        limit: int = 100,
        offset: int = 0,
        since: datetime | None = None,
    ) -> list[NormalizedDocument]:
        """
        Get documents by platform.

        Args:
            platform: Platform to filter by
            limit: Maximum documents to return
            offset: Offset for pagination
            since: Optional timestamp filter

        Returns:
            List of documents
        """
        if since:
            sql = """
                SELECT * FROM documents
                WHERE platform = $1 AND timestamp >= $2
                ORDER BY timestamp DESC
                LIMIT $3 OFFSET $4
            """
            rows = await self._db.fetch(
                sql,
                platform.value,
                since,
                limit,
                offset,
            )
        else:
            sql = """
                SELECT * FROM documents
                WHERE platform = $1
                ORDER BY timestamp DESC
                LIMIT $2 OFFSET $3
            """
            rows = await self._db.fetch(sql, platform.value, limit, offset)

        return [self._row_to_document(row) for row in rows]

    async def get_by_ticker(
        self,
        ticker: str,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[NormalizedDocument]:
        """
        Get documents mentioning a ticker.

        Args:
            ticker: Ticker symbol
            limit: Maximum documents to return
            since: Optional timestamp filter

        Returns:
            List of documents
        """
        ticker = ticker.upper()

        if since:
            sql = """
                SELECT * FROM documents
                WHERE $1 = ANY(tickers) AND timestamp >= $2
                ORDER BY timestamp DESC
                LIMIT $3
            """
            rows = await self._db.fetch(sql, ticker, since, limit)
        else:
            sql = """
                SELECT * FROM documents
                WHERE $1 = ANY(tickers)
                ORDER BY timestamp DESC
                LIMIT $2
            """
            rows = await self._db.fetch(sql, ticker, limit)

        return [self._row_to_document(row) for row in rows]

    async def search_content(
        self,
        query: str,
        limit: int = 100,
    ) -> list[NormalizedDocument]:
        """
        Full-text search on document content.

        Args:
            query: Search query
            limit: Maximum documents to return

        Returns:
            List of matching documents
        """
        sql = """
            SELECT *, ts_rank_cd(to_tsvector('english', content), query) AS rank
            FROM documents, plainto_tsquery('english', $1) query
            WHERE to_tsvector('english', content) @@ query
            ORDER BY rank DESC
            LIMIT $2
        """
        rows = await self._db.fetch(sql, query, limit)
        return [self._row_to_document(row) for row in rows]

    async def get_document_count(
        self,
        platform: Platform | None = None,
        since: datetime | None = None,
    ) -> int:
        """
        Get count of documents.

        Args:
            platform: Optional platform filter
            since: Optional timestamp filter

        Returns:
            Document count
        """
        conditions = []
        params: list[Any] = []
        param_idx = 1

        if platform:
            conditions.append(f"platform = ${param_idx}")
            params.append(platform.value)
            param_idx += 1

        if since:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(since)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        sql = f"SELECT COUNT(*) FROM documents WHERE {where_clause}"

        return await self._db.fetchval(sql, *params)

    async def delete_old_documents(
        self,
        days: int = 30,
    ) -> int:
        """
        Delete documents older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of deleted documents
        """
        sql = """
            DELETE FROM documents
            WHERE timestamp < NOW() - make_interval(days => $1)
            RETURNING id
        """
        result = await self._db.fetch(sql, days)
        count = len(result)
        logger.info(f"Deleted {count} documents older than {days} days")
        return count

    async def record_metric(
        self,
        name: str,
        value: float,
        dimensions: dict[str, str] | None = None,
    ) -> None:
        """
        Record a processing metric.

        Args:
            name: Metric name
            value: Metric value
            dimensions: Optional dimension labels
        """
        sql = """
            INSERT INTO processing_metrics (metric_name, metric_value, dimensions)
            VALUES ($1, $2, $3)
        """
        await self._db.execute(
            sql,
            name,
            value,
            json.dumps(dimensions or {}),
        )

    def _row_to_document(self, row: asyncpg.Record) -> NormalizedDocument:
        """Convert database row to NormalizedDocument."""
        # Parse engagement JSON
        engagement_data = row["engagement"]
        if isinstance(engagement_data, str):
            engagement_data = json.loads(engagement_data)
        engagement = EngagementMetrics(**engagement_data)

        # Parse sentiment JSON
        sentiment = row.get("sentiment")
        if isinstance(sentiment, str):
            sentiment = json.loads(sentiment)

        # Parse raw_data JSON
        raw_data = row.get("raw_data")
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        return NormalizedDocument(
            id=row["id"],
            platform=Platform(row["platform"]),
            url=row.get("url"),
            timestamp=row["timestamp"],
            fetched_at=row.get("fetched_at", row["timestamp"]),
            author_id=row["author_id"],
            author_name=row["author_name"],
            author_followers=row.get("author_followers"),
            author_verified=row.get("author_verified", False),
            content=row["content"],
            content_type=row.get("content_type", "post"),
            title=row.get("title"),
            engagement=engagement,
            tickers_mentioned=list(row.get("tickers", [])),
            urls_mentioned=list(row.get("urls_mentioned", [])),
            spam_score=row.get("spam_score", 0.0),
            bot_probability=row.get("bot_probability", 0.0),
            embedding=row.get("embedding"),
            sentiment=sentiment,
            theme_ids=list(row.get("theme_ids", [])),
            raw_data=raw_data or {},
        )
