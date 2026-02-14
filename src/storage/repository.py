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
            entities_mentioned JSONB NOT NULL DEFAULT '[]',
            keywords_extracted JSONB NOT NULL DEFAULT '[]',
            urls_mentioned TEXT[] NOT NULL DEFAULT '{}',
            spam_score REAL NOT NULL DEFAULT 0.0,
            bot_probability REAL NOT NULL DEFAULT 0.0,
            authority_score REAL,
            embedding vector(768),
            embedding_minilm vector(384),
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
        CREATE INDEX IF NOT EXISTS idx_documents_entities
            ON documents USING GIN(entities_mentioned);
        CREATE INDEX IF NOT EXISTS idx_documents_keywords
            ON documents USING GIN(keywords_extracted);
        CREATE INDEX IF NOT EXISTS idx_documents_created_at
            ON documents(created_at DESC);

        -- Full-text search index on content
        CREATE INDEX IF NOT EXISTS idx_documents_content_search
            ON documents USING GIN(to_tsvector('english', content));

        -- Partial index for high-quality documents (low spam)
        CREATE INDEX IF NOT EXISTS idx_documents_quality
            ON documents(platform, timestamp DESC)
            WHERE spam_score < 0.5;

        -- HNSW index for efficient similarity search on embeddings
        -- Only create if extension supports it (pgvector >= 0.5.0)
        CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
            ON documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);

        -- HNSW index for MiniLM embeddings (384-dim)
        CREATE INDEX IF NOT EXISTS idx_documents_embedding_minilm_hnsw
            ON documents
            USING hnsw (embedding_minilm vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);

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

        -- Themes table (clustering results from BERTopicService)
        CREATE TABLE IF NOT EXISTS themes (
            theme_id        TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            description     TEXT,
            centroid        vector(768) NOT NULL,
            top_keywords    TEXT[] NOT NULL DEFAULT '{}',
            top_tickers     TEXT[] NOT NULL DEFAULT '{}',
            top_entities    JSONB NOT NULL DEFAULT '[]',
            document_count  INTEGER NOT NULL DEFAULT 0,
            lifecycle_stage TEXT NOT NULL DEFAULT 'emerging'
                CHECK (lifecycle_stage IN ('emerging', 'accelerating', 'mature', 'fading')),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata        JSONB NOT NULL DEFAULT '{}',
            deleted_at      TIMESTAMPTZ
        );

        CREATE INDEX IF NOT EXISTS idx_themes_centroid_hnsw
            ON themes
            USING hnsw (centroid vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        CREATE INDEX IF NOT EXISTS idx_themes_top_keywords
            ON themes USING GIN(top_keywords);
        CREATE INDEX IF NOT EXISTS idx_themes_lifecycle_stage
            ON themes(lifecycle_stage);
        CREATE INDEX IF NOT EXISTS idx_themes_updated_at
            ON themes(updated_at DESC);

        -- Auto-update updated_at on themes (reuses trigger function from documents)
        DROP TRIGGER IF EXISTS update_themes_updated_at ON themes;
        CREATE TRIGGER update_themes_updated_at
            BEFORE UPDATE ON themes
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();

        -- Theme metrics time series (one row per theme per day)
        CREATE TABLE IF NOT EXISTS theme_metrics (
            theme_id        TEXT NOT NULL REFERENCES themes(theme_id) ON DELETE CASCADE,
            date            DATE NOT NULL,
            document_count  INTEGER NOT NULL DEFAULT 0,
            weighted_volume REAL,
            sentiment_score REAL,
            volume_zscore   REAL,
            velocity        REAL,
            acceleration    REAL,
            avg_authority   REAL,
            bullish_ratio   REAL,
            PRIMARY KEY (theme_id, date)
        );

        CREATE INDEX IF NOT EXISTS idx_theme_metrics_date
            ON theme_metrics(date);

        -- Events table (extracted events from documents)
        CREATE TABLE IF NOT EXISTS events (
            event_id            TEXT PRIMARY KEY,
            doc_id              TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            event_type          TEXT NOT NULL,
            actor               TEXT,
            action              TEXT NOT NULL,
            object              TEXT,
            time_ref            TEXT,
            quantity            TEXT,
            tickers             TEXT[] NOT NULL DEFAULT '{}',
            confidence          REAL NOT NULL DEFAULT 0.0,
            span_start          INTEGER NOT NULL,
            span_end            INTEGER NOT NULL,
            extractor_version   TEXT NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_events_event_type
            ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_tickers
            ON events USING GIN(tickers);
        CREATE INDEX IF NOT EXISTS idx_events_doc_id
            ON events(doc_id);
        CREATE INDEX IF NOT EXISTS idx_events_created_at
            ON events(created_at DESC);
        """

        await self._db.execute(create_sql)

        # Run migrations for existing tables (add columns that may not exist)
        migrations_sql = """
        -- Add authority_score column if it doesn't exist (for existing tables)
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS authority_score REAL;

        -- Add entities_mentioned column if it doesn't exist
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS entities_mentioned JSONB NOT NULL DEFAULT '[]';

        -- Add index for authority_score filtering
        CREATE INDEX IF NOT EXISTS idx_documents_authority_score
            ON documents(authority_score DESC)
            WHERE authority_score IS NOT NULL;

        -- Composite index for common search pattern (platform + authority)
        CREATE INDEX IF NOT EXISTS idx_documents_platform_authority
            ON documents(platform, authority_score DESC)
            WHERE authority_score IS NOT NULL AND embedding IS NOT NULL;

        -- GIN index for entity queries
        CREATE INDEX IF NOT EXISTS idx_documents_entities
            ON documents USING GIN(entities_mentioned);

        -- Add keywords_extracted column if it doesn't exist
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS keywords_extracted JSONB NOT NULL DEFAULT '[]';

        -- GIN index for keyword queries
        CREATE INDEX IF NOT EXISTS idx_documents_keywords
            ON documents USING GIN(keywords_extracted);

        -- Add weighted_volume column to theme_metrics if it doesn't exist
        ALTER TABLE theme_metrics ADD COLUMN IF NOT EXISTS weighted_volume REAL;

        -- Add events_extracted column if it doesn't exist
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS events_extracted JSONB NOT NULL DEFAULT '[]';

        -- GIN index for event queries
        CREATE INDEX IF NOT EXISTS idx_documents_events_extracted
            ON documents USING GIN(events_extracted);
        """
        await self._db.execute(migrations_sql)

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
            engagement, tickers, entities_mentioned, keywords_extracted,
            events_extracted, urls_mentioned,
            spam_score, bot_probability, authority_score,
            embedding, sentiment, theme_ids, raw_data
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9,
            $10, $11, $12,
            $13, $14, $15, $16,
            $17, $18,
            $19, $20, $21,
            $22, $23, $24, $25
        )
        ON CONFLICT (id) DO UPDATE SET
            engagement = EXCLUDED.engagement,
            entities_mentioned = EXCLUDED.entities_mentioned,
            keywords_extracted = EXCLUDED.keywords_extracted,
            events_extracted = EXCLUDED.events_extracted,
            spam_score = EXCLUDED.spam_score,
            bot_probability = EXCLUDED.bot_probability,
            authority_score = EXCLUDED.authority_score,
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
            json.dumps(doc.entities_mentioned),
            json.dumps(doc.keywords_extracted),
            json.dumps(doc.events_extracted),
            doc.urls_mentioned,
            doc.spam_score,
            doc.bot_probability,
            doc.authority_score,
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
            engagement, tickers, entities_mentioned, keywords_extracted,
            events_extracted, urls_mentioned,
            spam_score, bot_probability, authority_score,
            raw_data
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9,
            $10, $11, $12,
            $13, $14, $15, $16,
            $17, $18,
            $19, $20, $21,
            $22
        )
        ON CONFLICT (id) DO UPDATE SET
            engagement = EXCLUDED.engagement,
            entities_mentioned = EXCLUDED.entities_mentioned,
            keywords_extracted = EXCLUDED.keywords_extracted,
            events_extracted = EXCLUDED.events_extracted,
            spam_score = EXCLUDED.spam_score,
            bot_probability = EXCLUDED.bot_probability,
            authority_score = EXCLUDED.authority_score,
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
                json.dumps(doc.entities_mentioned),
                json.dumps(doc.keywords_extracted),
                json.dumps(doc.events_extracted),
                doc.urls_mentioned,
                doc.spam_score,
                doc.bot_probability,
                doc.authority_score,
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

    # Entity query methods

    async def get_documents_by_entity(
        self,
        entity_type: str,
        entity_normalized: str,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[NormalizedDocument]:
        """
        Get documents mentioning a specific entity.

        Uses JSONB containment query on entities_mentioned.

        Args:
            entity_type: Entity type (COMPANY, PRODUCT, TECHNOLOGY, METRIC)
            entity_normalized: Normalized entity name to search for
            limit: Maximum documents to return
            since: Optional timestamp filter

        Returns:
            List of documents mentioning the entity
        """
        # Build JSONB containment query
        entity_filter = json.dumps([{"type": entity_type, "normalized": entity_normalized}])

        if since:
            sql = """
                SELECT * FROM documents
                WHERE entities_mentioned @> $1::jsonb
                  AND timestamp >= $2
                ORDER BY timestamp DESC
                LIMIT $3
            """
            rows = await self._db.fetch(sql, entity_filter, since, limit)
        else:
            sql = """
                SELECT * FROM documents
                WHERE entities_mentioned @> $1::jsonb
                ORDER BY timestamp DESC
                LIMIT $2
            """
            rows = await self._db.fetch(sql, entity_filter, limit)

        return [self._row_to_document(row) for row in rows]

    async def get_entity_counts(
        self,
        entity_type: str | None = None,
        limit: int = 50,
    ) -> list[tuple[str, str, int]]:
        """
        Get counts of entities by type and normalized name.

        Args:
            entity_type: Optional entity type filter
            limit: Maximum results to return

        Returns:
            List of (type, normalized, count) tuples
        """
        if entity_type:
            sql = """
                SELECT
                    entity->>'type' AS type,
                    entity->>'normalized' AS normalized,
                    COUNT(*) AS cnt
                FROM documents,
                     jsonb_array_elements(entities_mentioned) AS entity
                WHERE entity->>'type' = $1
                GROUP BY entity->>'type', entity->>'normalized'
                ORDER BY cnt DESC
                LIMIT $2
            """
            rows = await self._db.fetch(sql, entity_type, limit)
        else:
            sql = """
                SELECT
                    entity->>'type' AS type,
                    entity->>'normalized' AS normalized,
                    COUNT(*) AS cnt
                FROM documents,
                     jsonb_array_elements(entities_mentioned) AS entity
                GROUP BY entity->>'type', entity->>'normalized'
                ORDER BY cnt DESC
                LIMIT $1
            """
            rows = await self._db.fetch(sql, limit)

        return [(row["type"], row["normalized"], row["cnt"]) for row in rows]

    # Embedding methods

    async def update_embedding(
        self,
        doc_id: str,
        embedding: list[float],
    ) -> bool:
        """
        Update a document's FinBERT embedding vector.

        Args:
            doc_id: Document ID
            embedding: 768-dimensional embedding vector

        Returns:
            True if document was updated, False if not found
        """
        # Convert list to pgvector string format
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        sql = """
            UPDATE documents
            SET embedding = $2, updated_at = NOW()
            WHERE id = $1
            RETURNING id
        """
        result = await self._db.fetchval(sql, doc_id, embedding_str)
        return result is not None

    async def update_embedding_minilm(
        self,
        doc_id: str,
        embedding: list[float],
    ) -> bool:
        """
        Update a document's MiniLM embedding vector.

        Args:
            doc_id: Document ID
            embedding: 384-dimensional embedding vector

        Returns:
            True if document was updated, False if not found
        """
        # Convert list to pgvector string format
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        sql = """
            UPDATE documents
            SET embedding_minilm = $2, updated_at = NOW()
            WHERE id = $1
            RETURNING id
        """
        result = await self._db.fetchval(sql, doc_id, embedding_str)
        return result is not None

    async def update_sentiment(
        self,
        doc_id: str,
        sentiment: dict[str, Any],
    ) -> bool:
        """
        Update a document's sentiment analysis.

        Args:
            doc_id: Document ID
            sentiment: Sentiment analysis result dictionary

        Returns:
            True if document was updated, False if not found
        """
        sql = """
            UPDATE documents
            SET sentiment = $2, updated_at = NOW()
            WHERE id = $1
            RETURNING id
        """
        result = await self._db.fetchval(sql, doc_id, json.dumps(sentiment))
        return result is not None

    async def update_themes(
        self,
        doc_id: str,
        theme_ids: list[str],
    ) -> bool:
        """
        Add theme IDs to a document's theme_ids array (merge, no duplicates).

        Uses array concatenation + DISTINCT unnest to safely merge
        new theme_ids with any existing ones, supporting concurrent
        workers assigning different themes.

        Args:
            doc_id: Document ID
            theme_ids: Theme IDs to add

        Returns:
            True if document was updated, False if not found
        """
        sql = """
            UPDATE documents
            SET theme_ids = (
                SELECT ARRAY(SELECT DISTINCT unnest(theme_ids || $2))
            ),
            updated_at = NOW()
            WHERE id = $1
            RETURNING id
        """
        result = await self._db.fetchval(sql, doc_id, theme_ids)
        return result is not None

    async def get_documents_without_sentiment(
        self,
        limit: int = 100,
        platform: Platform | None = None,
    ) -> list[NormalizedDocument]:
        """
        Get documents that don't have sentiment analysis yet.

        Useful for backfilling sentiment on existing documents.

        Args:
            limit: Maximum documents to return
            platform: Optional platform filter

        Returns:
            List of documents without sentiment
        """
        if platform:
            sql = """
                SELECT * FROM documents
                WHERE sentiment IS NULL AND platform = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            rows = await self._db.fetch(sql, platform.value, limit)
        else:
            sql = """
                SELECT * FROM documents
                WHERE sentiment IS NULL
                ORDER BY created_at DESC
                LIMIT $1
            """
            rows = await self._db.fetch(sql, limit)

        return [self._row_to_document(row) for row in rows]

    async def get_documents_without_embedding(
        self,
        limit: int = 100,
        platform: Platform | None = None,
    ) -> list[NormalizedDocument]:
        """
        Get documents that don't have embeddings yet.

        Useful for backfilling embeddings on existing documents.

        Args:
            limit: Maximum documents to return
            platform: Optional platform filter

        Returns:
            List of documents without embeddings
        """
        if platform:
            sql = """
                SELECT * FROM documents
                WHERE embedding IS NULL AND platform = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            rows = await self._db.fetch(sql, platform.value, limit)
        else:
            sql = """
                SELECT * FROM documents
                WHERE embedding IS NULL
                ORDER BY created_at DESC
                LIMIT $1
            """
            rows = await self._db.fetch(sql, limit)

        return [self._row_to_document(row) for row in rows]

    async def similarity_search(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        platform: Platform | None = None,
        exclude_doc_ids: list[str] | None = None,
    ) -> list[tuple[NormalizedDocument, float]]:
        """
        Find documents similar to a given embedding using cosine similarity.

        Uses pgvector's HNSW index for efficient approximate nearest neighbor search.

        Args:
            embedding: Query embedding vector (768 dimensions)
            limit: Maximum documents to return
            threshold: Minimum similarity score (0-1, higher = more similar)
            platform: Optional platform filter
            exclude_doc_ids: Optional list of document IDs to exclude

        Returns:
            List of (document, similarity_score) tuples, sorted by similarity
        """
        # Build query with optional filters
        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = [embedding]
        param_idx = 2

        if platform:
            conditions.append(f"platform = ${param_idx}")
            params.append(platform.value)
            param_idx += 1

        if exclude_doc_ids:
            conditions.append(f"id != ALL(${param_idx})")
            params.append(exclude_doc_ids)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Use cosine distance operator (<=>), convert to similarity (1 - distance)
        sql = f"""
            SELECT *,
                   1 - (embedding <=> $1) AS similarity
            FROM documents
            WHERE {where_clause}
              AND 1 - (embedding <=> $1) >= ${param_idx}
            ORDER BY embedding <=> $1
            LIMIT ${param_idx + 1}
        """
        params.extend([threshold, limit])

        rows = await self._db.fetch(sql, *params)

        return [
            (self._row_to_document(row), float(row["similarity"]))
            for row in rows
        ]

    async def similarity_search_minilm(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        platform: Platform | None = None,
        exclude_doc_ids: list[str] | None = None,
    ) -> list[tuple[NormalizedDocument, float]]:
        """
        Find documents similar to a given MiniLM embedding using cosine similarity.

        Uses pgvector's HNSW index for efficient approximate nearest neighbor search.

        Args:
            embedding: Query embedding vector (384 dimensions)
            limit: Maximum documents to return
            threshold: Minimum similarity score (0-1, higher = more similar)
            platform: Optional platform filter
            exclude_doc_ids: Optional list of document IDs to exclude

        Returns:
            List of (document, similarity_score) tuples, sorted by similarity
        """
        # Build query with optional filters
        conditions = ["embedding_minilm IS NOT NULL"]
        params: list[Any] = [embedding]
        param_idx = 2

        if platform:
            conditions.append(f"platform = ${param_idx}")
            params.append(platform.value)
            param_idx += 1

        if exclude_doc_ids:
            conditions.append(f"id != ALL(${param_idx})")
            params.append(exclude_doc_ids)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Use cosine distance operator (<=>), convert to similarity (1 - distance)
        sql = f"""
            SELECT *,
                   1 - (embedding_minilm <=> $1) AS similarity
            FROM documents
            WHERE {where_clause}
              AND 1 - (embedding_minilm <=> $1) >= ${param_idx}
            ORDER BY embedding_minilm <=> $1
            LIMIT ${param_idx + 1}
        """
        params.extend([threshold, limit])

        rows = await self._db.fetch(sql, *params)

        return [
            (self._row_to_document(row), float(row["similarity"]))
            for row in rows
        ]

    async def get_embedding_stats(self) -> dict[str, Any]:
        """
        Get statistics about embeddings in the database.

        Returns:
            Dictionary with embedding statistics for both FinBERT and MiniLM
        """
        sql = """
            SELECT
                COUNT(*) AS total_documents,
                COUNT(embedding) AS documents_with_finbert,
                COUNT(embedding_minilm) AS documents_with_minilm,
                COUNT(*) - COUNT(embedding) AS documents_without_finbert,
                COUNT(*) - COUNT(embedding_minilm) AS documents_without_minilm,
                COUNT(embedding) * 100.0 / NULLIF(COUNT(*), 0) AS finbert_coverage_pct,
                COUNT(embedding_minilm) * 100.0 / NULLIF(COUNT(*), 0) AS minilm_coverage_pct
            FROM documents
        """
        row = await self._db.fetchrow(sql)

        return {
            "total_documents": row["total_documents"],
            "documents_with_finbert": row["documents_with_finbert"],
            "documents_with_minilm": row["documents_with_minilm"],
            "documents_without_finbert": row["documents_without_finbert"],
            "documents_without_minilm": row["documents_without_minilm"],
            "finbert_coverage_pct": round(row["finbert_coverage_pct"] or 0, 2),
            "minilm_coverage_pct": round(row["minilm_coverage_pct"] or 0, 2),
        }

    def _parse_embedding(self, value: str | list | None) -> list[float] | None:
        """Parse pgvector string to list of floats."""
        if value is None:
            return None
        if isinstance(value, list):
            return value
        # pgvector returns string like "[0.1,0.2,...]"
        if isinstance(value, str):
            return [float(x) for x in value.strip("[]").split(",")]
        return None

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

        # Parse entities_mentioned JSON
        entities_mentioned = row.get("entities_mentioned", [])
        if isinstance(entities_mentioned, str):
            entities_mentioned = json.loads(entities_mentioned)

        # Parse keywords_extracted JSON
        keywords_extracted = row.get("keywords_extracted", [])
        if isinstance(keywords_extracted, str):
            keywords_extracted = json.loads(keywords_extracted)

        # Parse events_extracted JSON
        events_extracted = row.get("events_extracted", [])
        if isinstance(events_extracted, str):
            events_extracted = json.loads(events_extracted)

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
            entities_mentioned=entities_mentioned or [],
            keywords_extracted=keywords_extracted or [],
            events_extracted=events_extracted or [],
            urls_mentioned=list(row.get("urls_mentioned", [])),
            spam_score=row.get("spam_score", 0.0),
            bot_probability=row.get("bot_probability", 0.0),
            authority_score=row.get("authority_score"),
            embedding=self._parse_embedding(row.get("embedding")),
            embedding_minilm=self._parse_embedding(row.get("embedding_minilm")),
            sentiment=sentiment,
            theme_ids=list(row.get("theme_ids", [])),
            raw_data=raw_data or {},
        )

    # Projection queries for batch processing

    async def get_with_embeddings_since(
        self,
        since: datetime,
        until: datetime,
        limit: int = 50_000,
    ) -> list[dict[str, Any]]:
        """
        Get lightweight document projections with FinBERT embeddings in a time window.

        Returns only the fields needed for batch clustering (6 fields vs 24+
        in full NormalizedDocument), avoiding materialization of large JSONB
        blobs and engagement objects for memory efficiency at scale.

        Args:
            since: Start of time window (inclusive).
            until: End of time window (exclusive).
            limit: Maximum documents to return.

        Returns:
            List of dicts with keys: id, content, embedding, authority_score,
            sentiment, theme_ids. Embedding is parsed to list[float].
        """
        sql = """
            SELECT id, content, embedding, authority_score, sentiment, theme_ids
            FROM documents
            WHERE embedding IS NOT NULL
              AND timestamp >= $1
              AND timestamp < $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        rows = await self._db.fetch(sql, since, until, limit)

        results = []
        for row in rows:
            sentiment = row.get("sentiment")
            if isinstance(sentiment, str):
                sentiment = json.loads(sentiment)

            results.append({
                "id": row["id"],
                "content": row["content"],
                "embedding": self._parse_embedding(row["embedding"]),
                "authority_score": row.get("authority_score"),
                "sentiment": sentiment,
                "theme_ids": list(row.get("theme_ids", [])),
            })

        return results

    # Sentiment aggregation query methods

    async def get_sentiments_for_ticker(
        self,
        ticker: str,
        since: datetime,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get sentiment data for documents mentioning a ticker.

        Returns lightweight sentiment records for aggregation, not full documents.

        Args:
            ticker: Ticker symbol (case-insensitive)
            since: Start of time window
            until: End of time window (defaults to now)

        Returns:
            List of sentiment records with fields:
            - document_id, timestamp, platform, authority_score, sentiment (JSONB)
        """
        ticker = ticker.upper()
        until = until or datetime.now()

        sql = """
            SELECT
                id AS document_id,
                timestamp,
                platform,
                authority_score,
                sentiment
            FROM documents
            WHERE $1 = ANY(tickers)
              AND sentiment IS NOT NULL
              AND timestamp >= $2
              AND timestamp <= $3
            ORDER BY timestamp DESC
        """
        rows = await self._db.fetch(sql, ticker, since, until)

        results = []
        for row in rows:
            sentiment = row["sentiment"]
            if isinstance(sentiment, str):
                sentiment = json.loads(sentiment)
            results.append({
                "document_id": row["document_id"],
                "timestamp": row["timestamp"],
                "platform": row["platform"],
                "authority_score": row["authority_score"],
                "sentiment": sentiment,
            })

        return results

    async def get_sentiments_for_theme(
        self,
        theme_id: str,
        since: datetime,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get sentiment data for documents in a theme.

        Returns lightweight sentiment records for aggregation, not full documents.

        Args:
            theme_id: Theme identifier
            since: Start of time window
            until: End of time window (defaults to now)

        Returns:
            List of sentiment records with fields:
            - document_id, timestamp, platform, authority_score, sentiment (JSONB)
        """
        until = until or datetime.now()

        sql = """
            SELECT
                id AS document_id,
                timestamp,
                platform,
                authority_score,
                sentiment
            FROM documents
            WHERE $1 = ANY(theme_ids)
              AND sentiment IS NOT NULL
              AND timestamp >= $2
              AND timestamp <= $3
            ORDER BY timestamp DESC
        """
        rows = await self._db.fetch(sql, theme_id, since, until)

        results = []
        for row in rows:
            sentiment = row["sentiment"]
            if isinstance(sentiment, str):
                sentiment = json.loads(sentiment)
            results.append({
                "document_id": row["document_id"],
                "timestamp": row["timestamp"],
                "platform": row["platform"],
                "authority_score": row["authority_score"],
                "sentiment": sentiment,
            })

        return results

    async def get_events_by_tickers(
        self,
        tickers: list[str],
        since: datetime,
        until: datetime | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get events whose tickers overlap with the given list.

        Uses the GIN index on events.tickers with the && (overlap) operator
        for efficient set intersection queries.

        Args:
            tickers: Ticker symbols to match against event tickers.
            since: Start of time window (inclusive).
            until: End of time window (optional, defaults to now).
            event_type: Optional event type filter.
            limit: Maximum events to return.

        Returns:
            List of event dicts with all event table columns.
        """
        if not tickers:
            return []

        until = until or datetime.now()

        conditions = ["tickers && $1::text[]", "created_at >= $2"]
        params: list[Any] = [tickers, since]
        param_idx = 3

        conditions.append(f"created_at <= ${param_idx}")
        params.append(until)
        param_idx += 1

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT event_id, doc_id, event_type, actor, action, object,
                   time_ref, quantity, tickers, confidence, span_start,
                   span_end, extractor_version, created_at
            FROM events
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx}
        """
        params.append(limit)

        rows = await self._db.fetch(sql, *params)

        return [
            {
                "event_id": row["event_id"],
                "doc_id": row["doc_id"],
                "event_type": row["event_type"],
                "actor": row["actor"],
                "action": row["action"],
                "object": row["object"],
                "time_ref": row["time_ref"],
                "quantity": row["quantity"],
                "tickers": list(row.get("tickers", [])),
                "confidence": row["confidence"],
                "span_start": row["span_start"],
                "span_end": row["span_end"],
                "extractor_version": row["extractor_version"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _build_document_filters(
        self,
        *,
        platform: str | None = None,
        content_type: str | None = None,
        ticker: str | None = None,
        q: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        max_spam: float | None = None,
        min_authority: float | None = None,
        active_sources_only: bool = False,
        param_idx: int = 1,
    ) -> tuple[str, list[Any], int]:
        """
        Build WHERE clause from optional document filters.

        Returns (where_clause, params, next_param_idx). Reused by
        list_documents and list_documents_count to avoid SQL duplication.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if platform is not None:
            conditions.append(f"platform = ${param_idx}")
            params.append(platform)
            param_idx += 1

        if content_type is not None:
            conditions.append(f"content_type = ${param_idx}")
            params.append(content_type)
            param_idx += 1

        if ticker is not None:
            conditions.append(f"${param_idx} = ANY(tickers)")
            params.append(ticker.upper())
            param_idx += 1

        if q is not None:
            conditions.append(
                f"to_tsvector('english', content) @@ plainto_tsquery('english', ${param_idx})"
            )
            params.append(q)
            param_idx += 1

        if since is not None:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(since)
            param_idx += 1

        if until is not None:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(until)
            param_idx += 1

        if max_spam is not None:
            conditions.append(f"spam_score <= ${param_idx}")
            params.append(max_spam)
            param_idx += 1

        if min_authority is not None:
            conditions.append(f"authority_score >= ${param_idx}")
            params.append(min_authority)
            param_idx += 1

        if active_sources_only:
            conditions.append(
                "NOT EXISTS ("
                "SELECT 1 FROM sources s "
                "WHERE s.platform = documents.platform "
                "AND s.identifier = documents.author_id "
                "AND s.is_active = FALSE"
                ")"
            )

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        return where_clause, params, param_idx

    async def list_documents(
        self,
        *,
        platform: str | None = None,
        content_type: str | None = None,
        ticker: str | None = None,
        q: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        max_spam: float | None = None,
        min_authority: float | None = None,
        active_sources_only: bool = False,
        sort: str = "timestamp",
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> list[asyncpg.Record]:
        """
        List documents with a lightweight projection (no embeddings/raw_data).

        Returns raw asyncpg Records to avoid materializing full
        NormalizedDocument objects. Content is truncated to 300 chars at DB level.
        """
        where_clause, params, idx = self._build_document_filters(
            platform=platform,
            content_type=content_type,
            ticker=ticker,
            q=q,
            since=since,
            until=until,
            max_spam=max_spam,
            min_authority=min_authority,
            active_sources_only=active_sources_only,
        )

        sql = f"""
            SELECT id, platform, content_type, title, LEFT(content, 300) AS content_preview,
                   url, author_name, author_verified, author_followers, tickers,
                   spam_score, authority_score, sentiment, engagement, theme_ids,
                   timestamp, fetched_at
            FROM documents
            WHERE {where_clause}
            ORDER BY {sort} {order}
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        params.extend([limit, offset])
        return await self._db.fetch(sql, *params)

    async def list_documents_count(
        self,
        *,
        platform: str | None = None,
        content_type: str | None = None,
        ticker: str | None = None,
        q: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        max_spam: float | None = None,
        min_authority: float | None = None,
        active_sources_only: bool = False,
    ) -> int:
        """Count documents matching the same filters as list_documents."""
        where_clause, params, _idx = self._build_document_filters(
            platform=platform,
            content_type=content_type,
            ticker=ticker,
            q=q,
            since=since,
            until=until,
            max_spam=max_spam,
            min_authority=min_authority,
            active_sources_only=active_sources_only,
        )
        sql = f"SELECT COUNT(*) FROM documents WHERE {where_clause}"
        return await self._db.fetchval(sql, *params)

    async def get_document_stats(self) -> dict[str, Any]:
        """
        Get aggregate document statistics for the explorer dashboard.

        Returns dict with total_count, platform_counts, embedding/sentiment
        coverage, and date range.
        """
        summary_sql = """
            SELECT
                COUNT(*) AS total,
                COUNT(embedding)::float / GREATEST(COUNT(*), 1) AS finbert_pct,
                COUNT(embedding_minilm)::float / GREATEST(COUNT(*), 1) AS minilm_pct,
                COUNT(sentiment)::float / GREATEST(COUNT(*), 1) AS sentiment_pct,
                MIN(timestamp) AS earliest,
                MAX(timestamp) AS latest,
                MAX(fetched_at) AS latest_fetched
            FROM documents
        """
        platform_sql = """
            SELECT platform, COUNT(*) AS count
            FROM documents
            GROUP BY platform
            ORDER BY count DESC
        """
        summary_row = await self._db.fetchrow(summary_sql)
        platform_rows = await self._db.fetch(platform_sql)

        return {
            "total_count": summary_row["total"],
            "platform_counts": [
                {"platform": r["platform"], "count": r["count"]}
                for r in platform_rows
            ],
            "embedding_coverage": {
                "finbert_pct": round(summary_row["finbert_pct"], 4),
                "minilm_pct": round(summary_row["minilm_pct"], 4),
            },
            "sentiment_coverage": round(summary_row["sentiment_pct"], 4),
            "earliest_document": (
                summary_row["earliest"].isoformat() if summary_row["earliest"] else None
            ),
            "latest_document": (
                summary_row["latest"].isoformat() if summary_row["latest"] else None
            ),
            "latest_fetched_at": (
                summary_row["latest_fetched"].isoformat()
                if summary_row["latest_fetched"]
                else None
            ),
        }

    # ── Entity Explorer Methods ─────────────────────────

    async def list_entities(
        self,
        entity_type: str | None = None,
        search: str | None = None,
        sort: str = "count",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Paginated entity directory with optional type filter and search.

        Returns (entities, total) where each entity dict has:
        type, normalized, mention_count, first_seen, last_seen.
        """
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if entity_type:
            conditions.append(f"entity->>'type' = ${idx}")
            params.append(entity_type)
            idx += 1

        if search:
            conditions.append(f"entity->>'normalized' ILIKE ${idx}")
            params.append(f"%{search}%")
            idx += 1

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        order_col = "mention_count DESC" if sort == "count" else "last_seen DESC"

        count_sql = f"""
            SELECT COUNT(*) FROM (
                SELECT entity->>'type' AS type, entity->>'normalized' AS normalized
                FROM documents, jsonb_array_elements(entities_mentioned) AS entity
                {where_clause}
                GROUP BY entity->>'type', entity->>'normalized'
            ) sub
        """
        total = await self._db.fetchval(count_sql, *params)

        data_sql = f"""
            SELECT
                entity->>'type' AS type,
                entity->>'normalized' AS normalized,
                COUNT(*) AS mention_count,
                MIN(d.timestamp) AS first_seen,
                MAX(d.timestamp) AS last_seen
            FROM documents d, jsonb_array_elements(d.entities_mentioned) AS entity
            {where_clause}
            GROUP BY entity->>'type', entity->>'normalized'
            ORDER BY {order_col}
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        params.extend([limit, offset])
        rows = await self._db.fetch(data_sql, *params)

        entities = [
            {
                "type": r["type"],
                "normalized": r["normalized"],
                "mention_count": r["mention_count"],
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
            }
            for r in rows
        ]
        return entities, total or 0

    async def get_entity_detail(
        self,
        entity_type: str,
        normalized: str,
    ) -> dict[str, Any] | None:
        """
        Single entity stats: count, first/last seen, platform breakdown.
        """
        entity_filter = json.dumps([{"type": entity_type, "normalized": normalized}])

        sql = """
            SELECT
                COUNT(*) AS mention_count,
                MIN(timestamp) AS first_seen,
                MAX(timestamp) AS last_seen
            FROM documents
            WHERE entities_mentioned @> $1::jsonb
        """
        row = await self._db.fetchrow(sql, entity_filter)
        if not row or row["mention_count"] == 0:
            return None

        platform_sql = """
            SELECT platform, COUNT(*) AS count
            FROM documents
            WHERE entities_mentioned @> $1::jsonb
            GROUP BY platform
            ORDER BY count DESC
        """
        platform_rows = await self._db.fetch(platform_sql, entity_filter)

        return {
            "type": entity_type,
            "normalized": normalized,
            "mention_count": row["mention_count"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "platforms": {r["platform"]: r["count"] for r in platform_rows},
        }

    async def get_entity_sentiment(
        self,
        entity_type: str,
        normalized: str,
    ) -> dict[str, Any] | None:
        """
        Aggregate sentiment from documents mentioning an entity.
        """
        entity_filter = json.dumps([{"type": entity_type, "normalized": normalized}])

        sql = """
            SELECT
                AVG((sentiment->>'overall_score')::float) AS avg_score,
                COUNT(*) FILTER (WHERE sentiment->>'overall_label' = 'positive') AS pos_count,
                COUNT(*) FILTER (WHERE sentiment->>'overall_label' = 'negative') AS neg_count,
                COUNT(*) FILTER (WHERE sentiment->>'overall_label' = 'neutral') AS neu_count,
                AVG((sentiment->>'overall_score')::float) FILTER (
                    WHERE timestamp >= NOW() - INTERVAL '7 days'
                ) AS recent_avg,
                AVG((sentiment->>'overall_score')::float) FILTER (
                    WHERE timestamp < NOW() - INTERVAL '7 days'
                      AND timestamp >= NOW() - INTERVAL '30 days'
                ) AS baseline_avg
            FROM documents
            WHERE entities_mentioned @> $1::jsonb
              AND sentiment IS NOT NULL
        """
        row = await self._db.fetchrow(sql, entity_filter)
        if not row or (row["pos_count"] + row["neg_count"] + row["neu_count"]) == 0:
            return None

        # Determine trend
        trend = "stable"
        if row["recent_avg"] is not None and row["baseline_avg"] is not None:
            delta = row["recent_avg"] - row["baseline_avg"]
            if delta > 0.05:
                trend = "improving"
            elif delta < -0.05:
                trend = "declining"

        return {
            "avg_score": row["avg_score"],
            "pos_count": row["pos_count"],
            "neg_count": row["neg_count"],
            "neu_count": row["neu_count"],
            "trend": trend,
        }

    async def get_trending_entities(
        self,
        hours_recent: int = 24,
        hours_baseline: int = 168,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Entities with mention spikes: compare recent window to baseline.
        """
        sql = """
            WITH recent AS (
                SELECT
                    entity->>'type' AS type,
                    entity->>'normalized' AS normalized,
                    COUNT(*) AS recent_count
                FROM documents d, jsonb_array_elements(d.entities_mentioned) AS entity
                WHERE d.fetched_at >= NOW() - make_interval(hours => $1)
                GROUP BY entity->>'type', entity->>'normalized'
            ),
            baseline AS (
                SELECT
                    entity->>'type' AS type,
                    entity->>'normalized' AS normalized,
                    COUNT(*) AS baseline_count
                FROM documents d, jsonb_array_elements(d.entities_mentioned) AS entity
                WHERE d.fetched_at >= NOW() - make_interval(hours => $2)
                  AND d.fetched_at < NOW() - make_interval(hours => $1)
                GROUP BY entity->>'type', entity->>'normalized'
            )
            SELECT
                r.type,
                r.normalized,
                r.recent_count,
                COALESCE(b.baseline_count, 0) AS baseline_count,
                CASE WHEN COALESCE(b.baseline_count, 0) = 0
                     THEN r.recent_count::float
                     ELSE r.recent_count::float * $2::float / (
                         GREATEST(b.baseline_count, 1) * $1::float
                     )
                END AS spike_ratio
            FROM recent r
            LEFT JOIN baseline b ON r.type = b.type AND r.normalized = b.normalized
            WHERE r.recent_count >= 2
            ORDER BY spike_ratio DESC
            LIMIT $3
        """
        rows = await self._db.fetch(sql, hours_recent, hours_baseline, limit)

        return [
            {
                "type": r["type"],
                "normalized": r["normalized"],
                "recent_count": r["recent_count"],
                "baseline_count": r["baseline_count"],
                "spike_ratio": round(float(r["spike_ratio"]), 2),
            }
            for r in rows
        ]

    async def get_cooccurring_entities(
        self,
        entity_type: str,
        normalized: str,
        limit: int = 20,
        min_count: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Entities frequently appearing in same documents as the target entity.
        """
        entity_filter = json.dumps([{"type": entity_type, "normalized": normalized}])

        sql = """
            WITH target_docs AS (
                SELECT id FROM documents
                WHERE entities_mentioned @> $1::jsonb
            ),
            target_count AS (
                SELECT COUNT(*) AS cnt FROM target_docs
            ),
            cooccur AS (
                SELECT
                    entity->>'type' AS type,
                    entity->>'normalized' AS normalized,
                    COUNT(DISTINCT d.id) AS cooccurrence_count
                FROM target_docs td
                JOIN documents d ON d.id = td.id,
                     jsonb_array_elements(d.entities_mentioned) AS entity
                WHERE NOT (entity->>'type' = $2 AND entity->>'normalized' = $3)
                GROUP BY entity->>'type', entity->>'normalized'
                HAVING COUNT(DISTINCT d.id) >= $4
            )
            SELECT
                c.type,
                c.normalized,
                c.cooccurrence_count,
                ROUND(c.cooccurrence_count::numeric / GREATEST(tc.cnt, 1), 4) AS jaccard
            FROM cooccur c, target_count tc
            ORDER BY c.cooccurrence_count DESC
            LIMIT $5
        """
        rows = await self._db.fetch(
            sql, entity_filter, entity_type, normalized, min_count, limit
        )

        return [
            {
                "type": r["type"],
                "normalized": r["normalized"],
                "cooccurrence_count": r["cooccurrence_count"],
                "jaccard": float(r["jaccard"]),
            }
            for r in rows
        ]

    async def merge_entity(
        self,
        from_type: str,
        from_normalized: str,
        to_type: str,
        to_normalized: str,
    ) -> int:
        """
        Rewrite entity references in all matching documents.

        Replaces all occurrences of (from_type, from_normalized) with
        (to_type, to_normalized) in entities_mentioned JSONB arrays.

        Returns the number of affected documents.
        """
        entity_filter = json.dumps(
            [{"type": from_type, "normalized": from_normalized}]
        )

        sql = """
            WITH target_docs AS (
                SELECT id FROM documents
                WHERE entities_mentioned @> $1::jsonb
            ),
            updated AS (
                UPDATE documents d
                SET entities_mentioned = (
                    SELECT jsonb_agg(
                        CASE
                            WHEN elem->>'type' = $2 AND elem->>'normalized' = $3
                            THEN jsonb_set(
                                jsonb_set(elem, '{type}', to_jsonb($4::text)),
                                '{normalized}', to_jsonb($5::text)
                            )
                            ELSE elem
                        END
                    )
                    FROM jsonb_array_elements(d.entities_mentioned) AS elem
                ),
                updated_at = NOW()
                WHERE d.id IN (SELECT id FROM target_docs)
                RETURNING d.id
            )
            SELECT COUNT(*) AS affected FROM updated
        """
        result = await self._db.fetchval(
            sql, entity_filter, from_type, from_normalized, to_type, to_normalized
        )
        return result or 0

    async def get_documents_by_theme(
        self,
        theme_id: str,
        limit: int = 50,
        offset: int = 0,
        platform: str | None = None,
        min_authority: float | None = None,
    ) -> list[NormalizedDocument]:
        """
        Get documents assigned to a theme.

        Uses dynamic SQL builder with optional platform and authority filters.

        Args:
            theme_id: Theme identifier
            limit: Maximum documents to return
            offset: Offset for pagination
            platform: Optional platform filter
            min_authority: Optional minimum authority score filter

        Returns:
            List of documents ordered by timestamp descending
        """
        conditions = ["$1 = ANY(theme_ids)"]
        params: list[Any] = [theme_id]
        param_idx = 2

        if platform:
            conditions.append(f"platform = ${param_idx}")
            params.append(platform)
            param_idx += 1

        if min_authority is not None:
            conditions.append(f"authority_score >= ${param_idx}")
            params.append(min_authority)
            param_idx += 1

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM documents
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        rows = await self._db.fetch(sql, *params)
        return [self._row_to_document(row) for row in rows]
