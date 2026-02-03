"""
pgvector implementation of the VectorStore interface.

Wraps the existing DocumentRepository and Database classes to provide
vector operations through the VectorStore abstraction.
"""

import structlog
from datetime import datetime
from typing import Any

from src.ingestion.schemas import NormalizedDocument
from src.storage.database import Database
from src.storage.repository import DocumentRepository
from src.vectorstore.base import VectorSearchFilter, VectorSearchResult, VectorStore
from src.vectorstore.config import VectorStoreConfig

logger = structlog.get_logger(__name__)


class PgVectorStore(VectorStore):
    """
    pgvector-based vector store implementation.

    Uses the existing DocumentRepository for data access and builds
    dynamic SQL queries for filtered similarity searches.

    Features:
    - HNSW index support for efficient approximate nearest neighbor search
    - Complex metadata filtering (platform, tickers, theme_ids, authority_score)
    - Integration with existing document storage
    """

    def __init__(
        self,
        database: Database,
        repository: DocumentRepository | None = None,
        config: VectorStoreConfig | None = None,
    ):
        """
        Initialize pgvector store.

        Args:
            database: Connected Database instance
            repository: Optional DocumentRepository (created if not provided)
            config: Optional configuration
        """
        self._db = database
        self._repo = repository or DocumentRepository(database)
        self._config = config or VectorStoreConfig()

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Update embeddings for existing documents.

        Note: This implementation only updates embeddings, not full documents.
        Use DocumentRepository.insert_batch() for full document upserts.

        Args:
            ids: Document IDs
            embeddings: 768-dimensional FinBERT embeddings
            metadata: Ignored (metadata is managed via DocumentRepository)

        Returns:
            Number of documents updated
        """
        if len(ids) != len(embeddings):
            raise ValueError(
                f"ids and embeddings must have same length: {len(ids)} != {len(embeddings)}"
            )

        updated = 0
        for doc_id, embedding in zip(ids, embeddings):
            if await self._repo.update_embedding(doc_id, embedding):
                updated += 1

        logger.info(f"Upserted {updated}/{len(ids)} embeddings")
        return updated

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar documents using cosine similarity.

        Builds a dynamic SQL query with the pgvector <=> operator
        (cosine distance) and optional filters.

        Args:
            query_embedding: Query vector (768 dimensions for FinBERT)
            limit: Maximum results to return
            threshold: Minimum similarity (0.0-1.0)
            filters: Optional filter criteria

        Returns:
            List of search results sorted by similarity (descending)
        """
        # Convert embedding to pgvector string format
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        # Build dynamic query with filters
        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = [embedding_str]
        param_idx = 2

        if filters:
            if filters.platforms:
                conditions.append(f"platform = ANY(${param_idx})")
                params.append(filters.platforms)
                param_idx += 1

            if filters.tickers:
                # Array overlap: document tickers && filter tickers
                conditions.append(f"tickers && ${param_idx}")
                params.append(filters.tickers)
                param_idx += 1

            if filters.theme_ids:
                # Array overlap: document theme_ids && filter theme_ids
                conditions.append(f"theme_ids && ${param_idx}")
                params.append(filters.theme_ids)
                param_idx += 1

            if filters.min_authority_score is not None:
                conditions.append(f"authority_score >= ${param_idx}")
                params.append(filters.min_authority_score)
                param_idx += 1

            if filters.exclude_ids:
                conditions.append(f"id != ALL(${param_idx})")
                params.append(filters.exclude_ids)
                param_idx += 1

            if filters.timestamp_after:
                conditions.append(f"timestamp >= ${param_idx}")
                params.append(filters.timestamp_after)
                param_idx += 1

            if filters.timestamp_before:
                conditions.append(f"timestamp <= ${param_idx}")
                params.append(filters.timestamp_before)
                param_idx += 1

        where_clause = " AND ".join(conditions)

        # Build query with similarity calculation
        sql = f"""
            SELECT
                id,
                platform,
                url,
                title,
                content,
                author_name,
                author_verified,
                author_followers,
                tickers,
                theme_ids,
                spam_score,
                authority_score,
                engagement,
                timestamp,
                1 - (embedding <=> $1) AS similarity
            FROM documents
            WHERE {where_clause}
              AND 1 - (embedding <=> $1) >= ${param_idx}
            ORDER BY embedding <=> $1
            LIMIT ${param_idx + 1}
        """
        params.extend([threshold, limit])

        rows = await self._db.fetch(sql, *params)

        return [self._row_to_result(row) for row in rows]

    async def search_by_centroid(
        self,
        centroid: list[float],
        limit: int = 100,
        threshold: float = 0.5,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for documents near a cluster centroid.

        Uses the same implementation as search() but with different defaults.
        """
        return await self.search(
            query_embedding=centroid,
            limit=limit,
            threshold=threshold,
            filters=filters,
        )

    async def delete(self, ids: list[str]) -> int:
        """
        Delete documents by ID.

        Args:
            ids: Document IDs to delete

        Returns:
            Number of documents deleted
        """
        if not ids:
            return 0

        sql = """
            DELETE FROM documents
            WHERE id = ANY($1)
            RETURNING id
        """
        rows = await self._db.fetch(sql, ids)
        deleted = len(rows)

        logger.info(f"Deleted {deleted}/{len(ids)} documents")
        return deleted

    async def get_by_ids(
        self,
        ids: list[str],
        include_embeddings: bool = False,
    ) -> list[VectorSearchResult]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: Document IDs to retrieve
            include_embeddings: Whether to include embedding vectors

        Returns:
            List of results (score is 1.0 for exact ID matches)
        """
        if not ids:
            return []

        # Select embedding column only if requested
        embedding_col = ", embedding" if include_embeddings else ""

        sql = f"""
            SELECT
                id,
                platform,
                url,
                title,
                content,
                author_name,
                author_verified,
                author_followers,
                tickers,
                theme_ids,
                spam_score,
                authority_score,
                engagement,
                timestamp
                {embedding_col}
            FROM documents
            WHERE id = ANY($1)
        """
        rows = await self._db.fetch(sql, ids)

        return [
            self._row_to_result(row, score=1.0, include_embedding=include_embeddings)
            for row in rows
        ]

    async def update_authority_score(
        self,
        doc_id: str,
        authority_score: float,
    ) -> bool:
        """
        Update a document's authority score.

        Args:
            doc_id: Document ID
            authority_score: Computed authority score (0.0-1.0)

        Returns:
            True if document was updated
        """
        sql = """
            UPDATE documents
            SET authority_score = $2, updated_at = NOW()
            WHERE id = $1
            RETURNING id
        """
        result = await self._db.fetchval(sql, doc_id, authority_score)
        return result is not None

    async def delete_before_timestamp(self, cutoff: datetime) -> int:
        """
        Delete documents with timestamp before cutoff.

        Used for storage cleanup to remove stale documents.

        Args:
            cutoff: Documents with timestamp before this will be deleted

        Returns:
            Number of documents deleted
        """
        sql = """
            DELETE FROM documents
            WHERE timestamp < $1
            RETURNING id
        """
        rows = await self._db.fetch(sql, cutoff)
        deleted = len(rows)

        logger.info(f"Deleted {deleted} documents before {cutoff.isoformat()}")
        return deleted

    def _row_to_result(
        self,
        row: Any,
        score: float | None = None,
        include_embedding: bool = False,
    ) -> VectorSearchResult:
        """Convert database row to VectorSearchResult."""
        import json

        # Parse engagement JSON if present
        engagement = row.get("engagement", {})
        if isinstance(engagement, str):
            engagement = json.loads(engagement)

        metadata = {
            "platform": row["platform"],
            "url": row.get("url"),
            "title": row.get("title"),
            "content_preview": (row.get("content") or "")[:200],
            "author_name": row.get("author_name"),
            "author_verified": row.get("author_verified", False),
            "author_followers": row.get("author_followers"),
            "tickers": list(row.get("tickers") or []),
            "theme_ids": list(row.get("theme_ids") or []),
            "spam_score": row.get("spam_score", 0.0),
            "authority_score": row.get("authority_score"),
            "engagement": engagement,
            "timestamp": row.get("timestamp").isoformat() if row.get("timestamp") else None,
        }

        # Get similarity score from row or use provided score
        result_score = score if score is not None else float(row.get("similarity", 0.0))

        # Parse embedding if included
        embedding = None
        if include_embedding and "embedding" in row and row["embedding"]:
            emb_value = row["embedding"]
            if isinstance(emb_value, str):
                embedding = [float(x) for x in emb_value.strip("[]").split(",")]
            elif isinstance(emb_value, list):
                embedding = emb_value

        return VectorSearchResult(
            document_id=row["id"],
            score=result_score,
            metadata=metadata,
            embedding=embedding,
        )
