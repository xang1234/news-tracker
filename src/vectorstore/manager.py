"""
High-level manager for vector store operations.

VectorStoreManager orchestrates embedding generation, authority score
computation, and vector storage in a single high-level API.
"""

import math
import structlog
from datetime import datetime, timedelta, timezone
from typing import Any

from src.embedding.service import EmbeddingService, ModelType
from src.ingestion.schemas import NormalizedDocument
from src.vectorstore.base import VectorSearchFilter, VectorSearchResult, VectorStore
from src.vectorstore.config import VectorStoreConfig

logger = structlog.get_logger(__name__)


class VectorStoreManager:
    """
    High-level orchestration for vector operations.

    Combines EmbeddingService and VectorStore to provide a unified API for:
    - Ingesting documents (embed + compute authority + store)
    - Querying by text (embed query + search)
    - Querying by embedding or centroid

    This is the primary interface for application code to use.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        config: VectorStoreConfig | None = None,
    ):
        """
        Initialize the manager.

        Args:
            vector_store: VectorStore implementation for storage/search
            embedding_service: Service for generating embeddings
            config: Optional configuration
        """
        self._store = vector_store
        self._embedding = embedding_service
        self._config = config or VectorStoreConfig()

    async def ingest_documents(
        self,
        documents: list[NormalizedDocument],
        compute_authority: bool = True,
    ) -> dict[str, Any]:
        """
        Ingest documents by generating embeddings and storing them.

        This method:
        1. Filters documents that already have embeddings (optional)
        2. Generates FinBERT embeddings for document content
        3. Computes authority scores
        4. Upserts embeddings to the vector store

        Args:
            documents: Documents to ingest
            compute_authority: Whether to compute authority scores

        Returns:
            Stats dict with processed/skipped/error counts
        """
        if not documents:
            return {"processed": 0, "skipped": 0, "errors": 0}

        stats = {"processed": 0, "skipped": 0, "errors": 0}

        # Filter documents needing embeddings
        docs_to_embed = [doc for doc in documents if doc.embedding is None]
        stats["skipped"] = len(documents) - len(docs_to_embed)

        if not docs_to_embed:
            logger.info("All documents already have embeddings, skipping")
            return stats

        try:
            # Extract content for embedding
            texts = [self._get_document_text(doc) for doc in docs_to_embed]

            # Generate embeddings using FinBERT (768-dim)
            embeddings = await self._embedding.embed_batch(
                texts,
                model_type=ModelType.FINBERT,
                show_progress=True,
            )

            # Compute authority scores if requested
            if compute_authority:
                for doc in docs_to_embed:
                    doc.authority_score = self._compute_authority_score(doc)

            # Prepare for upsert
            ids = [doc.id for doc in docs_to_embed]
            metadata = [self._document_to_metadata(doc) for doc in docs_to_embed]

            # Upsert to vector store
            upserted = await self._store.upsert(ids, embeddings, metadata)
            stats["processed"] = upserted

            logger.info(
                "Ingested documents",
                processed=upserted,
                skipped=stats["skipped"],
            )

        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            stats["errors"] = len(docs_to_embed)
            raise

        return stats

    async def query(
        self,
        text: str,
        limit: int | None = None,
        threshold: float | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for documents similar to a text query.

        Generates an embedding for the query text and performs
        similarity search in the vector store.

        Args:
            text: Query text to find similar documents
            limit: Maximum results (default from config)
            threshold: Minimum similarity (default from config)
            filters: Optional filter criteria

        Returns:
            List of search results sorted by similarity
        """
        if not text.strip():
            return []

        # Use defaults from config if not specified
        limit = limit or self._config.default_limit
        threshold = threshold or self._config.default_threshold

        # Generate query embedding
        query_embedding = await self._embedding.embed_finbert(text)

        # Search vector store
        return await self._store.search(
            query_embedding=query_embedding,
            limit=limit,
            threshold=threshold,
            filters=filters,
        )

    async def query_by_embedding(
        self,
        embedding: list[float],
        limit: int | None = None,
        threshold: float | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search using a pre-computed embedding.

        Useful when you already have an embedding (e.g., from a cached query
        or another document).

        Args:
            embedding: Pre-computed 768-dimensional embedding
            limit: Maximum results
            threshold: Minimum similarity
            filters: Optional filter criteria

        Returns:
            List of search results
        """
        limit = limit or self._config.default_limit
        threshold = threshold or self._config.default_threshold

        return await self._store.search(
            query_embedding=embedding,
            limit=limit,
            threshold=threshold,
            filters=filters,
        )

    async def query_by_theme_centroid(
        self,
        centroid: list[float],
        limit: int | None = None,
        threshold: float | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Find documents belonging to a theme cluster.

        Uses centroid search with relaxed threshold defaults,
        intended for BERTopic or other clustering applications.

        Args:
            centroid: Theme/cluster centroid embedding
            limit: Maximum results (default higher for clusters)
            threshold: Minimum similarity (default lower for centroids)
            filters: Optional filter criteria

        Returns:
            List of documents in the cluster
        """
        limit = limit or self._config.centroid_default_limit
        threshold = threshold or self._config.centroid_default_threshold

        return await self._store.search_by_centroid(
            centroid=centroid,
            limit=limit,
            threshold=threshold,
            filters=filters,
        )

    async def cleanup_old_documents(self, days_to_keep: int = 90) -> int:
        """
        Remove documents older than the specified threshold.

        Used for storage management to prevent unbounded growth.
        Documents with timestamp before (now - days_to_keep) will be deleted.

        Args:
            days_to_keep: Number of days of data to retain (default 90)

        Returns:
            Number of documents deleted

        Raises:
            ValueError: If days_to_keep is not positive
        """
        if days_to_keep <= 0:
            raise ValueError(f"days_to_keep must be positive, got {days_to_keep}")

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        deleted = await self._store.delete_before_timestamp(cutoff)

        logger.info(
            "Cleanup complete",
            deleted=deleted,
            days_to_keep=days_to_keep,
            cutoff=cutoff.isoformat(),
        )
        return deleted

    def _compute_authority_score(self, doc: NormalizedDocument) -> float:
        """
        Compute authority score for a document.

        Score components (total max = 1.0):
        - Verified author: +0.2
        - Follower count (log-scaled): max +0.3
        - Engagement (log-scaled): max +0.3
        - Inverse spam score: +0.2 * (1 - spam_score)

        Args:
            doc: Document to score

        Returns:
            Authority score between 0.0 and 1.0
        """
        score = 0.0

        # Verified bonus
        if doc.author_verified:
            score += self._config.authority_verified_bonus

        # Follower score (log-scaled)
        if doc.author_followers and doc.author_followers > 0:
            follower_score = (
                self._config.authority_follower_max
                * math.log(doc.author_followers + 1)
                / math.log(self._config.authority_follower_scale + 1)
            )
            score += min(self._config.authority_follower_max, follower_score)

        # Engagement score (log-scaled)
        engagement_value = (
            doc.engagement.likes
            + doc.engagement.shares * 2  # Shares weighted more
            + doc.engagement.comments
        )
        if engagement_value > 0:
            engagement_score = (
                self._config.authority_engagement_max
                * math.log(engagement_value + 1)
                / math.log(self._config.authority_engagement_scale + 1)
            )
            score += min(self._config.authority_engagement_max, engagement_score)

        # Inverse spam penalty
        spam_penalty = self._config.authority_spam_penalty_max * (1.0 - doc.spam_score)
        score += spam_penalty

        # Clamp to valid range
        return max(0.0, min(1.0, score))

    def _get_document_text(self, doc: NormalizedDocument) -> str:
        """Extract text content for embedding."""
        # Combine title and content for richer embeddings
        if doc.title:
            return f"{doc.title} {doc.content}"
        return doc.content

    def _document_to_metadata(self, doc: NormalizedDocument) -> dict[str, Any]:
        """Convert document to metadata dict for storage."""
        return {
            "platform": doc.platform.value if hasattr(doc.platform, "value") else doc.platform,
            "url": doc.url,
            "title": doc.title,
            "author_name": doc.author_name,
            "author_verified": doc.author_verified,
            "author_followers": doc.author_followers,
            "tickers": doc.tickers_mentioned,
            "theme_ids": doc.theme_ids,
            "spam_score": doc.spam_score,
            "authority_score": doc.authority_score,
            "timestamp": doc.timestamp.isoformat() if doc.timestamp else None,
        }
