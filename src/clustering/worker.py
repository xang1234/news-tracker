"""
Clustering worker - assigns documents to themes via pgvector similarity.

Runs as a standalone service that:
1. Consumes document IDs from the clustering queue (Redis Stream)
2. Fetches document embeddings from PostgreSQL
3. Finds similar theme centroids via pgvector HNSW search
4. Assigns documents to matching themes
5. Updates theme centroids via EMA (exponential moving average)
6. Increments theme document_count atomically

Unlike BERTopicService (batch discovery), this worker handles real-time
per-document assignment using pre-computed centroids in the themes table.
"""

import asyncio
import time
from typing import Any

import numpy as np
import redis.asyncio as redis
import structlog

from src.clustering.config import ClusteringConfig
from src.clustering.queue import ClusteringJob, ClusteringQueue
from src.config.settings import get_settings
from src.observability.metrics import get_metrics
from src.storage.database import Database
from src.storage.repository import DocumentRepository
from src.themes.repository import ThemeRepository

logger = structlog.get_logger(__name__)

# Idempotency key prefix and TTL (7 days)
_IDEMPOTENCY_PREFIX = "idempotent:cluster"
_IDEMPOTENCY_TTL = 7 * 24 * 3600


class ClusteringWorker:
    """
    Worker that assigns documents to themes in real time.

    Consumes document IDs from the clustering Redis Stream, looks up
    their FinBERT embeddings, performs pgvector HNSW similarity search
    against theme centroids, and updates both the document (theme_ids)
    and the theme (centroid EMA, document_count).

    Features:
    - Batch processing with configurable timeout
    - Idempotency via Redis SET NX keys
    - Atomic document_count increment (no read-modify-write race)
    - EMA centroid updates for theme drift tracking
    - Skips MiniLM-only documents (theme centroids are 768-dim FinBERT)
    - Graceful shutdown with drain

    Usage:
        worker = ClusteringWorker()
        await worker.start()  # Runs until stopped
    """

    def __init__(
        self,
        queue: ClusteringQueue | None = None,
        database: Database | None = None,
        config: ClusteringConfig | None = None,
        batch_size: int | None = None,
    ):
        """
        Initialize the clustering worker.

        Args:
            queue: Clustering job queue (or create from config)
            database: Database connection (or create from config)
            config: Clustering configuration
            batch_size: Jobs to process per batch (default: 32)
        """
        self._config = config or ClusteringConfig()
        self._queue = queue or ClusteringQueue(config=self._config)
        self._database = database or Database()
        self._batch_size = batch_size or 32

        self._doc_repo: DocumentRepository | None = None
        self._theme_repo: ThemeRepository | None = None
        self._redis: redis.Redis | None = None
        self._running = False

        logger.info(
            "ClusteringWorker initialized",
            batch_size=self._batch_size,
        )

    async def start(self) -> None:
        """
        Start the clustering worker.

        Connects to Redis, PostgreSQL, and the clustering queue,
        then enters the processing loop until stop() is called.
        """
        self._running = True
        settings = get_settings()

        logger.info("Starting clustering worker")

        # Connect to dependencies
        await self._queue.connect()
        await self._database.connect()

        self._redis = redis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )

        # Create repositories
        self._doc_repo = DocumentRepository(self._database)
        self._theme_repo = ThemeRepository(self._database)

        try:
            await self._process_loop()
        except asyncio.CancelledError:
            logger.info("Clustering worker cancelled")
        except Exception as e:
            logger.error("Clustering worker error", error=str(e))
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the clustering worker gracefully."""
        logger.info("Stopping clustering worker")
        self._running = False

    async def _cleanup(self) -> None:
        """Clean up resources."""
        await self._queue.close()
        await self._database.close()
        if self._redis:
            await self._redis.close()
        logger.info("Clustering worker cleaned up")

    async def _process_loop(self) -> None:
        """Main processing loop — accumulate batches, then process."""
        batch: list[ClusteringJob] = []
        batch_start = time.monotonic()
        batch_timeout = self._config.worker_batch_timeout
        metrics = get_metrics()

        async for job in self._queue.consume(
            count=self._batch_size,
            block_ms=5000,
        ):
            if not self._running:
                break

            batch.append(job)

            # Process when batch is full or timeout exceeded
            if (
                len(batch) >= self._batch_size
                or (time.monotonic() - batch_start) > batch_timeout
            ):
                await self._process_batch(batch)
                batch = []
                batch_start = time.monotonic()

                # Update queue depth metric
                try:
                    pending = await self._queue.get_pending_count()
                    metrics.set_clustering_queue_depth(pending)
                except Exception:
                    pass  # Don't fail on metrics

        # Drain remaining
        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[ClusteringJob]) -> None:
        """
        Process a batch of clustering jobs.

        For each job: check idempotency → fetch doc → get embedding →
        find_similar themes → update document theme_ids → EMA centroid →
        atomic document_count increment → ack.
        """
        if not batch:
            return

        start_time = time.monotonic()
        processed = 0
        skipped = 0
        errors = 0
        metrics = get_metrics()

        for job in batch:
            try:
                # --- Idempotency check ---
                idem_key = f"{_IDEMPOTENCY_PREFIX}:{job.document_id}:{job.embedding_model}"
                was_set = await self._redis.set(
                    idem_key, "1", nx=True, ex=_IDEMPOTENCY_TTL
                )
                if not was_set:
                    logger.debug(
                        "Clustering job already processed",
                        document_id=job.document_id,
                    )
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue

                # --- Fetch document ---
                doc = await self._doc_repo.get_by_id(job.document_id)
                if doc is None:
                    logger.warning(
                        "Document not found for clustering",
                        document_id=job.document_id,
                    )
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue

                # --- Get FinBERT embedding ---
                embedding = self._get_embedding(doc, job.embedding_model)
                if embedding is None:
                    logger.warning(
                        "No suitable embedding for clustering",
                        document_id=job.document_id,
                        embedding_model=job.embedding_model,
                    )
                    # Clear idempotency key so it can be retried when
                    # the embedding becomes available
                    await self._redis.delete(idem_key)
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue

                # --- Skip MiniLM-only docs (centroids are 768-dim) ---
                if job.embedding_model == "minilm":
                    logger.debug(
                        "Skipping MiniLM-embedded document",
                        document_id=job.document_id,
                    )
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue

                # --- Find similar themes via pgvector HNSW ---
                embedding_array = np.array(embedding, dtype=np.float32)
                similar = await self._theme_repo.find_similar(
                    centroid=embedding_array,
                    limit=1,
                    threshold=self._config.similarity_threshold_assign,
                )

                if not similar:
                    logger.debug(
                        "No matching theme for document",
                        document_id=job.document_id,
                    )
                    # Record the best similarity even for non-matches
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue

                theme, similarity = similar[0]

                # Record similarity distribution
                metrics.clustering_similarity.observe(similarity)

                # --- Assign document to theme ---
                await self._doc_repo.update_themes(
                    job.document_id, [theme.theme_id]
                )

                # --- EMA centroid update ---
                new_centroid = self._ema_centroid_update(
                    theme.centroid,
                    embedding_array,
                    self._config.centroid_learning_rate,
                )
                await self._theme_repo.update_centroid(
                    theme.theme_id, new_centroid
                )

                # --- Atomic document_count increment ---
                await self._database.execute(
                    "UPDATE themes SET document_count = document_count + 1 "
                    "WHERE theme_id = $1",
                    theme.theme_id,
                )

                processed += 1
                metrics.record_clustering_assigned(
                    platform=doc.platform,
                )

                logger.debug(
                    "Document assigned to theme",
                    document_id=job.document_id,
                    theme_id=theme.theme_id,
                    similarity=round(similarity, 4),
                )

                await self._queue.ack(job.message_id)

            except Exception as e:
                logger.error(
                    "Error processing clustering job",
                    document_id=job.document_id,
                    error=str(e),
                )
                errors += 1
                metrics.record_clustering_error(type(e).__name__)
                await self._queue.nack(job.message_id, str(e))

        # Batch metrics
        elapsed = time.monotonic() - start_time
        metrics.record_clustering_batch(
            processed=processed,
            skipped=skipped,
            errors=errors,
            latency=elapsed,
        )

        logger.info(
            "Clustering batch processed",
            total=len(batch),
            processed=processed,
            skipped=skipped,
            errors=errors,
            elapsed_seconds=round(elapsed, 2),
        )

    @staticmethod
    def _get_embedding(doc: Any, embedding_model: str) -> list[float] | None:
        """
        Extract the appropriate embedding from a document.

        Args:
            doc: NormalizedDocument instance
            embedding_model: Model name ("finbert" or "minilm")

        Returns:
            Embedding vector or None if not available.
        """
        if embedding_model == "finbert":
            return doc.embedding
        if embedding_model == "minilm":
            return doc.embedding_minilm
        # Default to FinBERT
        return doc.embedding

    @staticmethod
    def _ema_centroid_update(
        centroid: np.ndarray,
        embedding: np.ndarray,
        learning_rate: float,
    ) -> np.ndarray:
        """
        Exponential moving average update for a theme centroid.

        Args:
            centroid: Current centroid vector.
            embedding: New document embedding.
            learning_rate: EMA learning rate (0 < lr <= 1).

        Returns:
            Updated centroid as float32 ndarray.
        """
        return (
            (1.0 - learning_rate) * centroid + learning_rate * embedding
        ).astype(np.float32)

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of the clustering worker.

        Returns:
            Dictionary with health status.
        """
        queue_healthy = await self._queue.health_check()
        db_healthy = await self._database.health_check()

        return {
            "running": self._running,
            "queue_healthy": queue_healthy,
            "database_healthy": db_healthy,
        }
