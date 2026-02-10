"""
Embedding worker - consumes embedding jobs and updates documents.

Runs as a standalone service that:
1. Consumes document IDs from the embedding queue
2. Fetches document content from PostgreSQL
3. Generates embeddings using FinBERT
4. Updates documents with embeddings
"""

import asyncio
import logging
import time
from typing import Any

import redis.asyncio as redis
import structlog

from src.clustering.queue import ClusteringQueue
from src.config.settings import get_settings
from src.embedding.config import EmbeddingConfig
from src.embedding.queue import EmbeddingJob, EmbeddingQueue
from src.embedding.service import EmbeddingService, ModelType
from src.ingestion.schemas import Platform
from src.observability.metrics import get_metrics
from src.storage.database import Database
from src.storage.repository import DocumentRepository

logger = structlog.get_logger(__name__)


class EmbeddingWorker:
    """
    Worker that processes embedding jobs from the queue.

    Consumes document IDs from Redis Streams, fetches documents from
    PostgreSQL, generates embeddings, and updates the database.

    Features:
    - Batch processing for efficiency
    - Graceful shutdown with drain
    - Error handling with DLQ
    - Metrics collection

    Usage:
        worker = EmbeddingWorker()
        await worker.start()  # Runs until stopped
    """

    def __init__(
        self,
        queue: EmbeddingQueue | None = None,
        database: Database | None = None,
        embedding_service: EmbeddingService | None = None,
        config: EmbeddingConfig | None = None,
        batch_size: int | None = None,
        clustering_queue: ClusteringQueue | None = None,
    ):
        """
        Initialize the embedding worker.

        Args:
            queue: Embedding job queue (or create from config)
            database: Database connection (or create from config)
            embedding_service: Embedding service (or create from config)
            config: Embedding configuration
            batch_size: Jobs to process per batch (uses config default)
            clustering_queue: Clustering queue for downstream enqueue (or create if enabled)
        """
        self._config = config or EmbeddingConfig()
        self._queue = queue or EmbeddingQueue(config=self._config)
        self._database = database or Database()

        # Create embedding service with shared Redis for caching
        self._embedding_service = embedding_service

        self._batch_size = batch_size or self._config.batch_size

        self._repository: DocumentRepository | None = None
        self._redis: redis.Redis | None = None
        self._running = False
        self._metrics = get_metrics()

        # Clustering queue: only used when clustering is enabled
        self._clustering_enabled = get_settings().clustering_enabled
        self._clustering_queue = clustering_queue

        logger.info(
            "EmbeddingWorker initialized",
            batch_size=self._batch_size,
            clustering_enabled=self._clustering_enabled,
        )

    async def start(self) -> None:
        """
        Start the embedding worker.

        Runs until stop() is called or a fatal error occurs.
        """
        self._running = True
        settings = get_settings()

        logger.info("Starting embedding worker")

        # Connect to dependencies
        await self._queue.connect()
        await self._database.connect()

        # Create Redis client for caching
        self._redis = redis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )

        # Initialize embedding service with caching
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(
                config=self._config,
                redis_client=self._redis,
            )

        # Create repository
        self._repository = DocumentRepository(self._database)

        # Connect clustering queue if enabled
        if self._clustering_enabled:
            if self._clustering_queue is None:
                self._clustering_queue = ClusteringQueue()
            await self._clustering_queue.connect()
            logger.info("Clustering queue connected")

        try:
            # Process jobs from queue
            await self._process_loop()

        except asyncio.CancelledError:
            logger.info("Embedding worker cancelled")
        except Exception as e:
            logger.error("Embedding worker error", error=str(e))
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the embedding worker gracefully."""
        logger.info("Stopping embedding worker")
        self._running = False

    async def _cleanup(self) -> None:
        """Clean up resources."""
        await self._queue.close()
        await self._database.close()
        if self._redis:
            await self._redis.close()
        if self._embedding_service:
            await self._embedding_service.close()
        if self._clustering_queue:
            await self._clustering_queue.close()
        logger.info("Embedding worker cleaned up")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        batch: list[EmbeddingJob] = []
        batch_start = time.monotonic()
        batch_timeout = self._config.worker_batch_timeout

        async for job in self._queue.consume(
            count=self._batch_size,
            block_ms=5000,
        ):
            if not self._running:
                break

            batch.append(job)

            # Process batch when full or timeout
            if (
                len(batch) >= self._batch_size
                or (time.monotonic() - batch_start) > batch_timeout
            ):
                await self._process_batch(batch)
                batch = []
                batch_start = time.monotonic()

        # Process remaining jobs
        if batch:
            await self._process_batch(batch)

    def _select_model(self, platform: Platform | str, content_length: int) -> ModelType:
        """
        Select embedding model based on platform and content length.

        Strategy:
        - Twitter + <300 chars → MiniLM (fast, lightweight)
        - Everything else → FinBERT (financial domain expertise)
        """
        platform_value = platform.value if isinstance(platform, Platform) else platform
        if platform_value == "twitter" and content_length < 300:
            return ModelType.MINILM
        return ModelType.FINBERT

    async def _process_batch(self, batch: list[EmbeddingJob]) -> None:
        """
        Process a batch of embedding jobs with model selection.

        Args:
            batch: List of embedding jobs
        """
        if not batch:
            return

        start_time = time.monotonic()
        processed = 0
        skipped = 0
        errors = 0

        # Fetch documents from database
        doc_ids = [job.document_id for job in batch]
        documents = await self._fetch_documents(doc_ids)

        # Map documents by ID for easy lookup
        doc_map = {doc.id: doc for doc in documents}

        # Group jobs by model type for efficient batch processing
        finbert_jobs: list[tuple[EmbeddingJob, str]] = []
        minilm_jobs: list[tuple[EmbeddingJob, str]] = []

        for job in batch:
            doc = doc_map.get(job.document_id)
            if doc is None:
                logger.warning(
                    "Document not found for embedding",
                    document_id=job.document_id,
                )
                skipped += 1
                await self._queue.ack(job.message_id)
                continue

            # Combine title and content for embedding
            text = doc.content
            if doc.title:
                text = f"{doc.title}. {text}"

            # Select model based on platform and content length
            model_type = self._select_model(doc.platform, len(doc.content))

            # Skip if document already has the appropriate embedding
            if model_type == ModelType.MINILM:
                if doc.embedding_minilm is not None:
                    logger.debug(
                        "Document already has MiniLM embedding",
                        document_id=job.document_id,
                    )
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue
                minilm_jobs.append((job, text))
            else:
                if doc.embedding is not None:
                    logger.debug(
                        "Document already has FinBERT embedding",
                        document_id=job.document_id,
                    )
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue
                finbert_jobs.append((job, text))

        # Process FinBERT batch
        if finbert_jobs:
            processed += await self._process_model_batch(
                finbert_jobs, ModelType.FINBERT, doc_map
            )

        # Process MiniLM batch
        if minilm_jobs:
            processed += await self._process_model_batch(
                minilm_jobs, ModelType.MINILM, doc_map
            )

        # Record metrics
        elapsed = time.monotonic() - start_time
        self._metrics.record_embedding_batch(
            processed=processed,
            skipped=skipped,
            errors=errors,
            latency=elapsed,
        )

        # Report queue depth after each batch
        try:
            depth = await self._queue.get_stream_length()
            self._metrics.set_embedding_queue_depth(depth)
        except Exception:
            pass  # non-critical: queue depth is best-effort

        logger.info(
            "Embedding batch processed",
            total=len(batch),
            processed=processed,
            skipped=skipped,
            finbert_count=len(finbert_jobs),
            minilm_count=len(minilm_jobs),
            elapsed_seconds=round(elapsed, 2),
        )

    async def _process_model_batch(
        self,
        jobs: list[tuple[EmbeddingJob, str]],
        model_type: ModelType,
        doc_map: dict,
    ) -> int:
        """Process a batch of jobs for a specific model type."""
        if not jobs:
            return 0

        processed = 0

        try:
            texts = [text for _, text in jobs]
            embeddings = await self._embedding_service.embed_batch(
                texts, model_type=model_type, show_progress=True
            )

            # Update documents with embeddings
            for (job, _), embedding in zip(jobs, embeddings):
                try:
                    if model_type == ModelType.MINILM:
                        success = await self._repository.update_embedding_minilm(
                            job.document_id, embedding
                        )
                    else:
                        success = await self._repository.update_embedding(
                            job.document_id, embedding
                        )

                    if success:
                        processed += 1
                        self._metrics.record_embedding_generated(
                            doc_map[job.document_id].platform,
                            model=model_type.value,
                        )

                        # Enqueue for clustering if enabled
                        if self._clustering_enabled and self._clustering_queue:
                            try:
                                await self._clustering_queue.publish(
                                    job.document_id, model_type.value
                                )
                            except Exception as enqueue_err:
                                logger.warning(
                                    "Failed to enqueue for clustering",
                                    document_id=job.document_id,
                                    error=str(enqueue_err),
                                )
                    else:
                        logger.error(
                            "Failed to update embedding",
                            document_id=job.document_id,
                            model=model_type.value,
                        )

                    await self._queue.ack(job.message_id)

                except Exception as e:
                    logger.error(
                        "Error updating embedding",
                        document_id=job.document_id,
                        model=model_type.value,
                        error=str(e),
                    )
                    await self._queue.nack(job.message_id, str(e))

        except Exception as e:
            # Batch embedding failed - nack all jobs
            logger.error(
                "Batch embedding failed",
                model=model_type.value,
                error=str(e),
            )
            for job, _ in jobs:
                await self._queue.nack(job.message_id, str(e))

        return processed

    async def _fetch_documents(self, doc_ids: list[str]) -> list[Any]:
        """Fetch documents from database by IDs."""
        documents = []
        for doc_id in doc_ids:
            doc = await self._repository.get_by_id(doc_id)
            if doc:
                documents.append(doc)
        return documents

    async def run_once(
        self,
        doc_ids: list[str],
    ) -> dict[str, int]:
        """
        Process a list of document IDs without queue.

        Useful for testing or backfilling embeddings.

        Args:
            doc_ids: Document IDs to process

        Returns:
            Processing statistics
        """
        settings = get_settings()
        await self._database.connect()

        self._redis = redis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )

        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(
                config=self._config,
                redis_client=self._redis,
            )

        self._repository = DocumentRepository(self._database)

        stats = {
            "total": len(doc_ids),
            "processed": 0,
            "skipped": 0,
            "errors": 0,
        }

        try:
            documents = await self._fetch_documents(doc_ids)
            doc_map = {doc.id: doc for doc in documents}

            for doc_id in doc_ids:
                try:
                    doc = doc_map.get(doc_id)
                    if doc is None:
                        stats["skipped"] += 1
                        continue

                    # Select model and check if already has embedding
                    model_type = self._select_model(doc.platform, len(doc.content))

                    if model_type == ModelType.MINILM:
                        if doc.embedding_minilm is not None:
                            stats["skipped"] += 1
                            continue
                    else:
                        if doc.embedding is not None:
                            stats["skipped"] += 1
                            continue

                    # Generate embedding
                    text = doc.content
                    if doc.title:
                        text = f"{doc.title}. {text}"

                    embedding = await self._embedding_service.embed(text, model_type)

                    # Update document with appropriate column
                    if model_type == ModelType.MINILM:
                        await self._repository.update_embedding_minilm(doc_id, embedding)
                    else:
                        await self._repository.update_embedding(doc_id, embedding)
                    stats["processed"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    logger.error(
                        "Error processing document",
                        document_id=doc_id,
                        error=str(e),
                    )

        finally:
            await self._database.close()
            if self._redis:
                await self._redis.close()

        return stats

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of the embedding worker.

        Returns:
            Dictionary with health status
        """
        queue_healthy = await self._queue.health_check()
        db_healthy = await self._database.health_check()

        return {
            "running": self._running,
            "queue_healthy": queue_healthy,
            "database_healthy": db_healthy,
            "embedding_service_stats": (
                self._embedding_service.get_stats()
                if self._embedding_service
                else None
            ),
        }
