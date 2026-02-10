"""
Sentiment worker - consumes sentiment jobs and updates documents.

Runs as a standalone service that:
1. Consumes document IDs from the sentiment queue
2. Fetches document content and entities from PostgreSQL
3. Generates sentiment using FinBERT
4. Updates documents with sentiment results
"""

import asyncio
import time
from typing import Any

import redis.asyncio as redis
import structlog

from src.config.settings import get_settings
from src.observability.metrics import get_metrics
from src.queues.backoff import ExponentialBackoff
from src.sentiment.config import SentimentConfig
from src.sentiment.queue import SentimentJob, SentimentQueue
from src.sentiment.service import SentimentService
from src.storage.database import Database
from src.storage.repository import DocumentRepository

logger = structlog.get_logger(__name__)


class SentimentWorker:
    """
    Worker that processes sentiment jobs from the queue.

    Consumes document IDs from Redis Streams, fetches documents from
    PostgreSQL, generates sentiment analysis, and updates the database.

    Features:
    - Batch processing for efficiency
    - Graceful shutdown with drain
    - Error handling with DLQ
    - Entity-level sentiment when NER data available
    - Skip documents that already have sentiment

    Usage:
        worker = SentimentWorker()
        await worker.start()  # Runs until stopped
    """

    def __init__(
        self,
        queue: SentimentQueue | None = None,
        database: Database | None = None,
        sentiment_service: SentimentService | None = None,
        config: SentimentConfig | None = None,
        batch_size: int | None = None,
    ):
        """
        Initialize the sentiment worker.

        Args:
            queue: Sentiment job queue (or create from config)
            database: Database connection (or create from config)
            sentiment_service: Sentiment service (or create from config)
            config: Sentiment configuration
            batch_size: Jobs to process per batch (uses config default)
        """
        self._config = config or SentimentConfig()
        self._queue = queue or SentimentQueue(config=self._config)
        self._database = database or Database()

        # Create sentiment service with shared Redis for caching
        self._sentiment_service = sentiment_service

        self._batch_size = batch_size or self._config.batch_size

        self._repository: DocumentRepository | None = None
        self._redis: redis.Redis | None = None
        self._running = False

        logger.info(
            "SentimentWorker initialized",
            batch_size=self._batch_size,
        )

    async def _connect_dependencies(self) -> None:
        """Connect to all external dependencies (Redis, DB, queues)."""
        settings = get_settings()

        await self._queue.connect()
        await self._database.connect()

        # Create Redis client for caching
        self._redis = redis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )

        # Initialize sentiment service with caching
        if self._sentiment_service is None:
            self._sentiment_service = SentimentService(
                config=self._config,
                redis_client=self._redis,
            )

        # Create repository
        self._repository = DocumentRepository(self._database)

    async def start(self) -> None:
        """
        Start the sentiment worker with supervised retry loop.

        Automatically reconnects on transient failures using exponential
        backoff. Exits after max_consecutive_failures or on CancelledError.
        """
        self._running = True
        settings = get_settings()
        backoff = ExponentialBackoff(
            base_delay=settings.worker_backoff_base_delay,
            max_delay=settings.worker_backoff_max_delay,
        )

        logger.info("Starting sentiment worker")

        while self._running:
            try:
                await self._connect_dependencies()
                await self._process_loop()
                if not self._running:
                    break
            except asyncio.CancelledError:
                logger.info("Sentiment worker cancelled")
                break
            except Exception as e:
                if backoff.attempt >= settings.worker_max_consecutive_failures:
                    logger.error(
                        "Sentiment worker exceeded max consecutive failures",
                        failures=backoff.attempt,
                        error=str(e),
                    )
                    raise
                delay = backoff.next_delay()
                logger.warning(
                    "Sentiment worker error, retrying",
                    error=str(e),
                    attempt=backoff.attempt,
                    retry_delay=round(delay, 1),
                )
                await self._cleanup()
                await asyncio.sleep(delay)
            else:
                backoff.reset()

        await self._cleanup()

    async def stop(self) -> None:
        """Stop the sentiment worker gracefully."""
        logger.info("Stopping sentiment worker")
        self._running = False

    async def _cleanup(self) -> None:
        """Clean up resources."""
        await self._queue.close()
        await self._database.close()
        if self._redis:
            await self._redis.close()
        if self._sentiment_service:
            await self._sentiment_service.close()
        logger.info("Sentiment worker cleaned up")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        batch: list[SentimentJob] = []
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

            # Process batch when full or timeout
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
                    metrics.set_sentiment_queue_depth(pending)
                except Exception:
                    pass  # Don't fail on metrics

        # Process remaining jobs
        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[SentimentJob]) -> None:
        """
        Process a batch of sentiment jobs.

        Args:
            batch: List of sentiment jobs
        """
        if not batch:
            return

        start_time = time.monotonic()
        processed = 0
        skipped = 0
        errors = 0
        metrics = get_metrics()

        # Fetch documents from database
        doc_ids = [job.document_id for job in batch]
        documents = await self._fetch_documents(doc_ids)

        # Map documents by ID for easy lookup
        doc_map = {doc.id: doc for doc in documents}

        for job in batch:
            try:
                doc = doc_map.get(job.document_id)
                if doc is None:
                    logger.warning(
                        "Document not found for sentiment",
                        document_id=job.document_id,
                    )
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue

                # Skip if document already has sentiment
                if doc.sentiment is not None:
                    logger.debug(
                        "Document already has sentiment",
                        document_id=job.document_id,
                    )
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue

                # Skip if document has no content
                if not doc.content or not doc.content.strip():
                    logger.warning(
                        "Document has empty content",
                        document_id=job.document_id,
                    )
                    skipped += 1
                    await self._queue.ack(job.message_id)
                    continue

                # Combine title and content for analysis
                text = doc.content
                if doc.title:
                    text = f"{doc.title}. {text}"

                # Check if document has entities for entity-level sentiment
                if doc.entities_mentioned and self._config.enable_entity_sentiment:
                    result = await self._sentiment_service.analyze_with_entities(
                        text, doc.entities_mentioned
                    )
                else:
                    # Document-level only
                    result = await self._sentiment_service.analyze(text)

                # Update document with sentiment
                success = await self._repository.update_sentiment(
                    job.document_id, result
                )

                if success:
                    processed += 1
                    # Record sentiment metrics
                    metrics.record_sentiment_analyzed(
                        platform=doc.platform,
                        label=result["label"],
                        confidence=result["confidence"],
                    )
                    logger.debug(
                        "Sentiment updated",
                        document_id=job.document_id,
                        label=result["label"],
                        confidence=result["confidence"],
                    )
                else:
                    logger.error(
                        "Failed to update sentiment",
                        document_id=job.document_id,
                    )
                    errors += 1
                    metrics.record_sentiment_error("db_update_failed")

                await self._queue.ack(job.message_id)

            except Exception as e:
                logger.error(
                    "Error processing sentiment job",
                    document_id=job.document_id,
                    error=str(e),
                )
                errors += 1
                metrics.record_sentiment_error(type(e).__name__)
                await self._queue.nack(job.message_id, str(e))

        # Log batch results
        elapsed = time.monotonic() - start_time

        # Record batch metrics
        metrics.record_sentiment_batch(
            processed=processed,
            skipped=skipped,
            errors=errors,
            latency=elapsed,
        )

        logger.info(
            "Sentiment batch processed",
            total=len(batch),
            processed=processed,
            skipped=skipped,
            errors=errors,
            elapsed_seconds=round(elapsed, 2),
        )

    async def _fetch_documents(self, doc_ids: list[str]) -> list[Any]:
        """Fetch documents from database by IDs."""
        if not self._repository:
            return []

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

        Useful for testing or backfilling sentiment.

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

        if self._sentiment_service is None:
            self._sentiment_service = SentimentService(
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

                    # Skip if already has sentiment
                    if doc.sentiment is not None:
                        stats["skipped"] += 1
                        continue

                    # Analyze
                    text = doc.content
                    if doc.title:
                        text = f"{doc.title}. {text}"

                    if doc.entities_mentioned and self._config.enable_entity_sentiment:
                        result = await self._sentiment_service.analyze_with_entities(
                            text, doc.entities_mentioned
                        )
                    else:
                        result = await self._sentiment_service.analyze(text)

                    # Update
                    await self._repository.update_sentiment(doc_id, result)
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
        Check health of the sentiment worker.

        Returns:
            Dictionary with health status
        """
        queue_healthy = await self._queue.health_check()
        db_healthy = await self._database.health_check()

        return {
            "running": self._running,
            "queue_healthy": queue_healthy,
            "database_healthy": db_healthy,
            "sentiment_service_stats": (
                self._sentiment_service.get_stats()
                if self._sentiment_service
                else None
            ),
        }
