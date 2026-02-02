"""
Processing service - consumes documents from queue and processes them.

Runs the preprocessing pipeline (spam detection, deduplication, ticker
extraction) and stores processed documents in PostgreSQL.

Features:
- Consumer group scaling
- Graceful shutdown with drain
- Dead letter queue for failures
- Metrics collection
"""

import asyncio
import logging
import time
from typing import Any

import structlog

from src.config.settings import get_settings
from src.ingestion.deduplication import Deduplicator
from src.ingestion.preprocessor import Preprocessor
from src.ingestion.queue import DocumentQueue, QueueMessage
from src.ingestion.schemas import NormalizedDocument
from src.observability.metrics import get_metrics
from src.storage.database import Database
from src.storage.repository import DocumentRepository

logger = structlog.get_logger(__name__)


class ProcessingService:
    """
    Service that processes documents from the queue.

    Consumes documents from Redis Streams, runs them through the
    preprocessing pipeline, and stores valid documents in PostgreSQL.

    Pipeline stages:
    1. Preprocessing (spam detection, ticker extraction)
    2. Deduplication (MinHash LSH)
    3. Storage (PostgreSQL with upsert)

    Usage:
        service = ProcessingService()
        await service.start()  # Runs until stopped
    """

    def __init__(
        self,
        queue: DocumentQueue | None = None,
        database: Database | None = None,
        preprocessor: Preprocessor | None = None,
        deduplicator: Deduplicator | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize processing service.

        Args:
            queue: Document queue (or create from config)
            database: Database connection (or create from config)
            preprocessor: Document preprocessor (or create default)
            deduplicator: Deduplication index (or create default)
            batch_size: Documents to process per batch
        """
        self._queue = queue or DocumentQueue()
        self._database = database or Database()
        self._preprocessor = preprocessor or Preprocessor()
        self._deduplicator = deduplicator or Deduplicator()
        self._batch_size = batch_size

        self._repository: DocumentRepository | None = None
        self._running = False
        self._metrics = get_metrics()

        logger.info(
            "Processing service initialized",
            batch_size=batch_size,
        )

    async def start(self) -> None:
        """
        Start the processing service.

        Runs until stop() is called or a fatal error occurs.
        """
        self._running = True

        logger.info("Starting processing service")

        # Connect to dependencies
        await self._queue.connect()
        await self._database.connect()

        # Create repository
        self._repository = DocumentRepository(self._database)
        await self._repository.create_tables()

        try:
            # Process messages from queue
            await self._process_loop()

        except asyncio.CancelledError:
            logger.info("Processing service cancelled")
        except Exception as e:
            logger.error("Processing service error", error=str(e))
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the processing service gracefully."""
        logger.info("Stopping processing service")
        self._running = False

    async def _cleanup(self) -> None:
        """Clean up resources."""
        await self._queue.close()
        await self._database.close()
        logger.info("Processing service cleaned up")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        batch: list[tuple[str, NormalizedDocument]] = []
        batch_start = time.monotonic()

        async for msg in self._queue.consume(
            count=self._batch_size,
            block_ms=5000,
        ):
            if not self._running:
                break

            batch.append((msg.message_id, msg.document))

            # Process batch when full or timeout
            if (
                len(batch) >= self._batch_size
                or (time.monotonic() - batch_start) > 5.0
            ):
                await self._process_batch(batch)
                batch = []
                batch_start = time.monotonic()

        # Process remaining documents
        if batch:
            await self._process_batch(batch)

    async def _process_batch(
        self,
        batch: list[tuple[str, NormalizedDocument]],
    ) -> None:
        """
        Process a batch of documents.

        Args:
            batch: List of (message_id, document) tuples
        """
        if not batch:
            return

        start_time = time.monotonic()
        processed = 0
        filtered = 0
        duplicates = 0
        errors = 0

        for msg_id, doc in batch:
            try:
                # Stage 1: Preprocessing
                stage_start = time.monotonic()
                doc = self._preprocessor.process(doc)
                self._metrics.record_stage_latency(
                    "preprocessing",
                    time.monotonic() - stage_start,
                )

                # Check if filtered
                if doc.should_filter:
                    filtered += 1
                    self._metrics.spam_filtered.labels(
                        platform=doc.platform,
                    ).inc()
                    await self._queue.ack(msg_id)
                    continue

                # Stage 2: Deduplication
                stage_start = time.monotonic()
                is_dup = not self._deduplicator.process(doc)
                self._metrics.record_stage_latency(
                    "deduplication",
                    time.monotonic() - stage_start,
                )

                if is_dup:
                    duplicates += 1
                    self._metrics.duplicates_detected.labels(
                        platform=doc.platform,
                    ).inc()
                    await self._queue.ack(msg_id)
                    continue

                # Stage 3: Storage
                stage_start = time.monotonic()
                await self._repository.insert(doc)
                self._metrics.record_stage_latency(
                    "storage",
                    time.monotonic() - stage_start,
                )

                processed += 1
                self._metrics.documents_stored.labels(
                    platform=doc.platform,
                ).inc()

                # Acknowledge successful processing
                await self._queue.ack(msg_id)

            except Exception as e:
                errors += 1
                logger.error(
                    "Error processing document",
                    doc_id=doc.id,
                    error=str(e),
                )
                self._metrics.record_error(
                    doc.platform,
                    type(e).__name__,
                    is_adapter=False,
                )

                # Move to dead letter queue
                await self._queue.nack(msg_id, str(e))

        # Update metrics
        elapsed = time.monotonic() - start_time
        self._metrics.dedup_index_size.set(self._deduplicator.stats["index_size"])

        logger.info(
            "Batch processed",
            total=len(batch),
            processed=processed,
            filtered=filtered,
            duplicates=duplicates,
            errors=errors,
            elapsed_seconds=round(elapsed, 2),
        )

    async def run_once(
        self,
        docs: list[NormalizedDocument],
    ) -> dict[str, int]:
        """
        Process a list of documents without queue.

        Useful for testing or manual processing.

        Args:
            docs: Documents to process

        Returns:
            Processing statistics
        """
        await self._database.connect()
        self._repository = DocumentRepository(self._database)
        await self._repository.create_tables()

        stats = {
            "total": len(docs),
            "processed": 0,
            "filtered": 0,
            "duplicates": 0,
            "errors": 0,
        }

        try:
            for doc in docs:
                try:
                    doc = self._preprocessor.process(doc)

                    if doc.should_filter:
                        stats["filtered"] += 1
                        continue

                    if not self._deduplicator.process(doc):
                        stats["duplicates"] += 1
                        continue

                    await self._repository.insert(doc)
                    stats["processed"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    logger.error("Error in run_once", error=str(e))

        finally:
            await self._database.close()

        return stats

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of the processing service.

        Returns:
            Dictionary with health status
        """
        queue_healthy = await self._queue.health_check()
        db_healthy = await self._database.health_check()

        return {
            "running": self._running,
            "queue_healthy": queue_healthy,
            "database_healthy": db_healthy,
            "dedup_stats": self._deduplicator.stats,
        }
