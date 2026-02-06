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
from src.embedding.queue import EmbeddingQueue
from src.ingestion.deduplication import Deduplicator
from src.ingestion.preprocessor import Preprocessor
from src.ingestion.queue import DocumentQueue, QueueMessage
from src.ingestion.schemas import NormalizedDocument
from src.observability.metrics import get_metrics
from src.sentiment.queue import SentimentQueue
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
        embedding_queue: EmbeddingQueue | None = None,
        sentiment_queue: SentimentQueue | None = None,
        batch_size: int = 32,
        enable_embedding_queue: bool = True,
        enable_sentiment_queue: bool = True,
    ):
        """
        Initialize processing service.

        Args:
            queue: Document queue (or create from config)
            database: Database connection (or create from config)
            preprocessor: Document preprocessor (or create default)
            deduplicator: Deduplication index (or create default)
            embedding_queue: Embedding job queue (or create from config)
            sentiment_queue: Sentiment job queue (or create from config)
            batch_size: Documents to process per batch
            enable_embedding_queue: Whether to publish to embedding queue
            enable_sentiment_queue: Whether to publish to sentiment queue
        """
        self._queue = queue or DocumentQueue()
        self._database = database or Database()
        self._preprocessor = preprocessor or self._create_default_preprocessor()
        self._deduplicator = deduplicator or Deduplicator()
        self._embedding_queue = embedding_queue
        self._sentiment_queue = sentiment_queue
        self._enable_embedding_queue = enable_embedding_queue
        self._enable_sentiment_queue = enable_sentiment_queue
        self._batch_size = batch_size

        self._repository: DocumentRepository | None = None
        self._running = False
        self._metrics = get_metrics()

        logger.info(
            "Processing service initialized",
            batch_size=batch_size,
            embedding_queue_enabled=enable_embedding_queue,
            sentiment_queue_enabled=enable_sentiment_queue,
        )

    def _create_default_preprocessor(self) -> Preprocessor:
        """
        Create a Preprocessor with auto-injected services based on settings.

        Auto-injects NERService when ner_enabled=True and
        KeywordsService when keywords_enabled=True.
        """
        settings = get_settings()

        ner_service = None
        enable_ner = settings.ner_enabled
        if enable_ner:
            try:
                from src.ner.service import NERService
                ner_service = NERService()
                logger.info("NER service auto-injected into preprocessor")
            except ImportError:
                logger.warning("NER enabled but spacy not available")
                enable_ner = False

        keywords_service = None
        enable_keywords = settings.keywords_enabled
        if enable_keywords:
            try:
                from src.keywords.service import KeywordsService
                keywords_service = KeywordsService()
                logger.info("Keywords service auto-injected into preprocessor")
            except ImportError:
                logger.warning("Keywords enabled but rapid-textrank not available")
                enable_keywords = False

        event_extractor = None
        enable_events = settings.events_enabled
        if enable_events:
            try:
                from src.event_extraction.patterns import PatternExtractor
                event_extractor = PatternExtractor()
                logger.info("Event extractor auto-injected into preprocessor")
            except ImportError:
                logger.warning("Events enabled but event_extraction module not available")
                enable_events = False

        return Preprocessor(
            ner_service=ner_service,
            enable_ner=enable_ner,
            keywords_service=keywords_service,
            enable_keywords=enable_keywords,
            event_extractor=event_extractor,
            enable_events=enable_events,
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

        # Connect embedding queue if enabled
        if self._enable_embedding_queue:
            if self._embedding_queue is None:
                self._embedding_queue = EmbeddingQueue()
            await self._embedding_queue.connect()

        # Connect sentiment queue if enabled
        if self._enable_sentiment_queue:
            if self._sentiment_queue is None:
                self._sentiment_queue = SentimentQueue()
            await self._sentiment_queue.connect()

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
        if self._embedding_queue:
            await self._embedding_queue.close()
        if self._sentiment_queue:
            await self._sentiment_queue.close()
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
                try:
                    stage_start = time.monotonic()
                    doc = self._preprocessor.process(doc)
                    self._metrics.record_stage_latency(
                        "preprocessing",
                        time.monotonic() - stage_start,
                    )
                except Exception as e:
                    errors += 1
                    logger.error(
                        "Error in preprocessing",
                        doc_id=doc.id,
                        error=str(e),
                    )
                    self._metrics.record_error(
                        "preprocessing",
                        type(e).__name__,
                        is_adapter=False,
                    )
                    await self._queue.nack(msg_id, str(e))
                    continue

                # Check if filtered
                if doc.should_filter:
                    filtered += 1
                    self._metrics.spam_filtered.labels(
                        platform=doc.platform.value,
                    ).inc()
                    await self._queue.ack(msg_id)
                    continue

                # Stage 2: Deduplication
                try:
                    stage_start = time.monotonic()
                    is_dup = not self._deduplicator.process(doc)
                    self._metrics.record_stage_latency(
                        "deduplication",
                        time.monotonic() - stage_start,
                    )
                except Exception as e:
                    errors += 1
                    logger.error(
                        "Error in deduplication",
                        doc_id=doc.id,
                        error=str(e),
                    )
                    self._metrics.record_error(
                        "deduplication",
                        type(e).__name__,
                        is_adapter=False,
                    )
                    await self._queue.nack(msg_id, str(e))
                    continue

                if is_dup:
                    duplicates += 1
                    self._metrics.duplicates_detected.labels(
                        platform=doc.platform.value,
                    ).inc()
                    await self._queue.ack(msg_id)
                    continue

                # Stage 3: Storage
                try:
                    stage_start = time.monotonic()
                    await self._repository.insert(doc)
                    self._metrics.record_stage_latency(
                        "storage",
                        time.monotonic() - stage_start,
                    )
                except Exception as e:
                    errors += 1
                    logger.error(
                        "Error in storage",
                        doc_id=doc.id,
                        error=str(e),
                    )
                    self._metrics.record_error(
                        "storage",
                        type(e).__name__,
                        is_adapter=False,
                    )
                    await self._queue.nack(msg_id, str(e))
                    continue

                # Stage 4: Queue for embedding generation
                if self._embedding_queue:
                    try:
                        await self._embedding_queue.publish(doc.id)
                    except Exception as e:
                        # Log but don't fail the document processing
                        logger.warning(
                            "Failed to queue embedding job",
                            doc_id=doc.id,
                            error=str(e),
                        )

                # Stage 5: Queue for sentiment analysis
                if self._sentiment_queue:
                    try:
                        await self._sentiment_queue.publish(doc.id)
                    except Exception as e:
                        # Log but don't fail the document processing
                        logger.warning(
                            "Failed to queue sentiment job",
                            doc_id=doc.id,
                            error=str(e),
                        )

                processed += 1
                self._metrics.documents_stored.labels(
                    platform=doc.platform.value,
                ).inc()

                # Acknowledge successful processing
                await self._queue.ack(msg_id)

            except Exception as e:
                # Catch any unexpected errors (e.g., from queue.ack/nack themselves)
                errors += 1
                logger.error(
                    "Unexpected error processing document",
                    doc_id=doc.id,
                    error=str(e),
                )
                self._metrics.record_error(
                    "unknown",
                    type(e).__name__,
                    is_adapter=False,
                )

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
        return_doc_ids: bool = False,
    ) -> dict[str, int] | tuple[dict[str, int], list[str]]:
        """
        Process a list of documents without queue.

        Useful for testing or manual processing.

        Args:
            docs: Documents to process
            return_doc_ids: If True, also return list of stored document IDs

        Returns:
            Processing statistics, or tuple of (stats, doc_ids) if return_doc_ids=True
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
        stored_doc_ids: list[str] = []

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
                    stored_doc_ids.append(doc.id)

                except Exception as e:
                    stats["errors"] += 1
                    logger.error("Error in run_once", error=str(e))

        finally:
            await self._database.close()

        if return_doc_ids:
            return stats, stored_doc_ids
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
        embedding_queue_healthy = (
            await self._embedding_queue.health_check()
            if self._embedding_queue
            else None
        )
        sentiment_queue_healthy = (
            await self._sentiment_queue.health_check()
            if self._sentiment_queue
            else None
        )

        return {
            "running": self._running,
            "queue_healthy": queue_healthy,
            "database_healthy": db_healthy,
            "embedding_queue_healthy": embedding_queue_healthy,
            "sentiment_queue_healthy": sentiment_queue_healthy,
            "dedup_stats": self._deduplicator.stats,
        }
