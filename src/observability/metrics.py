"""
Prometheus metrics for monitoring the ingestion pipeline.

Defines and exposes metrics for:
- Document ingestion rates
- Processing latency
- Queue depth
- Error rates
- Adapter health

Metrics are exposed via HTTP endpoint for Prometheus scraping.
"""

import logging
from typing import Any

from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from src.config.settings import get_settings
from src.ingestion.schemas import Platform

logger = logging.getLogger(__name__)

# Buckets for latency histograms (in seconds)
LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class MetricsCollector:
    """
    Prometheus metrics collector for the news-tracker pipeline.

    Provides methods to record metrics for ingestion, processing,
    and system health.

    Usage:
        metrics = MetricsCollector()
        metrics.start_server()

        # Record metrics
        metrics.documents_ingested.labels(platform="twitter").inc()
        metrics.processing_latency.labels(stage="spam_detection").observe(0.05)
    """

    def __init__(self):
        """Initialize Prometheus metrics."""

        # Document counters
        self.documents_ingested = Counter(
            "news_tracker_documents_ingested_total",
            "Total number of documents ingested",
            ["platform"],
        )

        self.documents_processed = Counter(
            "news_tracker_documents_processed_total",
            "Total number of documents processed",
            ["platform", "status"],  # status: success, filtered, error
        )

        self.documents_stored = Counter(
            "news_tracker_documents_stored_total",
            "Total number of documents stored in database",
            ["platform"],
        )

        # Error counters
        self.adapter_errors = Counter(
            "news_tracker_adapter_errors_total",
            "Total adapter errors",
            ["platform", "error_type"],
        )

        self.processing_errors = Counter(
            "news_tracker_processing_errors_total",
            "Total processing errors",
            ["stage", "error_type"],
        )

        # Latency histograms
        self.ingestion_latency = Histogram(
            "news_tracker_ingestion_latency_seconds",
            "Time to fetch documents from platform",
            ["platform"],
            buckets=LATENCY_BUCKETS,
        )

        self.processing_latency = Histogram(
            "news_tracker_processing_latency_seconds",
            "Time to process a document through pipeline stages",
            ["stage"],  # spam_detection, deduplication, ticker_extraction
            buckets=LATENCY_BUCKETS,
        )

        self.storage_latency = Histogram(
            "news_tracker_storage_latency_seconds",
            "Time to store documents in database",
            ["operation"],  # insert, batch_insert, query
            buckets=LATENCY_BUCKETS,
        )

        # Queue metrics
        self.queue_depth = Gauge(
            "news_tracker_queue_depth",
            "Number of messages in Redis queue",
            ["stream"],
        )

        self.queue_pending = Gauge(
            "news_tracker_queue_pending",
            "Number of pending (unacknowledged) messages",
            ["consumer_group"],
        )

        # Adapter health
        self.adapter_health = Gauge(
            "news_tracker_adapter_health",
            "Adapter health status (1=healthy, 0=unhealthy)",
            ["platform"],
        )

        # Deduplication metrics
        self.duplicates_detected = Counter(
            "news_tracker_duplicates_detected_total",
            "Total near-duplicates detected",
            ["platform"],
        )

        self.dedup_index_size = Gauge(
            "news_tracker_dedup_index_size",
            "Number of documents in deduplication index",
        )

        # Spam filtering metrics
        self.spam_filtered = Counter(
            "news_tracker_spam_filtered_total",
            "Total documents filtered as spam",
            ["platform"],
        )

        self.bot_filtered = Counter(
            "news_tracker_bot_filtered_total",
            "Total documents filtered as bot content",
            ["platform"],
        )

        # Database metrics
        self.db_connections_active = Gauge(
            "news_tracker_db_connections_active",
            "Number of active database connections",
        )

        self.db_connections_idle = Gauge(
            "news_tracker_db_connections_idle",
            "Number of idle database connections",
        )

        # Embedding metrics
        self.embeddings_generated = Counter(
            "news_tracker_embeddings_generated_total",
            "Total embeddings generated",
            ["platform", "model"],
        )

        self.embedding_latency = Histogram(
            "news_tracker_embedding_latency_seconds",
            "Time to generate embeddings",
            ["operation"],  # single, batch
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )

        self.embedding_cache_hits = Counter(
            "news_tracker_embedding_cache_hits_total",
            "Total embedding cache hits",
        )

        self.embedding_cache_misses = Counter(
            "news_tracker_embedding_cache_misses_total",
            "Total embedding cache misses",
        )

        self.embedding_queue_depth = Gauge(
            "news_tracker_embedding_queue_depth",
            "Number of jobs in embedding queue",
        )

        self.embedding_batch_size = Histogram(
            "news_tracker_embedding_batch_size",
            "Embedding batch sizes",
            buckets=(1, 5, 10, 20, 32, 50, 100),
        )

        # Sentiment metrics
        self.sentiment_analyzed = Counter(
            "news_tracker_sentiment_analyzed_total",
            "Total sentiment analyses performed",
            ["platform", "label"],  # label: positive, negative, neutral
        )

        self.sentiment_latency = Histogram(
            "news_tracker_sentiment_latency_seconds",
            "Time to analyze sentiment",
            ["operation"],  # single, batch, entity
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        self.sentiment_cache_hits = Counter(
            "news_tracker_sentiment_cache_hits_total",
            "Total sentiment cache hits",
        )

        self.sentiment_cache_misses = Counter(
            "news_tracker_sentiment_cache_misses_total",
            "Total sentiment cache misses",
        )

        self.sentiment_errors = Counter(
            "news_tracker_sentiment_errors_total",
            "Total sentiment analysis errors",
            ["error_type"],
        )

        self.sentiment_queue_depth = Gauge(
            "news_tracker_sentiment_queue_depth",
            "Number of jobs in sentiment queue",
        )

        self.sentiment_batch_size = Histogram(
            "news_tracker_sentiment_batch_size",
            "Sentiment batch sizes",
            buckets=(1, 5, 10, 20, 32, 50, 100),
        )

        self.sentiment_confidence = Histogram(
            "news_tracker_sentiment_confidence",
            "Distribution of sentiment confidence scores",
            ["label"],
            buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
        )

        self.sentiment_entity_count = Histogram(
            "news_tracker_sentiment_entity_count",
            "Number of entities analyzed per document",
            buckets=(0, 1, 2, 3, 5, 10, 20),
        )

        # Clustering metrics
        self.clustering_assigned = Counter(
            "news_tracker_clustering_assigned_total",
            "Total documents assigned to themes",
            ["platform"],
        )

        self.clustering_errors = Counter(
            "news_tracker_clustering_errors_total",
            "Total clustering processing errors",
            ["error_type"],
        )

        self.clustering_queue_depth = Gauge(
            "news_tracker_clustering_queue_depth",
            "Number of jobs in clustering queue",
        )

        self.clustering_batch_size = Histogram(
            "news_tracker_clustering_batch_size",
            "Clustering batch sizes",
            buckets=(1, 5, 10, 20, 32, 50, 100),
        )

        self.clustering_similarity = Histogram(
            "news_tracker_clustering_similarity",
            "Distribution of best theme similarity scores",
            buckets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
        )

        self.clustering_latency = Histogram(
            "news_tracker_clustering_latency_seconds",
            "Time to process clustering operations",
            ["operation"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # Queue reclaim metrics
        self.pending_reclaimed = Counter(
            "news_tracker_queue_pending_reclaimed_total",
            "Total messages reclaimed from pending state",
            ["queue"],
        )

        self.dlq_max_retries = Counter(
            "news_tracker_queue_dlq_max_retries_total",
            "Total messages moved to DLQ due to max retries exceeded",
            ["queue"],
        )

        logger.info("Prometheus metrics initialized")

    def start_server(self, port: int | None = None) -> None:
        """
        Start Prometheus metrics HTTP server.

        Args:
            port: Port to expose metrics on (default from settings)
        """
        settings = get_settings()
        port = port or settings.metrics_port

        start_http_server(port, registry=REGISTRY)
        logger.info(f"Prometheus metrics server started on port {port}")

    # Convenience methods

    def record_ingestion(
        self,
        platform: Platform | str,
        count: int = 1,
        latency: float | None = None,
    ) -> None:
        """
        Record document ingestion.

        Args:
            platform: Source platform
            count: Number of documents ingested
            latency: Optional ingestion latency in seconds
        """
        platform_str = platform.value if isinstance(platform, Platform) else platform
        self.documents_ingested.labels(platform=platform_str).inc(count)

        if latency is not None:
            self.ingestion_latency.labels(platform=platform_str).observe(latency)

    def record_processing(
        self,
        platform: Platform | str,
        status: str,
        count: int = 1,
    ) -> None:
        """
        Record document processing result.

        Args:
            platform: Source platform
            status: Processing status (success, filtered, error)
            count: Number of documents
        """
        platform_str = platform.value if isinstance(platform, Platform) else platform
        self.documents_processed.labels(
            platform=platform_str,
            status=status,
        ).inc(count)

    def record_stage_latency(self, stage: str, latency: float) -> None:
        """
        Record processing stage latency.

        Args:
            stage: Stage name (spam_detection, deduplication, etc.)
            latency: Latency in seconds
        """
        self.processing_latency.labels(stage=stage).observe(latency)

    def record_error(
        self,
        platform: Platform | str,
        error_type: str,
        is_adapter: bool = True,
    ) -> None:
        """
        Record an error.

        Args:
            platform: Source platform or stage
            error_type: Error type/category
            is_adapter: Whether this is an adapter error
        """
        platform_str = platform.value if isinstance(platform, Platform) else platform

        if is_adapter:
            self.adapter_errors.labels(
                platform=platform_str,
                error_type=error_type,
            ).inc()
        else:
            self.processing_errors.labels(
                stage=platform_str,
                error_type=error_type,
            ).inc()

    def set_adapter_health(self, platform: Platform | str, healthy: bool) -> None:
        """
        Set adapter health status.

        Args:
            platform: Platform
            healthy: Whether adapter is healthy
        """
        platform_str = platform.value if isinstance(platform, Platform) else platform
        self.adapter_health.labels(platform=platform_str).set(1 if healthy else 0)

    def set_queue_depth(self, stream: str, depth: int) -> None:
        """
        Set queue depth metric.

        Args:
            stream: Stream name
            depth: Number of messages
        """
        self.queue_depth.labels(stream=stream).set(depth)

    def record_embedding_generated(
        self,
        platform: Platform | str,
        model: str = "finbert",
        count: int = 1,
    ) -> None:
        """
        Record embedding generation.

        Args:
            platform: Source platform
            model: Model used (finbert or minilm)
            count: Number of embeddings generated
        """
        platform_str = platform.value if isinstance(platform, Platform) else platform
        self.embeddings_generated.labels(platform=platform_str, model=model).inc(count)

    def record_embedding_latency(
        self,
        operation: str,
        latency: float,
    ) -> None:
        """
        Record embedding generation latency.

        Args:
            operation: Operation type (single, batch)
            latency: Latency in seconds
        """
        self.embedding_latency.labels(operation=operation).observe(latency)

    def record_embedding_cache(self, hit: bool) -> None:
        """
        Record embedding cache hit or miss.

        Args:
            hit: True for cache hit, False for miss
        """
        if hit:
            self.embedding_cache_hits.inc()
        else:
            self.embedding_cache_misses.inc()

    def set_embedding_queue_depth(self, depth: int) -> None:
        """
        Set embedding queue depth metric.

        Args:
            depth: Number of jobs in queue
        """
        self.embedding_queue_depth.set(depth)

    def record_embedding_batch(
        self,
        processed: int,
        skipped: int,
        errors: int,
        latency: float,
    ) -> None:
        """
        Record embedding batch processing metrics.

        Args:
            processed: Number of embeddings generated
            skipped: Number of documents skipped
            errors: Number of errors
            latency: Total batch latency in seconds
        """
        total = processed + skipped + errors
        if total > 0:
            self.embedding_batch_size.observe(total)
        if latency > 0:
            self.embedding_latency.labels(operation="batch").observe(latency)

    def record_sentiment_analyzed(
        self,
        platform: Platform | str,
        label: str,
        confidence: float | None = None,
        count: int = 1,
    ) -> None:
        """
        Record sentiment analysis.

        Args:
            platform: Source platform
            label: Sentiment label (positive, negative, neutral)
            confidence: Optional confidence score to record
            count: Number of analyses
        """
        platform_str = platform.value if isinstance(platform, Platform) else platform
        self.sentiment_analyzed.labels(platform=platform_str, label=label).inc(count)

        if confidence is not None:
            self.sentiment_confidence.labels(label=label).observe(confidence)

    def record_sentiment_latency(
        self,
        operation: str,
        latency: float,
    ) -> None:
        """
        Record sentiment analysis latency.

        Args:
            operation: Operation type (single, batch, entity)
            latency: Latency in seconds
        """
        self.sentiment_latency.labels(operation=operation).observe(latency)

    def record_sentiment_cache(self, hit: bool) -> None:
        """
        Record sentiment cache hit or miss.

        Args:
            hit: True for cache hit, False for miss
        """
        if hit:
            self.sentiment_cache_hits.inc()
        else:
            self.sentiment_cache_misses.inc()

    def record_sentiment_error(self, error_type: str) -> None:
        """
        Record sentiment analysis error.

        Args:
            error_type: Error type/category
        """
        self.sentiment_errors.labels(error_type=error_type).inc()

    def set_sentiment_queue_depth(self, depth: int) -> None:
        """
        Set sentiment queue depth metric.

        Args:
            depth: Number of jobs in queue
        """
        self.sentiment_queue_depth.set(depth)

    def record_sentiment_batch(
        self,
        processed: int,
        skipped: int,
        errors: int,
        latency: float,
    ) -> None:
        """
        Record sentiment batch processing metrics.

        Args:
            processed: Number of documents analyzed
            skipped: Number of documents skipped
            errors: Number of errors
            latency: Total batch latency in seconds
        """
        total = processed + skipped + errors
        if total > 0:
            self.sentiment_batch_size.observe(total)
        if latency > 0:
            self.sentiment_latency.labels(operation="batch").observe(latency)

    def record_sentiment_entities(self, entity_count: int) -> None:
        """
        Record number of entities analyzed in a document.

        Args:
            entity_count: Number of entities
        """
        self.sentiment_entity_count.observe(entity_count)

    def record_clustering_assigned(
        self,
        platform: Platform | str,
        count: int = 1,
    ) -> None:
        """
        Record document theme assignment.

        Args:
            platform: Source platform
            count: Number of documents assigned
        """
        platform_str = platform.value if isinstance(platform, Platform) else platform
        self.clustering_assigned.labels(platform=platform_str).inc(count)

    def record_clustering_error(self, error_type: str) -> None:
        """
        Record clustering processing error.

        Args:
            error_type: Error type/category
        """
        self.clustering_errors.labels(error_type=error_type).inc()

    def set_clustering_queue_depth(self, depth: int) -> None:
        """
        Set clustering queue depth metric.

        Args:
            depth: Number of jobs in queue
        """
        self.clustering_queue_depth.set(depth)

    def record_clustering_batch(
        self,
        processed: int,
        skipped: int,
        errors: int,
        latency: float,
    ) -> None:
        """
        Record clustering batch processing metrics.

        Args:
            processed: Number of documents assigned to themes
            skipped: Number of documents skipped
            errors: Number of errors
            latency: Total batch latency in seconds
        """
        total = processed + skipped + errors
        if total > 0:
            self.clustering_batch_size.observe(total)
        if latency > 0:
            self.clustering_latency.labels(operation="batch").observe(latency)


# Global metrics instance
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
