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


# Global metrics instance
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
