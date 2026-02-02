"""
Ingestion service - fetches documents from platform adapters.

Runs continuously, polling adapters at configured intervals and
publishing documents to the Redis queue for processing.

Features:
- Concurrent adapter execution
- Graceful shutdown
- Health monitoring
- Metrics collection
"""

import asyncio
import logging
import time
from typing import Any

import structlog

from src.config.settings import get_settings
from src.ingestion.base_adapter import BaseAdapter
from src.ingestion.mock_adapter import MockAdapter, create_mock_adapters
from src.ingestion.news_adapter import NewsAdapter
from src.ingestion.queue import DocumentQueue
from src.ingestion.reddit_adapter import RedditAdapter
from src.ingestion.schemas import Platform
from src.ingestion.substack_adapter import SubstackAdapter
from src.ingestion.twitter_adapter import TwitterAdapter
from src.observability.metrics import get_metrics

logger = structlog.get_logger(__name__)


class IngestionService:
    """
    Service that orchestrates document ingestion from all platforms.

    Runs adapters concurrently and publishes documents to Redis queue.
    Handles graceful shutdown and error recovery.

    Usage:
        service = IngestionService()
        await service.start()  # Runs until stopped
    """

    def __init__(
        self,
        adapters: dict[Platform, BaseAdapter] | None = None,
        queue: DocumentQueue | None = None,
        use_mock: bool = False,
    ):
        """
        Initialize ingestion service.

        Args:
            adapters: Platform adapters (or auto-create from config)
            queue: Document queue (or create from config)
            use_mock: Use mock adapters instead of real APIs
        """
        settings = get_settings()

        self._poll_interval = settings.poll_interval_seconds
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._metrics = get_metrics()

        # Initialize queue
        self._queue = queue or DocumentQueue()

        # Initialize adapters
        if adapters:
            self._adapters = adapters
        elif use_mock:
            self._adapters = create_mock_adapters()
        else:
            self._adapters = self._create_adapters(settings)

        logger.info(
            "Ingestion service initialized",
            adapters=list(self._adapters.keys()),
            poll_interval=self._poll_interval,
        )

    def _create_adapters(self, settings) -> dict[Platform, BaseAdapter]:
        """Create adapters based on available configuration."""
        adapters = {}

        # Twitter
        if settings.twitter_configured:
            adapters[Platform.TWITTER] = TwitterAdapter(
                rate_limit=settings.twitter_rate_limit,
            )
            logger.info("Twitter adapter enabled")

        # Reddit
        if settings.reddit_configured:
            adapters[Platform.REDDIT] = RedditAdapter(
                rate_limit=settings.reddit_rate_limit,
            )
            logger.info("Reddit adapter enabled")

        # Substack (always available - uses public RSS)
        adapters[Platform.SUBSTACK] = SubstackAdapter(
            rate_limit=settings.substack_rate_limit,
        )
        logger.info("Substack adapter enabled")

        # News APIs
        if settings.news_api_configured:
            adapters[Platform.NEWS] = NewsAdapter(
                rate_limit=settings.news_rate_limit,
            )
            logger.info("News adapter enabled")

        # If no adapters configured, use mock
        if not adapters:
            logger.warning("No API credentials configured, using mock adapters")
            return create_mock_adapters()

        return adapters

    async def start(self) -> None:
        """
        Start the ingestion service.

        Runs until stop() is called or a fatal error occurs.
        """
        self._running = True

        logger.info("Starting ingestion service")

        # Connect to queue
        await self._queue.connect()

        try:
            # Create tasks for each adapter
            self._tasks = [
                asyncio.create_task(
                    self._run_adapter(platform, adapter),
                    name=f"adapter_{platform.value}",
                )
                for platform, adapter in self._adapters.items()
            ]

            # Wait for all tasks
            await asyncio.gather(*self._tasks, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("Ingestion service cancelled")
        except Exception as e:
            logger.error("Ingestion service error", error=str(e))
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the ingestion service gracefully."""
        logger.info("Stopping ingestion service")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for cancellation
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        await self._queue.close()
        self._tasks.clear()
        logger.info("Ingestion service cleaned up")

    async def _run_adapter(
        self,
        platform: Platform,
        adapter: BaseAdapter,
    ) -> None:
        """
        Run a single adapter in a loop.

        Args:
            platform: Platform being polled
            adapter: Adapter instance
        """
        logger.info("Starting adapter", platform=platform.value)

        while self._running:
            start_time = time.monotonic()

            try:
                # Check adapter health
                healthy = await adapter.health_check()
                self._metrics.set_adapter_health(platform, healthy)

                if not healthy:
                    logger.warning(
                        "Adapter health check failed",
                        platform=platform.value,
                    )
                    await asyncio.sleep(self._poll_interval)
                    continue

                # Fetch documents
                doc_count = 0
                async for doc in adapter.fetch():
                    await self._queue.publish(doc)
                    doc_count += 1

                # Record metrics
                elapsed = time.monotonic() - start_time
                self._metrics.record_ingestion(
                    platform=platform,
                    count=doc_count,
                    latency=elapsed,
                )

                logger.info(
                    "Adapter fetch completed",
                    platform=platform.value,
                    documents=doc_count,
                    elapsed_seconds=round(elapsed, 2),
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Adapter error",
                    platform=platform.value,
                    error=str(e),
                )
                self._metrics.record_error(platform, type(e).__name__)

            # Wait before next poll
            await asyncio.sleep(self._poll_interval)

        logger.info("Adapter stopped", platform=platform.value)

    async def run_once(self) -> dict[Platform, int]:
        """
        Run one ingestion cycle for all adapters.

        Useful for testing or manual triggers.

        Returns:
            Dictionary of platform -> document count
        """
        results = {}

        await self._queue.connect()

        try:
            for platform, adapter in self._adapters.items():
                count = 0
                try:
                    async for doc in adapter.fetch():
                        await self._queue.publish(doc)
                        count += 1
                except Exception as e:
                    logger.error(
                        "Adapter error in run_once",
                        platform=platform.value,
                        error=str(e),
                    )
                results[platform] = count
        finally:
            await self._queue.close()

        return results

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of the ingestion service.

        Returns:
            Dictionary with health status
        """
        adapter_health = {}
        for platform, adapter in self._adapters.items():
            try:
                adapter_health[platform.value] = await adapter.health_check()
            except Exception:
                adapter_health[platform.value] = False

        queue_healthy = await self._queue.health_check()

        return {
            "running": self._running,
            "queue_healthy": queue_healthy,
            "adapters": adapter_health,
            "active_tasks": len([t for t in self._tasks if not t.done()]),
        }
