"""
Command-line interface for news-tracker.

Provides commands to run ingestion and processing services,
initialize the database, and run diagnostic checks.

Usage:
    news-tracker ingest   # Run ingestion service
    news-tracker process  # Run processing service
    news-tracker worker   # Run both services
    news-tracker init-db  # Initialize database
    news-tracker health   # Check service health
"""

import asyncio
import signal
import sys
from typing import Any

import click

from src.config.settings import get_settings
from src.observability.logging import setup_logging
from src.observability.metrics import get_metrics


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool) -> None:
    """News Tracker - Multi-platform financial data ingestion."""
    if debug:
        import os
        os.environ["LOG_LEVEL"] = "DEBUG"

    setup_logging()


@main.command()
@click.option("--mock", is_flag=True, help="Use mock adapters")
@click.option("--metrics/--no-metrics", default=True, help="Enable metrics server")
def ingest(mock: bool, metrics: bool) -> None:
    """Run the ingestion service."""
    from src.services.ingestion_service import IngestionService

    async def run():
        service = IngestionService(use_mock=mock)

        if metrics:
            get_metrics().start_server()

        # Handle shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(service.stop()))

        await service.start()

    asyncio.run(run())


@main.command()
@click.option("--batch-size", default=32, help="Processing batch size")
@click.option("--metrics/--no-metrics", default=True, help="Enable metrics server")
def process(batch_size: int, metrics: bool) -> None:
    """Run the processing service."""
    from src.services.processing_service import ProcessingService

    async def run():
        service = ProcessingService(batch_size=batch_size)

        if metrics:
            get_metrics().start_server()

        # Handle shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(service.stop()))

        await service.start()

    asyncio.run(run())


@main.command()
@click.option("--mock", is_flag=True, help="Use mock adapters")
@click.option("--metrics-port", default=8000, help="Metrics server port")
def worker(mock: bool, metrics_port: int) -> None:
    """Run both ingestion and processing services."""
    from src.services.ingestion_service import IngestionService
    from src.services.processing_service import ProcessingService

    async def run():
        ingestion = IngestionService(use_mock=mock)
        processing = ProcessingService()

        # Start metrics server
        get_metrics().start_server(port=metrics_port)

        # Handle shutdown signals
        async def shutdown():
            await ingestion.stop()
            await processing.stop()

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

        # Run both services concurrently
        await asyncio.gather(
            ingestion.start(),
            processing.start(),
        )

    asyncio.run(run())


@main.command("init-db")
def init_db() -> None:
    """Initialize the database schema."""
    from src.storage.database import Database
    from src.storage.repository import DocumentRepository

    async def run():
        db = Database()
        await db.connect()

        repo = DocumentRepository(db)
        await repo.create_tables()

        click.echo("Database initialized successfully")

        await db.close()

    asyncio.run(run())


@main.command()
def health() -> None:
    """Check health of all dependencies."""
    import structlog
    logger = structlog.get_logger()

    async def check():
        results: dict[str, bool] = {}

        # Check Redis
        try:
            from src.ingestion.queue import DocumentQueue
            queue = DocumentQueue()
            await queue.connect()
            results["redis"] = await queue.health_check()
            await queue.close()
        except Exception as e:
            results["redis"] = False
            logger.error("Redis health check failed", error=str(e))

        # Check PostgreSQL
        try:
            from src.storage.database import Database
            db = Database()
            await db.connect()
            results["postgres"] = await db.health_check()
            await db.close()
        except Exception as e:
            results["postgres"] = False
            logger.error("Postgres health check failed", error=str(e))

        # Check adapters
        settings = get_settings()
        results["twitter_configured"] = settings.twitter_configured
        results["reddit_configured"] = settings.reddit_configured
        results["news_api_configured"] = settings.news_api_configured

        # Print results
        click.echo("\nHealth Check Results:")
        click.echo("-" * 40)

        all_healthy = True
        for name, status in results.items():
            icon = "✓" if status else "✗"
            color = "green" if status else "red"
            click.echo(click.style(f"  {icon} {name}: {status}", fg=color))
            if name in ("redis", "postgres") and not status:
                all_healthy = False

        click.echo("-" * 40)

        if all_healthy:
            click.echo(click.style("All core services healthy!", fg="green"))
            sys.exit(0)
        else:
            click.echo(click.style("Some services unhealthy!", fg="red"))
            sys.exit(1)

    asyncio.run(check())


@main.command("run-once")
@click.option("--mock", is_flag=True, help="Use mock adapters")
def run_once(mock: bool) -> None:
    """Run one ingestion + processing cycle."""
    from src.services.ingestion_service import IngestionService
    from src.services.processing_service import ProcessingService

    async def run():
        # Ingestion
        ingestion = IngestionService(use_mock=mock)
        ingestion_results = await ingestion.run_once()

        click.echo("\nIngestion Results:")
        for platform, count in ingestion_results.items():
            click.echo(f"  {platform.value}: {count} documents")

        # Processing
        processing = ProcessingService()

        # Get documents from queue
        from src.ingestion.queue import DocumentQueue
        queue = DocumentQueue()
        await queue.connect()

        docs = []
        try:
            # Read all pending messages
            async for msg in queue.consume(count=1000, block_ms=1000):
                docs.append(msg.document)
                await queue.ack(msg.message_id)
                if len(docs) >= 1000:
                    break
        except asyncio.TimeoutError:
            pass
        finally:
            await queue.close()

        if docs:
            processing_results = await processing.run_once(docs)

            click.echo("\nProcessing Results:")
            for key, value in processing_results.items():
                click.echo(f"  {key}: {value}")
        else:
            click.echo("\nNo documents to process")

    asyncio.run(run())


if __name__ == "__main__":
    main()
