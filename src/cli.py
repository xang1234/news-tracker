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


@main.command()
@click.option("--host", default=None, help="API server host")
@click.option("--port", default=None, type=int, help="API server port")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev only)")
@click.option("--metrics-port", default=8000, help="Metrics server port")
def serve(host: str | None, port: int | None, reload: bool, metrics_port: int) -> None:
    """Start the embedding API server."""
    import uvicorn

    settings = get_settings()
    host = host or settings.api_host
    port = port or settings.api_port

    # Start metrics server on separate port
    get_metrics().start_server(port=metrics_port)

    click.echo(f"Starting API server on {host}:{port}")
    click.echo(f"Metrics available on http://localhost:{metrics_port}/metrics")
    click.echo(f"API docs available on http://localhost:{port}/docs")

    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@main.command("run-once")
@click.option("--mock", is_flag=True, help="Use mock adapters")
@click.option("--with-embeddings", is_flag=True, help="Run embedding worker after processing")
@click.option("--with-sentiment", is_flag=True, help="Run sentiment worker after processing")
@click.option("--verify", is_flag=True, help="Query DB to confirm embeddings exist")
def run_once(mock: bool, with_embeddings: bool, with_sentiment: bool, verify: bool) -> None:
    """Run one ingestion + processing cycle.

    With --with-embeddings, also generates embeddings for processed documents.
    With --with-sentiment, also generates sentiment analysis for processed documents.
    With --verify, queries the database to confirm documents have embeddings.
    """
    from src.services.ingestion_service import IngestionService
    from src.services.processing_service import ProcessingService

    async def run():
        exit_code = 0

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
            # Read all pending messages with a timeout
            # The consume() generator runs forever, so we use asyncio.timeout
            # to break out after messages stop arriving
            async with asyncio.timeout(3):  # 3 second timeout for idle
                async for msg in queue.consume(count=1000, block_ms=1000):
                    docs.append(msg.document)
                    await queue.ack(msg.message_id)
                    if len(docs) >= 1000:
                        break
        except TimeoutError:
            # Expected when no more messages arrive within timeout
            pass
        finally:
            await queue.close()

        stored_doc_ids: list[str] = []
        if docs:
            result = await processing.run_once(docs, return_doc_ids=with_embeddings)

            if with_embeddings:
                processing_results, stored_doc_ids = result
            else:
                processing_results = result

            click.echo("\nProcessing Results:")
            click.echo(f"  stored: {processing_results['processed']}")
            click.echo(f"  filtered (spam): {processing_results['filtered']}")
            click.echo(f"  duplicates: {processing_results['duplicates']}")
            click.echo(f"  errors: {processing_results['errors']}")
        else:
            click.echo("\nNo documents to process")

        # Embedding stage (optional)
        if with_embeddings and stored_doc_ids:
            from src.embedding.worker import EmbeddingWorker

            click.echo("\nGenerating embeddings...")
            worker = EmbeddingWorker()
            embedding_results = await worker.run_once(stored_doc_ids)

            # Calculate model breakdown from stats
            # Note: run_once doesn't track per-model counts, so we report totals
            click.echo("\nEmbedding Results:")
            click.echo(f"  processed: {embedding_results['processed']}")
            click.echo(f"  skipped: {embedding_results['skipped']}")
            click.echo(f"  errors: {embedding_results['errors']}")

            if embedding_results['errors'] > 0:
                exit_code = 1

        # Sentiment stage (optional)
        if with_sentiment and stored_doc_ids:
            from src.sentiment.worker import SentimentWorker

            click.echo("\nGenerating sentiment analysis...")
            sentiment_worker = SentimentWorker()
            sentiment_results = await sentiment_worker.run_once(stored_doc_ids)

            click.echo("\nSentiment Results:")
            click.echo(f"  processed: {sentiment_results['processed']}")
            click.echo(f"  skipped: {sentiment_results['skipped']}")
            click.echo(f"  errors: {sentiment_results['errors']}")

            if sentiment_results['errors'] > 0:
                exit_code = 1

        # Verification stage (optional)
        if verify and stored_doc_ids:
            from src.storage.database import Database
            from src.storage.repository import DocumentRepository

            db = Database()
            await db.connect()
            repo = DocumentRepository(db)

            try:
                # Count documents and those with embeddings/sentiment
                total_count = 0
                with_embedding_count = 0
                with_sentiment_count = 0

                for doc_id in stored_doc_ids:
                    doc = await repo.get_by_id(doc_id)
                    if doc:
                        total_count += 1
                        if doc.embedding is not None or doc.embedding_minilm is not None:
                            with_embedding_count += 1
                        if doc.sentiment is not None:
                            with_sentiment_count += 1

                click.echo("\nVerification:")
                if total_count == len(stored_doc_ids):
                    click.echo(click.style(f"  ✓ {total_count} documents in database", fg="green"))
                else:
                    click.echo(click.style(f"  ✗ Only {total_count}/{len(stored_doc_ids)} documents in database", fg="red"))
                    exit_code = 1

                if with_embeddings:
                    if with_embedding_count == total_count:
                        click.echo(click.style(f"  ✓ {with_embedding_count} documents have embeddings", fg="green"))
                    else:
                        click.echo(click.style(f"  ✗ Only {with_embedding_count}/{total_count} documents have embeddings", fg="red"))
                        exit_code = 1

                if with_sentiment:
                    if with_sentiment_count == total_count:
                        click.echo(click.style(f"  ✓ {with_sentiment_count} documents have sentiment", fg="green"))
                    else:
                        click.echo(click.style(f"  ✗ Only {with_sentiment_count}/{total_count} documents have sentiment", fg="red"))
                        exit_code = 1

                if exit_code == 0:
                    click.echo(click.style("  ✓ Pipeline completed successfully", fg="green"))
                else:
                    click.echo(click.style("  ✗ Pipeline completed with errors", fg="red"))

            finally:
                await db.close()

        return exit_code

    result = asyncio.run(run())
    if result != 0:
        sys.exit(result)


@main.command()
@click.option("--days", default=90, help="Days of data to keep")
@click.option("--dry-run", is_flag=True, help="Show count without deleting")
def cleanup(days: int, dry_run: bool) -> None:
    """Remove documents older than specified days.

    Example:
        news-tracker cleanup --days 30              # Delete docs older than 30 days
        news-tracker cleanup --days 30 --dry-run   # Preview without deleting
    """
    from datetime import datetime, timedelta, timezone

    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

            if dry_run:
                # Count documents that would be deleted
                sql = "SELECT COUNT(*) FROM documents WHERE timestamp < $1"
                count = await db.fetchval(sql, cutoff)

                click.echo(f"\nDry run - would delete {count} documents older than {days} days")
                click.echo(f"Cutoff: {cutoff.isoformat()}")
                click.echo("\nRun without --dry-run to actually delete.")
            else:
                # Perform the deletion
                sql = """
                    DELETE FROM documents
                    WHERE timestamp < $1
                    RETURNING id
                """
                rows = await db.fetch(sql, cutoff)
                deleted = len(rows)

                click.echo(f"\nDeleted {deleted} documents older than {days} days")
                click.echo(f"Cutoff: {cutoff.isoformat()}")

        finally:
            await db.close()

    asyncio.run(run())


@main.command("sentiment-worker")
@click.option("--batch-size", default=None, type=int, help="Jobs to process per batch")
@click.option("--metrics/--no-metrics", default=True, help="Enable metrics server")
@click.option("--metrics-port", default=8001, help="Metrics server port")
def sentiment_worker(batch_size: int | None, metrics: bool, metrics_port: int) -> None:
    """Run the sentiment analysis worker.

    Consumes document IDs from Redis Streams sentiment queue,
    generates sentiment using FinBERT, and updates the database.

    Example:
        news-tracker sentiment-worker
        news-tracker sentiment-worker --batch-size 8
    """
    from src.sentiment.worker import SentimentWorker

    async def run():
        worker = SentimentWorker(batch_size=batch_size)

        if metrics:
            get_metrics().start_server(port=metrics_port)

        # Handle shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.stop()))

        await worker.start()

    asyncio.run(run())


@main.command("vector-search")
@click.argument("query")
@click.option("--limit", default=10, help="Maximum results to return")
@click.option("--threshold", default=0.7, type=float, help="Minimum similarity threshold")
@click.option("--platform", multiple=True, help="Filter by platform (can repeat)")
@click.option("--ticker", multiple=True, help="Filter by ticker (can repeat)")
@click.option("--min-authority", type=float, help="Minimum authority score")
def vector_search(
    query: str,
    limit: int,
    threshold: float,
    platform: tuple[str, ...],
    ticker: tuple[str, ...],
    min_authority: float | None,
) -> None:
    """Search for semantically similar documents.

    Example:
        news-tracker vector-search "NVIDIA AI demand" --limit 5
        news-tracker vector-search "semiconductor supply chain" --platform twitter --ticker NVDA
    """
    import redis.asyncio as redis
    from src.embedding.config import EmbeddingConfig
    from src.embedding.service import EmbeddingService
    from src.storage.database import Database
    from src.storage.repository import DocumentRepository
    from src.vectorstore.config import VectorStoreConfig
    from src.vectorstore.manager import VectorStoreManager
    from src.vectorstore.pgvector_store import PgVectorStore
    from src.vectorstore.base import VectorSearchFilter

    async def run():
        settings = get_settings()

        # Initialize database
        db = Database()
        await db.connect()

        # Initialize Redis for embedding cache
        redis_client = redis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )

        try:
            # Create services
            embedding_config = EmbeddingConfig(
                model_name=settings.embedding_model_name,
                batch_size=settings.embedding_batch_size,
                device=settings.embedding_device,
                cache_enabled=settings.embedding_cache_enabled,
            )
            embedding_service = EmbeddingService(
                config=embedding_config,
                redis_client=redis_client,
            )

            repository = DocumentRepository(db)
            vector_store = PgVectorStore(
                database=db,
                repository=repository,
            )
            manager = VectorStoreManager(
                vector_store=vector_store,
                embedding_service=embedding_service,
            )

            # Build filters
            filters = None
            if platform or ticker or min_authority is not None:
                filters = VectorSearchFilter(
                    platforms=list(platform) if platform else None,
                    tickers=list(ticker) if ticker else None,
                    min_authority_score=min_authority,
                )

            # Execute search
            click.echo(f"\nSearching for: {query}")
            click.echo("-" * 60)

            results = await manager.query(
                text=query,
                limit=limit,
                threshold=threshold,
                filters=filters,
            )

            if not results:
                click.echo("No results found.")
                return

            # Display results
            for i, result in enumerate(results, 1):
                meta = result.metadata
                platform_name = meta.get("platform", "unknown")
                title = meta.get("title") or meta.get("content_preview", "")[:50]
                score = result.score
                authority = meta.get("authority_score")
                tickers = meta.get("tickers", [])

                click.echo(f"\n{i}. [{platform_name}] {title}")
                click.echo(f"   Score: {score:.4f} | Authority: {authority or 'N/A'}")
                if tickers:
                    click.echo(f"   Tickers: {', '.join(tickers)}")
                click.echo(f"   ID: {result.document_id}")

            click.echo(f"\n{'-' * 60}")
            click.echo(f"Found {len(results)} results")

        finally:
            await embedding_service.close()
            await redis_client.close()
            await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
