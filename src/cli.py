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
from datetime import UTC, datetime
from typing import Any

import click

from src.config.settings import get_settings
from src.observability.logging import setup_logging
from src.observability.metrics import get_metrics


async def _load_db_sources() -> tuple[
    list[str] | None, list[str] | None, list[tuple[str, str, str]] | None
]:
    """Load active sources from DB if the sources feature is enabled.

    Returns (twitter_sources, reddit_sources, substack_sources).
    All None when sources_enabled is False (falls back to adapter defaults).
    """
    settings = get_settings()
    if not settings.sources_enabled:
        return None, None, None

    from src.sources.service import SourcesService
    from src.storage.database import Database

    db = Database()
    await db.connect()
    try:
        svc = SourcesService(db)
        twitter = await svc.get_twitter_sources()
        reddit = await svc.get_reddit_sources()
        substack = await svc.get_substack_sources()
        return twitter, reddit, substack
    finally:
        await db.close()


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool) -> None:
    """News Tracker - Multi-platform financial data ingestion."""
    if debug:
        import os

        os.environ["LOG_LEVEL"] = "DEBUG"

    setup_logging()

    # Initialize tracing if enabled
    settings = get_settings()
    if settings.tracing_enabled:
        from src.observability.tracing import setup_tracing

        setup_tracing(
            service_name=settings.otel_service_name,
            otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        )


@main.command()
@click.option("--mock", is_flag=True, help="Use mock adapters")
@click.option("--metrics/--no-metrics", default=True, help="Enable metrics server")
def ingest(mock: bool, metrics: bool) -> None:
    """Run the ingestion service."""
    from src.services.ingestion_service import IngestionService

    async def run():
        twitter_src, reddit_src, substack_src = await _load_db_sources()
        service = IngestionService(
            use_mock=mock,
            twitter_sources=twitter_src,
            reddit_sources=reddit_src,
            substack_sources=substack_src,
        )

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
        twitter_src, reddit_src, substack_src = await _load_db_sources()
        ingestion = IngestionService(
            use_mock=mock,
            twitter_sources=twitter_src,
            reddit_sources=reddit_src,
            substack_sources=substack_src,
        )
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
    import pathlib

    from src.storage.database import Database
    from src.storage.migrations import apply_migrations

    async def run():
        db = Database()
        await db.connect()

        migrations_dir = pathlib.Path(__file__).resolve().parent.parent / "migrations"
        if migrations_dir.is_dir():
            applied = await apply_migrations(db)
            if applied:
                for migration_name in applied:
                    click.echo(f"Applied migration: {migration_name}")
            else:
                click.echo("No pending migrations")

        # Also seed sources if feature is enabled
        from src.config.settings import get_settings

        settings = get_settings()
        if settings.sources_enabled:
            from src.sources.service import SourcesService

            svc = SourcesService(db)
            await svc.ensure_seeded()
            click.echo("Sources table initialized and seeded")

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
        results["twitter_configured"] = settings.twitter_configured or settings.xui_configured
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
        twitter_src, reddit_src, substack_src = await _load_db_sources()
        ingestion = IngestionService(
            use_mock=mock,
            twitter_sources=twitter_src,
            reddit_sources=reddit_src,
            substack_sources=substack_src,
        )
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

            if embedding_results["errors"] > 0:
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

            if sentiment_results["errors"] > 0:
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
                    click.echo(
                        click.style(
                            f"  ✗ Only {total_count}/{len(stored_doc_ids)} documents in database",
                            fg="red",
                        )
                    )
                    exit_code = 1

                if with_embeddings:
                    if with_embedding_count == total_count:
                        click.echo(
                            click.style(
                                f"  ✓ {with_embedding_count} documents have embeddings", fg="green"
                            )
                        )
                    else:
                        click.echo(
                            click.style(
                                f"  ✗ Only {with_embedding_count}/{total_count}"
                                " documents have embeddings",
                                fg="red",
                            )
                        )
                        exit_code = 1

                if with_sentiment:
                    if with_sentiment_count == total_count:
                        click.echo(
                            click.style(
                                f"  ✓ {with_sentiment_count} documents have sentiment", fg="green"
                            )
                        )
                    else:
                        click.echo(
                            click.style(
                                f"  ✗ Only {with_sentiment_count}/{total_count}"
                                " documents have sentiment",
                                fg="red",
                            )
                        )
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
    from datetime import datetime, timedelta

    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            cutoff = datetime.now(UTC) - timedelta(days=days)

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


@main.command("embedding-worker")
@click.option("--batch-size", default=None, type=int, help="Jobs to process per batch")
@click.option("--metrics/--no-metrics", default=True, help="Enable metrics server")
@click.option("--metrics-port", default=8003, help="Metrics server port")
def embedding_worker(batch_size: int | None, metrics: bool, metrics_port: int) -> None:
    """Run the embedding generation worker.

    Consumes document IDs from Redis Streams embedding queue,
    generates FinBERT/MiniLM embeddings, and updates the database.
    Also forwards to clustering queue when clustering is enabled.

    Example:
        news-tracker embedding-worker
        news-tracker embedding-worker --batch-size 16
    """
    from src.embedding.worker import EmbeddingWorker

    async def run():
        worker = EmbeddingWorker(batch_size=batch_size)

        if metrics:
            get_metrics().start_server(port=metrics_port)

        # Handle shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.stop()))

        await worker.start()

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


@main.command("clustering-worker")
@click.option("--batch-size", default=None, type=int, help="Jobs to process per batch")
@click.option("--metrics/--no-metrics", default=True, help="Enable metrics server")
@click.option("--metrics-port", default=8002, help="Metrics server port")
def clustering_worker(batch_size: int | None, metrics: bool, metrics_port: int) -> None:
    """Run the clustering worker for real-time theme assignment.

    Consumes document IDs from Redis Streams clustering queue,
    finds similar theme centroids via pgvector HNSW, and assigns
    documents to matching themes.

    Example:
        news-tracker clustering-worker
        news-tracker clustering-worker --batch-size 16
    """
    from src.clustering.worker import ClusteringWorker

    async def run():
        worker = ClusteringWorker(batch_size=batch_size)

        if metrics:
            get_metrics().start_server(port=metrics_port)

        # Handle shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.stop()))

        await worker.start()

    asyncio.run(run())


@main.command("daily-clustering")
@click.option(
    "--date",
    "target_date",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Date to process (default: today UTC)",
)
@click.option("--dry-run", is_flag=True, help="Preview document count without running")
def daily_clustering(target_date: Any, dry_run: bool) -> None:
    """Run daily batch clustering for theme assignment and metrics.

    Re-assigns recent documents to themes, detects emerging themes,
    computes daily metrics, and runs weekly theme merges (Mondays).

    Designed for cron scheduling: 0 4 * * * news-tracker daily-clustering

    Example:
        news-tracker daily-clustering                    # Process today
        news-tracker daily-clustering --date 2026-02-05  # Process specific date
        news-tracker daily-clustering --dry-run          # Preview only
    """
    from datetime import timedelta

    from src.clustering.daily_job import run_daily_clustering
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            d = target_date.date() if target_date else None

            if dry_run:
                from src.storage.repository import DocumentRepository

                d = d or datetime.now(UTC).date()
                repo = DocumentRepository(db)
                since = datetime(d.year, d.month, d.day, tzinfo=UTC)
                until = since + timedelta(days=1)
                docs = await repo.get_with_embeddings_since(since, until)

                # Count existing themes
                from src.themes.repository import ThemeRepository

                theme_repo = ThemeRepository(db)
                themes = await theme_repo.get_all(limit=500)

                click.echo(f"\nDry run for {d}")
                click.echo(f"  Documents with embeddings: {len(docs)}")
                click.echo(f"  Existing themes: {len(themes)}")
                click.echo(
                    f"  Day of week: {d.strftime('%A')}{' (merge day)' if d.weekday() == 0 else ''}"
                )
                click.echo("\nRun without --dry-run to execute.")
            else:
                result = await run_daily_clustering(db, target_date=d)

                click.echo(f"\nDaily Clustering Results ({result.date}):")
                click.echo(f"  Documents fetched:  {result.documents_fetched}")
                click.echo(f"  Documents assigned: {result.documents_assigned}")
                click.echo(f"  Unassigned:         {result.documents_unassigned}")
                click.echo(f"  New themes created: {result.new_themes_created}")
                click.echo(f"  Themes merged:      {result.themes_merged}")
                click.echo(f"  Metrics computed:   {result.metrics_computed}")
                click.echo(f"  Errors:             {len(result.errors)}")
                click.echo(f"  Elapsed:            {result.elapsed_seconds:.2f}s")

                if result.errors:
                    click.echo("\nErrors:")
                    for err in result.errors:
                        click.echo(click.style(f"  - {err}", fg="red"))

        finally:
            await db.close()

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
    from src.vectorstore.base import VectorSearchFilter
    from src.vectorstore.manager import VectorStoreManager
    from src.vectorstore.pgvector_store import PgVectorStore

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
                backend=settings.embedding_backend,
                device=settings.embedding_device,
                execution_provider=settings.embedding_execution_provider,
                onnx_model_path=settings.embedding_onnx_model_path,
                onnx_minilm_model_path=settings.embedding_minilm_onnx_model_path,
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


@main.group()
def cluster() -> None:
    """Clustering management commands."""


@cluster.command("fit")
@click.option("--days", default=30, type=int, help="Days of historical data to use")
def cluster_fit(days: int) -> None:
    """Discover themes from recent documents using BERTopic.

    Fetches documents with FinBERT embeddings from the last N days,
    runs UMAP + HDBSCAN + c-TF-IDF, and persists discovered themes.

    Example:
        news-tracker cluster fit              # Last 30 days
        news-tracker cluster fit --days 7     # Last 7 days
    """
    from datetime import timedelta

    import numpy as np

    from src.clustering.config import ClusteringConfig
    from src.clustering.daily_job import _cluster_to_theme
    from src.clustering.service import BERTopicService
    from src.storage.database import Database
    from src.storage.repository import DocumentRepository
    from src.themes.repository import ThemeRepository

    async def run():
        db = Database()
        await db.connect()

        try:
            doc_repo = DocumentRepository(db)
            theme_repo = ThemeRepository(db)

            until = datetime.now(UTC)
            since = until - timedelta(days=days)

            click.echo(f"Fetching documents from last {days} days...")
            docs = await doc_repo.get_with_embeddings_since(since, until)

            if not docs:
                click.echo("No documents with embeddings found.")
                return

            click.echo(f"Found {len(docs)} documents with embeddings")

            # Extract texts and embeddings
            texts = [d["content"] for d in docs]
            embeddings = np.array([d["embedding"] for d in docs], dtype=np.float32)
            doc_ids = [d["id"] for d in docs]

            # Run BERTopic fit
            click.echo("Running BERTopic clustering...")
            config = ClusteringConfig()
            service = BERTopicService(config=config)
            theme_clusters = service.fit(texts, embeddings, doc_ids)

            if not theme_clusters:
                click.echo("No themes discovered.")
                return

            # Persist to DB
            click.echo(f"Persisting {len(theme_clusters)} themes...")
            created = 0
            for cluster in theme_clusters.values():
                theme = _cluster_to_theme(cluster)
                try:
                    await theme_repo.create(theme)
                    created += 1
                except Exception as e:
                    click.echo(click.style(f"  Failed to create {cluster.theme_id}: {e}", fg="red"))

            # Display results
            click.echo("\nResults:")
            click.echo(f"  Documents processed: {len(docs)}")
            click.echo(f"  Themes discovered:   {len(theme_clusters)}")
            click.echo(f"  Themes persisted:    {created}")

            click.echo("\nThemes:")
            for tc in theme_clusters.values():
                keywords = ", ".join(w for w, _ in tc.topic_words[:5])
                click.echo(f"  {tc.theme_id[:20]:20s}  {tc.document_count:4d} docs  [{keywords}]")

        finally:
            await db.close()

    asyncio.run(run())


@cluster.command("run")
@click.option(
    "--date",
    "target_date",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Date to process (default: today UTC)",
)
@click.option("--dry-run", is_flag=True, help="Preview document count without running")
def cluster_run(target_date: Any, dry_run: bool) -> None:
    """Run daily batch clustering for a specific date.

    Same as 'daily-clustering' but under the cluster group namespace.

    Example:
        news-tracker cluster run                    # Process today
        news-tracker cluster run --date 2026-02-05  # Specific date
        news-tracker cluster run --dry-run           # Preview only
    """
    from datetime import timedelta

    from src.clustering.daily_job import run_daily_clustering
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            d = target_date.date() if target_date else None

            if dry_run:
                from src.storage.repository import DocumentRepository
                from src.themes.repository import ThemeRepository

                d = d or datetime.now(UTC).date()
                repo = DocumentRepository(db)
                since = datetime(d.year, d.month, d.day, tzinfo=UTC)
                until = since + timedelta(days=1)
                docs = await repo.get_with_embeddings_since(since, until)

                theme_repo = ThemeRepository(db)
                themes = await theme_repo.get_all(limit=500)

                click.echo(f"\nDry run for {d}")
                click.echo(f"  Documents with embeddings: {len(docs)}")
                click.echo(f"  Existing themes: {len(themes)}")
                click.echo(
                    f"  Day of week: {d.strftime('%A')}{' (merge day)' if d.weekday() == 0 else ''}"
                )
                click.echo("\nRun without --dry-run to execute.")
            else:
                result = await run_daily_clustering(db, target_date=d)

                click.echo(f"\nDaily Clustering Results ({result.date}):")
                click.echo(f"  Documents fetched:  {result.documents_fetched}")
                click.echo(f"  Documents assigned: {result.documents_assigned}")
                click.echo(f"  Unassigned:         {result.documents_unassigned}")
                click.echo(f"  New themes created: {result.new_themes_created}")
                click.echo(f"  Themes merged:      {result.themes_merged}")
                click.echo(f"  Metrics computed:   {result.metrics_computed}")
                click.echo(f"  Errors:             {len(result.errors)}")
                click.echo(f"  Elapsed:            {result.elapsed_seconds:.2f}s")

                if result.errors:
                    click.echo("\nErrors:")
                    for err in result.errors:
                        click.echo(click.style(f"  - {err}", fg="red"))

        finally:
            await db.close()

    asyncio.run(run())


@cluster.command("backfill")
@click.option(
    "--start",
    "start_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (inclusive)",
)
@click.option(
    "--end",
    "end_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (inclusive)",
)
def cluster_backfill(start_date: Any, end_date: Any) -> None:
    """Run daily clustering for a range of dates.

    Processes each date sequentially. Continues on per-date errors
    but stops on fatal errors (DB connection lost).

    Example:
        news-tracker cluster backfill --start 2026-01-01 --end 2026-01-31
    """
    from datetime import timedelta

    from src.clustering.daily_job import run_daily_clustering
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            start = start_date.date()
            end = end_date.date()

            if start > end:
                click.echo(click.style("Error: start date must be before end date", fg="red"))
                return

            total_days = (end - start).days + 1
            click.echo(f"Backfilling {total_days} days: {start} to {end}")
            click.echo("-" * 50)

            success_count = 0
            error_count = 0
            current = start

            while current <= end:
                try:
                    result = await run_daily_clustering(db, target_date=current)
                    status = click.style("OK", fg="green")
                    detail = (
                        f"fetched={result.documents_fetched} "
                        f"assigned={result.documents_assigned} "
                        f"errors={len(result.errors)}"
                    )
                    click.echo(f"  {current}  {status}  {detail}")
                    success_count += 1

                except Exception as e:
                    error_msg = str(e)
                    status = click.style("FAIL", fg="red")
                    click.echo(f"  {current}  {status}  {error_msg}")
                    error_count += 1

                    # Stop on DB connection errors
                    if "connection" in error_msg.lower() or "pool" in error_msg.lower():
                        click.echo(click.style("\nFatal: DB connection lost, stopping.", fg="red"))
                        break

                current += timedelta(days=1)

            click.echo("-" * 50)
            click.echo(f"Done: {success_count} succeeded, {error_count} failed")

        finally:
            await db.close()

    asyncio.run(run())


@cluster.command("merge")
@click.option("--dry-run", is_flag=True, help="Show what would merge without persisting")
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Override similarity threshold for merge (default: 0.85)",
)
def cluster_merge(dry_run: bool, threshold: float | None) -> None:
    """Merge similar themes based on centroid similarity.

    Loads all themes, finds pairs with centroid similarity above
    the threshold, and merges the smaller into the larger.

    Example:
        news-tracker cluster merge                    # Merge with default threshold
        news-tracker cluster merge --dry-run          # Preview without persisting
        news-tracker cluster merge --threshold 0.80   # More aggressive merge
    """
    from src.clustering.config import ClusteringConfig
    from src.clustering.daily_job import _run_weekly_merge, _theme_to_cluster
    from src.clustering.service import BERTopicService
    from src.storage.database import Database
    from src.themes.repository import ThemeRepository

    async def run():
        db = Database()
        await db.connect()

        try:
            theme_repo = ThemeRepository(db)
            themes = await theme_repo.get_all(limit=500)

            if len(themes) < 2:
                click.echo(f"Only {len(themes)} theme(s) found — nothing to merge.")
                return

            config = ClusteringConfig()
            if threshold is not None:
                config.similarity_threshold_merge = threshold

            if dry_run:
                # Build service to find merge candidates without persisting
                service = BERTopicService(config=config)
                for theme in themes:
                    service._themes[theme.theme_id] = _theme_to_cluster(theme)
                service._initialized = True

                merge_results = service.merge_similar_themes()

                if not merge_results:
                    click.echo("No themes similar enough to merge.")
                    return

                click.echo(f"\nDry run — {len(merge_results)} merge(s) would occur:\n")
                for absorbed_id, survivor_id in merge_results:
                    absorbed = next((t for t in themes if t.theme_id == absorbed_id), None)
                    survivor = next((t for t in themes if t.theme_id == survivor_id), None)
                    a_name = absorbed.name if absorbed else absorbed_id
                    s_name = survivor.name if survivor else survivor_id
                    click.echo(f"  {a_name} → {s_name}")

                click.echo("\nRun without --dry-run to execute.")
            else:
                merge_count = await _run_weekly_merge(themes, config, theme_repo, db)

                if merge_count == 0:
                    click.echo("No themes similar enough to merge.")
                else:
                    click.echo(f"\nMerged {merge_count} theme(s) successfully.")

        finally:
            await db.close()

    asyncio.run(run())


@cluster.command("status")
def cluster_status() -> None:
    """Show clustering status and theme summary.

    Displays total theme count, lifecycle stage breakdown,
    and the time of the most recent theme update.

    Example:
        news-tracker cluster status
    """
    from src.storage.database import Database
    from src.themes.repository import ThemeRepository

    async def run():
        db = Database()
        await db.connect()

        try:
            theme_repo = ThemeRepository(db)
            themes = await theme_repo.get_all(limit=500)

            if not themes:
                click.echo("No themes found.")
                return

            # Lifecycle breakdown
            lifecycle_counts: dict[str, int] = {}
            total_docs = 0
            for theme in themes:
                stage = theme.lifecycle_stage
                lifecycle_counts[stage] = lifecycle_counts.get(stage, 0) + 1
                total_docs += theme.document_count

            # Most recent update
            last_updated = max(t.updated_at for t in themes)

            click.echo("\nClustering Status")
            click.echo("=" * 40)
            click.echo(f"  Total themes:     {len(themes)}")
            click.echo(f"  Total documents:  {total_docs}")
            click.echo(f"  Last updated:     {last_updated.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            click.echo("\n  Lifecycle Stages:")
            for stage in ("emerging", "accelerating", "mature", "fading"):
                count = lifecycle_counts.get(stage, 0)
                click.echo(f"    {stage:15s} {count}")

        finally:
            await db.close()

    asyncio.run(run())


@cluster.command("recompute-centroids")
def cluster_recompute_centroids() -> None:
    """Recompute theme centroids from document embeddings.

    For each theme, fetches all assigned document embeddings and
    recalculates the centroid as the mean embedding vector.

    Example:
        news-tracker cluster recompute-centroids
    """
    import numpy as np

    from src.storage.database import Database
    from src.themes.repository import ThemeRepository

    async def run():
        db = Database()
        await db.connect()

        try:
            theme_repo = ThemeRepository(db)
            themes = await theme_repo.get_all(limit=500)

            if not themes:
                click.echo("No themes found.")
                return

            click.echo(f"Recomputing centroids for {len(themes)} themes...")

            updated = 0
            skipped = 0

            for theme in themes:
                # Fetch embeddings for documents assigned to this theme
                rows = await db.fetch(
                    "SELECT embedding FROM documents "
                    "WHERE $1 = ANY(theme_ids) AND embedding IS NOT NULL",
                    theme.theme_id,
                )

                if not rows:
                    click.echo(f"  {theme.theme_id[:20]:20s}  skipped (no embeddings)")
                    skipped += 1
                    continue

                embeddings = np.array([list(row["embedding"]) for row in rows], dtype=np.float32)
                new_centroid = np.mean(embeddings, axis=0)
                await theme_repo.update_centroid(theme.theme_id, new_centroid)

                click.echo(f"  {theme.theme_id[:20]:20s}  updated ({len(rows)} docs)")
                updated += 1

            click.echo(f"\nDone: {updated} updated, {skipped} skipped")

        finally:
            await db.close()

    asyncio.run(run())


@main.group()
def narrative() -> None:
    """Narrative momentum commands."""


async def _build_narrative_worker_for_cli(db: Any) -> Any:
    """Create a narrative worker with only the dependencies needed for CLI jobs."""
    from src.alerts.config import AlertConfig
    from src.alerts.repository import AlertRepository
    from src.alerts.service import AlertService
    from src.narrative.config import NarrativeConfig
    from src.narrative.repository import NarrativeRepository
    from src.narrative.worker import NarrativeWorker
    from src.storage.repository import DocumentRepository

    worker = NarrativeWorker(database=db, config=NarrativeConfig())
    worker._doc_repo = DocumentRepository(db)
    worker._narrative_repo = NarrativeRepository(db)
    worker._alert_repo = AlertRepository(db)
    worker._alert_service = AlertService(
        config=AlertConfig(),
        alert_repo=worker._alert_repo,
        redis_client=None,
        dispatcher=None,
    )
    return worker


def _extract_narrative_tickers(trigger_data: dict[str, Any]) -> list[str]:
    """Normalize ticker payloads stored in narrative trigger data."""
    raw = trigger_data.get("top_tickers") or []
    tickers: list[str] = []

    for item in raw:
        if isinstance(item, str) and item:
            tickers.append(item.upper())
        elif isinstance(item, dict):
            ticker = item.get("ticker")
            if ticker:
                tickers.append(str(ticker).upper())
            elif item:
                first_key = next(iter(item))
                tickers.append(str(first_key).upper())

    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)
    return deduped


@narrative.command("worker")
@click.option("--batch-size", default=None, type=int, help="Jobs to process per batch")
@click.option("--metrics/--no-metrics", default=True, help="Enable metrics server")
@click.option("--metrics-port", default=8003, help="Metrics server port")
def narrative_worker(batch_size: int | None, metrics: bool, metrics_port: int) -> None:
    """Run the narrative worker for real-time run assignment and alerting."""
    from src.narrative.worker import NarrativeWorker

    async def run():
        worker = NarrativeWorker(batch_size=batch_size)

        if metrics:
            get_metrics().start_server(port=metrics_port)

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.stop()))

        await worker.start()

    asyncio.run(run())


@narrative.command("backfill")
@click.option(
    "--start",
    "start_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (inclusive)",
)
@click.option(
    "--end",
    "end_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (inclusive)",
)
@click.option(
    "--reset/--no-reset", default=True, help="Truncate narrative tables before rebuilding"
)
def narrative_backfill(start_date: Any, end_date: Any, reset: bool) -> None:
    """Rebuild narrative runs from theme-assigned documents in a date range."""
    from datetime import timedelta

    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            start = start_date.date()
            end = end_date.date()
            if start > end:
                click.echo(click.style("Error: start date must be before end date", fg="red"))
                return

            if reset:
                await db.execute(
                    """
                    TRUNCATE TABLE
                        narrative_signal_state,
                        narrative_run_documents,
                        narrative_run_buckets,
                        narrative_runs
                    """
                )

            worker = await _build_narrative_worker_for_cli(db)
            start_ts = datetime(start.year, start.month, start.day, tzinfo=UTC)
            end_ts = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
            rows = await db.fetch(
                """
                SELECT id, theme_ids, timestamp
                FROM documents
                WHERE embedding IS NOT NULL
                  AND cardinality(theme_ids) > 0
                  AND timestamp >= $1
                  AND timestamp < $2
                ORDER BY timestamp ASC
                """,
                start_ts,
                end_ts,
            )

            click.echo(f"Backfilling narrative runs for {start} to {end}")
            click.echo(f"  Documents queued: {len(rows)}")
            click.echo(f"  Reset state: {'yes' if reset else 'no'}")

            processed_docs = 0
            processed_assignments = 0
            for row in rows:
                doc = await worker._doc_repo.get_by_id(row["id"])
                if doc is None:
                    continue

                processed_docs += 1
                for theme_id in row["theme_ids"]:
                    run = await worker.process_document_for_theme(
                        doc,
                        theme_id=theme_id,
                        theme_similarity=1.0,
                        publish_alerts=False,
                    )
                    if run is not None:
                        processed_assignments += 1

                if processed_docs % 250 == 0:
                    click.echo(f"  Processed {processed_docs} documents...")

            total_runs = await db.fetchval("SELECT COUNT(*) FROM narrative_runs")
            click.echo("")
            click.echo(f"Documents processed:  {processed_docs}")
            click.echo(f"Theme assignments:    {processed_assignments}")
            click.echo(f"Narrative runs built: {total_runs}")

        finally:
            await db.close()

    asyncio.run(run())


@narrative.command("replay")
@click.option(
    "--publish/--dry-run",
    default=False,
    help="Persist alerts while replaying instead of reporting only",
)
@click.option(
    "--status",
    "run_status",
    default=None,
    type=click.Choice(["active", "cooling", "closed"]),
    help="Optional run status filter",
)
@click.option("--limit", default=None, type=int, help="Maximum runs to replay")
def narrative_replay(publish: bool, run_status: str | None, limit: int | None) -> None:
    """Re-run signal evaluation from existing narrative runs and buckets."""
    from types import SimpleNamespace

    from src.narrative.config import NarrativeConfig
    from src.narrative.signals import evaluate_all_signals
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            worker = await _build_narrative_worker_for_cli(db)
            repo = worker._narrative_repo
            config = NarrativeConfig()
            params: list[Any] = []
            where_clause = ""
            if run_status:
                where_clause = "WHERE status = $1"
                params.append(run_status)
            limit_clause = ""
            if limit is not None:
                limit_clause = f" LIMIT ${len(params) + 1}"
                params.append(limit)

            rows = await db.fetch(
                f"""
                SELECT run_id
                FROM narrative_runs
                {where_clause}
                ORDER BY updated_at DESC
                {limit_clause}
                """,
                *params,
            )

            replayed = 0
            triggered_counts: dict[str, int] = {}
            published_alerts = 0

            for row in rows:
                run = await repo.get_by_id(row["run_id"])
                if run is None:
                    continue
                buckets = await repo.get_recent_buckets(run.run_id, limit=288)
                evaluations = evaluate_all_signals(run, buckets, config)
                replayed += 1

                for evaluation in evaluations:
                    if evaluation.triggered:
                        triggered_counts[evaluation.trigger_type] = (
                            triggered_counts.get(evaluation.trigger_type, 0) + 1
                        )

                if publish:
                    docs = await repo.get_run_documents(run.run_id, limit=1)
                    latest_doc_id = docs[0]["document_id"] if docs else run.run_id
                    before_count = await db.fetchval(
                        "SELECT COUNT(*) FROM alerts WHERE subject_type = 'narrative_run'"
                    )
                    await worker._evaluate_and_publish_alerts(
                        run,
                        buckets,
                        SimpleNamespace(id=latest_doc_id),
                    )
                    after_count = await db.fetchval(
                        "SELECT COUNT(*) FROM alerts WHERE subject_type = 'narrative_run'"
                    )
                    published_alerts += max(0, int(after_count or 0) - int(before_count or 0))

            click.echo(f"Runs replayed: {replayed}")
            for trigger_type, count in sorted(triggered_counts.items()):
                click.echo(f"  {trigger_type:26s} {count}")
            if publish:
                click.echo(f"Alerts published: {published_alerts}")
            else:
                click.echo("Dry run only; no alerts persisted.")

        finally:
            await db.close()

    asyncio.run(run())


@narrative.command("evaluate")
@click.option(
    "--start",
    "start_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (inclusive)",
)
@click.option(
    "--end",
    "end_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (inclusive)",
)
@click.option("--horizon", default=5, type=int, help="Forward return horizon in trading days")
def narrative_evaluate(start_date: Any, end_date: Any, horizon: int) -> None:
    """Score historical narrative alerts using cached forward returns."""
    import json
    from collections import defaultdict
    from datetime import timedelta

    from src.backtest.config import BacktestConfig
    from src.backtest.data_feeds import PriceDataFeed
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            start = start_date.date()
            end = end_date.date()
            if start > end:
                click.echo(click.style("Error: start date must be before end date", fg="red"))
                return

            start_ts = datetime(start.year, start.month, start.day, tzinfo=UTC)
            end_ts = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
            rows = await db.fetch(
                """
                SELECT trigger_type, created_at, conviction_score, trigger_data
                FROM alerts
                WHERE subject_type = 'narrative_run'
                  AND created_at >= $1
                  AND created_at < $2
                ORDER BY created_at ASC
                """,
                start_ts,
                end_ts,
            )

            if not rows:
                click.echo("No narrative alerts found in the requested range.")
                return

            feed = PriceDataFeed(
                db,
                BacktestConfig(
                    price_cache_enabled=False,
                    default_forward_horizons=[horizon],
                ),
            )
            by_trigger: dict[str, list[dict[str, float]]] = defaultdict(list)
            scored_records: list[dict[str, float]] = []

            for row in rows:
                trigger_data = row["trigger_data"]
                if isinstance(trigger_data, str):
                    trigger_data = json.loads(trigger_data)

                tickers = _extract_narrative_tickers(trigger_data)
                if not tickers:
                    continue

                returns = await feed.get_forward_returns(
                    tickers=tickers,
                    as_of=row["created_at"].date(),
                    horizons=[horizon],
                )
                realized = [
                    horizon_returns.get(horizon)
                    for horizon_returns in returns.values()
                    if horizon_returns.get(horizon) is not None
                ]
                if not realized:
                    continue

                avg_return = sum(realized) / len(realized)
                record = {
                    "avg_return": avg_return,
                    "score": float(row["conviction_score"] or 0.0),
                }
                by_trigger[row["trigger_type"]].append(record)
                scored_records.append(record)

            if not by_trigger:
                click.echo("No cached forward returns available for the requested alerts.")
                return

            click.echo(f"Narrative signal evaluation ({start} to {end}, {horizon}d horizon)")
            click.echo("=" * 72)
            click.echo(f"{'Trigger':26s} {'Count':>6} {'Hit Rate':>10} {'Mean Return':>14}")
            click.echo("-" * 72)

            for trigger_type, records in sorted(by_trigger.items()):
                count = len(records)
                hit_rate = sum(1 for r in records if r["avg_return"] > 0) / count
                mean_return = sum(r["avg_return"] for r in records) / count
                click.echo(f"{trigger_type:26s} {count:>6d} {hit_rate:>9.1%} {mean_return:>13.2%}")

            scored_records.sort(key=lambda item: item["score"])
            if scored_records:
                click.echo("")
                click.echo("Conviction calibration")
                click.echo("-" * 72)
                buckets = [("low", 0.0, 0.33), ("mid", 0.33, 0.67), ("high", 0.67, 1.01)]
                total = len(scored_records)
                for label, lower, upper in buckets:
                    start_idx = int(total * lower)
                    end_idx = int(total * upper)
                    bucket = scored_records[start_idx:end_idx]
                    if not bucket:
                        continue
                    hit_rate = sum(1 for r in bucket if r["avg_return"] > 0) / len(bucket)
                    mean_return = sum(r["avg_return"] for r in bucket) / len(bucket)
                    avg_score = sum(r["score"] for r in bucket) / len(bucket)
                    click.echo(
                        f"{label:>4s}  count={len(bucket):>4d}  avg_score={avg_score:>6.1f}  "
                        f"hit_rate={hit_rate:>6.1%}  mean_return={mean_return:>7.2%}"
                    )

        finally:
            await db.close()

    asyncio.run(run())


@main.group()
def backtest() -> None:
    """Backtest commands."""


@backtest.command("run")
@click.option(
    "--start",
    "start_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (inclusive)",
)
@click.option(
    "--end",
    "end_date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (inclusive)",
)
@click.option(
    "--strategy",
    default="swing",
    type=click.Choice(["swing", "position"]),
    help="Ranking strategy (default: swing)",
)
@click.option("--top-n", default=10, type=int, help="Number of top themes per day (default: 10)")
@click.option(
    "--horizon", default=5, type=int, help="Forward return horizon in trading days (default: 5)"
)
def backtest_run(
    start_date: Any,
    end_date: Any,
    strategy: str,
    top_n: int,
    horizon: int,
) -> None:
    """Run a historical backtest simulation.

    Iterates over trading days, ranks themes using point-in-time data,
    collects tickers from top themes, measures forward returns, and
    computes performance metrics.

    Example:
        news-tracker backtest run --start 2025-01-01 --end 2025-06-30
        news-tracker backtest run --start 2025-01-01 --end 2025-06-30 \\
            --strategy position --horizon 20
    """
    from src.backtest.engine import BacktestEngine
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            engine = BacktestEngine(db)
            start = start_date.date()
            end = end_date.date()

            if start > end:
                click.echo(click.style("Error: start date must be before end date", fg="red"))
                return

            click.echo(f"Running backtest: {start} to {end}")
            click.echo(f"  Strategy: {strategy}")
            click.echo(f"  Top N: {top_n}")
            click.echo(f"  Horizon: {horizon} trading days")
            click.echo("")

            results = await engine.run_backtest(
                start_date=start,
                end_date=end,
                strategy=strategy,
                top_n=top_n,
                horizon=horizon,
            )

            # Summary table
            click.echo(f"Backtest Results ({results.run_id})")
            click.echo("=" * 50)
            click.echo(f"  Trading days:   {results.trading_days}")
            click.echo(f"  Hit rate:       {_fmt_pct(results.hit_rate)}")
            click.echo(f"  Mean return:    {_fmt_pct(results.mean_return)}")
            click.echo(f"  Total return:   {_fmt_pct(results.total_return)}")
            click.echo(f"  Volatility:     {_fmt_pct(results.volatility)}")
            click.echo(f"  Sharpe ratio:   {_fmt_float(results.sharpe_ratio)}")
            click.echo(f"  Sortino ratio:  {_fmt_float(results.sortino_ratio)}")
            click.echo(f"  Max drawdown:   {_fmt_pct(results.max_drawdown)}")
            click.echo(f"  Win rate:       {_fmt_pct(results.win_rate)}")
            click.echo(f"  Profit factor:  {_fmt_float(results.profit_factor)}")

            if results.calibration:
                click.echo("\n  Calibration Buckets:")
                hdr = f"  {'Bucket':<25} {'Count':>6} {'Avg Score':>10}"
                hdr += f" {'Avg Return':>11} {'Hit Rate':>9}"
                click.echo(hdr)
                click.echo(f"  {'-' * 25} {'-' * 6} {'-' * 10} {'-' * 11} {'-' * 9}")
                for b in results.calibration:
                    click.echo(
                        f"  {b['bucket_label']:<25} {b['count']:>6} "
                        f"{b['avg_score']:>10.4f} {b['avg_return']:>10.4%} "
                        f"{b['hit_rate']:>8.1%}"
                    )

        finally:
            await db.close()

    asyncio.run(run())


@backtest.command("plot")
@click.option("--run-id", required=True, help="Backtest run ID to visualize")
@click.option(
    "--output-dir",
    default="./backtest_plots",
    help="Directory to save plots (default: ./backtest_plots)",
)
def backtest_plot(run_id: str, output_dir: str) -> None:
    """Generate visualization charts for a completed backtest run.

    Loads the stored results from a previous backtest run and produces
    four charts: cumulative returns, drawdown, score vs return scatter,
    and monthly performance heatmap.

    Example:
        news-tracker backtest plot --run-id run_abc123def456
        news-tracker backtest plot --run-id run_abc123def456 --output-dir ./my_plots
    """
    from datetime import date as date_type

    from src.backtest.audit import BacktestRunRepository
    from src.backtest.engine import BacktestResults, DailyBacktestResult
    from src.backtest.visualization import BacktestVisualizer
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            repo = BacktestRunRepository(db)
            bt_run = await repo.get_by_id(run_id)

            if bt_run is None:
                click.echo(click.style(f"Error: Backtest run '{run_id}' not found", fg="red"))
                return

            if bt_run.status != "completed":
                click.echo(
                    click.style(
                        f"Error: Backtest run '{run_id}' has status"
                        f" '{bt_run.status}' (expected 'completed')",
                        fg="red",
                    )
                )
                return

            if not bt_run.results:
                click.echo(
                    click.style(f"Error: Backtest run '{run_id}' has no results data", fg="red")
                )
                return

            # Reconstruct BacktestResults from stored JSONB
            data = bt_run.results
            daily_results = []
            for dr in data.get("daily_results", []):
                daily_results.append(
                    DailyBacktestResult(
                        date=date_type.fromisoformat(dr["date"])
                        if isinstance(dr["date"], str)
                        else dr["date"],
                        top_n_tickers=dr.get("top_n_tickers", []),
                        top_n_avg_return=dr.get("top_n_avg_return"),
                        direction_correct=dr.get("direction_correct"),
                        theme_count=dr.get("theme_count", 0),
                        ranked_themes=dr.get("ranked_themes", []),
                    )
                )

            results = BacktestResults(
                run_id=data.get("run_id", run_id),
                strategy=data.get("strategy", bt_run.parameters.get("strategy", "swing")),
                horizon=data.get("horizon", bt_run.parameters.get("horizon", 5)),
                top_n=data.get("top_n", bt_run.parameters.get("top_n", 10)),
                trading_days=data.get("trading_days", len(daily_results)),
                daily_results=daily_results,
                hit_rate=data.get("hit_rate"),
                mean_return=data.get("mean_return"),
                total_return=data.get("total_return"),
                max_drawdown=data.get("max_drawdown"),
            )

            click.echo(f"Generating plots for backtest run '{run_id}'...")
            paths = BacktestVisualizer.generate_all(results, output_dir)

            if paths:
                click.echo(click.style(f"\nGenerated {len(paths)} chart(s):", fg="green"))
                for p in paths:
                    click.echo(f"  {p}")
            else:
                click.echo(click.style("No charts generated (insufficient data)", fg="yellow"))

        finally:
            await db.close()

    asyncio.run(run())


def _fmt_pct(value: float | None) -> str:
    """Format a float as percentage or N/A."""
    if value is None:
        return "N/A"
    return f"{value:.4%}"


def _fmt_float(value: float | None) -> str:
    """Format a float to 4 decimal places or N/A."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


@main.group()
def graph() -> None:
    """Causal graph management commands."""


@graph.command("seed")
def graph_seed() -> None:
    """Seed the causal graph with semiconductor supply chain data.

    Populates ~50 nodes and ~100+ edges covering foundry supply chains,
    equipment suppliers, memory suppliers, EDA/IP, competition, technology
    dependencies, and demand drivers.

    Idempotent: safe to re-run (uses ON CONFLICT upserts).

    Example:
        news-tracker graph seed
    """
    from src.graph.seed_data import seed_graph
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            result = await seed_graph(db)

            click.echo(f"\nGraph Seed Results (v{result['seed_version']}):")
            click.echo(f"  Nodes seeded: {result['node_count']}")
            click.echo(f"  Edges seeded: {result['edge_count']}")
            click.echo(click.style("\nGraph seeded successfully!", fg="green"))

        finally:
            await db.close()

    asyncio.run(run())


@graph.command("sync")
def graph_sync() -> None:
    """Sync assertion-derived edges into the causal graph.

    Reads resolved assertions from the intelligence layer, derives
    graph edges, and persists them to causal_edges.  Seed edges are
    treated as bootstrap priors; evidence-backed edges with sufficient
    support override them.

    Idempotent: safe to re-run (uses ON CONFLICT upserts).

    Example:
        news-tracker graph sync
    """
    from src.graph.sync import GraphSyncService
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        try:
            service = GraphSyncService(db)
            result = await service.sync()

            click.echo("\nGraph Sync Results:")
            click.echo(f"  Assertions read:  {result.assertions_read}")
            click.echo(f"  Edges derived:    {result.edges_derived}")
            click.echo(f"  Edges synced:     {result.edges_synced}")
            click.echo(f"  Edges removed:    {result.edges_removed}")
            click.echo(f"  Edges skipped:    {result.edges_skipped}")
            if result.errors:
                click.echo(f"  Errors:           {len(result.errors)}")
                for err in result.errors[:5]:
                    click.echo(f"    - {err}")

            click.echo(click.style("\nGraph sync complete!", fg="green"))

        finally:
            await db.close()

    asyncio.run(run())


@main.group()
def drift() -> None:
    """Drift detection and monitoring commands."""


@drift.command("check-quick")
def drift_check_quick() -> None:
    """Run quick embedding drift check (hourly cron).

    Compares L2 norm distribution of recent vs baseline embeddings
    using KL divergence.

    Example:
        news-tracker drift check-quick
    """
    from src.monitoring.config import DriftConfig
    from src.monitoring.service import DriftService
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        metrics = get_metrics()
        metrics.start_server()

        try:
            service = DriftService(db, DriftConfig())
            report = await service.run_quick_check()
            _display_drift_report(report)

            metrics.record_drift_check(report)

            # Allow one Prometheus scrape cycle before exiting
            await asyncio.sleep(16)

        finally:
            await db.close()

    asyncio.run(run())


@drift.command("check-daily")
def drift_check_daily() -> None:
    """Run all four drift checks (daily cron).

    Checks embedding drift, theme fragmentation, sentiment
    calibration, and cluster stability.

    Example:
        news-tracker drift check-daily
    """
    from src.monitoring.config import DriftConfig
    from src.monitoring.service import DriftService
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        metrics = get_metrics()
        metrics.start_server()

        try:
            service = DriftService(db, DriftConfig())
            report = await service.run_daily_check()
            _display_drift_report(report)

            metrics.record_drift_check(report)

            # Allow one Prometheus scrape cycle before exiting
            await asyncio.sleep(16)

        finally:
            await db.close()

    asyncio.run(run())


@drift.command("report")
def drift_report() -> None:
    """Run weekly drift report with verbose output.

    Same checks as daily but with detailed metadata output.

    Example:
        news-tracker drift report
    """
    from src.monitoring.config import DriftConfig
    from src.monitoring.service import DriftService
    from src.storage.database import Database

    async def run():
        db = Database()
        await db.connect()

        metrics = get_metrics()
        metrics.start_server()

        try:
            service = DriftService(db, DriftConfig())
            report = await service.run_weekly_report()
            _display_drift_report(report, verbose=True)

            metrics.record_drift_check(report)

            # Allow one Prometheus scrape cycle before exiting
            await asyncio.sleep(16)

        finally:
            await db.close()

    asyncio.run(run())


def _display_drift_report(report: Any, verbose: bool = False) -> None:
    """Format and display a DriftReport to the terminal.

    Args:
        report: DriftReport instance.
        verbose: If True, show metadata details for each check.
    """
    severity_colors = {"ok": "green", "warning": "yellow", "critical": "red"}
    severity_icons = {"ok": "OK", "warning": "WARN", "critical": "CRIT"}

    click.echo(f"\nDrift Report ({len(report.results)} check(s))")
    click.echo("=" * 50)

    for result in report.results:
        icon = severity_icons.get(result.severity, "?")
        color = severity_colors.get(result.severity, "white")
        label = result.drift_type.replace("_", " ").title()

        click.echo(click.style(f"  [{icon:4s}] ", fg=color) + f"{label}: {result.message}")

        if verbose and result.metadata:
            for key, val in result.metadata.items():
                click.echo(f"         {key}: {val}")

    click.echo("=" * 50)
    overall_color = severity_colors.get(report.overall_severity, "white")
    click.echo(
        "Overall: " + click.style(report.overall_severity.upper(), fg=overall_color, bold=True)
    )

    if report.has_issues:
        click.echo(
            click.style(
                "  Action recommended — review drift details above.",
                fg="yellow",
            )
        )


if __name__ == "__main__":
    main()
