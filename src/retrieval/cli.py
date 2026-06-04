"""CLI commands for the claim retrieval substrate.

Registered into the top-level ``news-tracker`` CLI via
``main.add_command(claim_retrieval)`` (see ``src/cli.py``), matching the
per-module CLI convention used by factors / security-master / market-structure.
"""

from __future__ import annotations

import asyncio

import click

from src.config.settings import get_settings


def _build_embedding_service(settings, redis_client):
    """Construct an EmbeddingService from settings (shared by retrieval cmds)."""
    from src.embedding.config import EmbeddingConfig
    from src.embedding.service import EmbeddingService

    config = EmbeddingConfig(
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
        backend=settings.embedding_backend,
        device=settings.embedding_device,
        execution_provider=settings.embedding_execution_provider,
        onnx_model_path=settings.embedding_onnx_model_path,
        onnx_minilm_model_path=settings.embedding_minilm_onnx_model_path,
        cache_enabled=settings.embedding_cache_enabled,
    )
    return EmbeddingService(config=config, redis_client=redis_client)


@click.group("claim-retrieval")
def claim_retrieval() -> None:
    """Semantic retrieval over the structured evidence-claim layer."""


@claim_retrieval.command("index")
@click.option("--limit", default=128, help="Max un-embedded claims to index this run")
@click.option("--all", "index_all", is_flag=True, help="Loop until no claims remain un-embedded")
def claim_retrieval_index(limit: int, index_all: bool) -> None:
    """Backfill retrieval embeddings for claims that lack one.

    Idempotent: only claims with a NULL embedding are picked up.

    Example:
        news-tracker claim-retrieval index --limit 500
        news-tracker claim-retrieval index --all
    """
    import redis.asyncio as redis

    from src.retrieval.service import ClaimRetrievalService
    from src.storage.database import Database

    async def run():
        settings = get_settings()
        db = Database()
        await db.connect()
        redis_client = redis.from_url(
            str(settings.redis_url), encoding="utf-8", decode_responses=True
        )
        embedding_service = _build_embedding_service(settings, redis_client)
        try:
            service = ClaimRetrievalService(database=db, embedding_service=embedding_service)
            total = 0
            while True:
                indexed = await service.index_pending(limit=limit)
                total += indexed
                click.echo(f"  Indexed {indexed} claim(s)")
                if not index_all or indexed == 0:
                    break
            click.echo(click.style(f"\nDone — indexed {total} claim(s) total.", fg="green"))
        finally:
            await embedding_service.close()
            await redis_client.close()
            await db.close()

    asyncio.run(run())


@claim_retrieval.command("search")
@click.argument("query")
@click.option("--limit", default=10, help="Maximum claims to return")
@click.option("--lane", multiple=True, help="Filter by lane (can repeat)")
@click.option("--theme", help="Filter to claims from documents in this theme")
@click.option("--min-confidence", type=float, help="Minimum claim confidence")
def claim_retrieval_search(
    query: str,
    limit: int,
    lane: tuple[str, ...],
    theme: str | None,
    min_confidence: float | None,
) -> None:
    """Retrieve the top-K verified claims most relevant to a query.

    Example:
        news-tracker claim-retrieval search "TSMC capex guidance" --limit 5
        news-tracker claim-retrieval search "who supplies NVIDIA" --lane narrative
    """
    import redis.asyncio as redis

    from src.retrieval.schemas import ClaimRetrievalFilter
    from src.retrieval.service import ClaimRetrievalService
    from src.storage.database import Database

    async def run():
        settings = get_settings()
        db = Database()
        await db.connect()
        redis_client = redis.from_url(
            str(settings.redis_url), encoding="utf-8", decode_responses=True
        )
        embedding_service = _build_embedding_service(settings, redis_client)
        try:
            service = ClaimRetrievalService(database=db, embedding_service=embedding_service)
            filters = ClaimRetrievalFilter(
                lanes=list(lane) if lane else None,
                theme_id=theme,
                min_confidence=min_confidence,
            )

            click.echo(f"\nRetrieving claims for: {query}")
            click.echo("-" * 60)

            results = await service.retrieve(query, limit=limit, filters=filters)
            if not results:
                click.echo("No claims found.")
                return

            for i, result in enumerate(results, 1):
                c = result.claim
                obj = f" → {c.object_text}" if c.object_text else ""
                click.echo(f"\n{i}. {c.subject_text} [{c.predicate}]{obj}")
                click.echo(
                    f"   Score: {result.score:.4f} | "
                    f"Confidence: {c.confidence:.2f} | Lane: {c.lane}"
                )
                click.echo(f"   Source: {c.source_type}:{c.source_id} | Claim: {c.claim_id}")

            click.echo(f"\n{'-' * 60}")
            click.echo(f"Found {len(results)} claim(s)")
        finally:
            await embedding_service.close()
            await redis_client.close()
            await db.close()

    asyncio.run(run())
