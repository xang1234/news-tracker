"""CLI commands for market-structure datasource maintenance."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import click

from src.market_structure.models import MarketStructureSourceFile
from src.market_structure.service import MarketStructureIngestionService


@click.command("ingest-market-structure")
@click.option(
    "--finra-short-volume-file",
    "finra_short_volume_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Local FINRA daily short-volume pipe-delimited file. Can be repeated.",
)
@click.option(
    "--sec-fails-file",
    "sec_fails_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Local SEC fails-to-deliver pipe-delimited file. Can be repeated.",
)
@click.option(
    "--fetched-at",
    default=None,
    help="Optional ISO timestamp to stamp local-file fetch lineage.",
)
def ingest_market_structure(
    finra_short_volume_files: tuple[str, ...],
    sec_fails_files: tuple[str, ...],
    fetched_at: str | None,
) -> None:
    """Ingest FINRA daily short-volume and SEC fails-to-deliver files."""
    if not finra_short_volume_files and not sec_fails_files:
        raise click.ClickException(
            "Provide at least one --finra-short-volume-file or --sec-fails-file",
        )
    fetched = _parse_optional_cli_datetime(fetched_at)
    finra_sources = [
        _source_file(Path(path), source_name="FINRA daily short-volume", fetched_at=fetched)
        for path in finra_short_volume_files
    ]
    sec_sources = [
        _source_file(Path(path), source_name="SEC fails-to-deliver", fetched_at=fetched)
        for path in sec_fails_files
    ]

    async def run() -> None:
        from src.storage.database import Database

        db = Database()
        await db.connect()
        try:
            service = MarketStructureIngestionService(db)
            result = await service.ingest_source_files(
                finra_short_volume_files=finra_sources,
                sec_fails_to_deliver_files=sec_sources,
            )
            click.echo("Market-structure files ingested")
            click.echo(f"  Total events: {result.total_events}")
            click.echo(f"  FINRA short-volume events: {result.finra_short_volume_count}")
            click.echo(f"  SEC fails-to-deliver events: {result.sec_fails_to_deliver_count}")
            click.echo(f"  Upserted events: {result.upserted_count}")
            click.echo(f"  Unresolved symbols: {result.unresolved_symbol_count}")
        finally:
            await db.close()

    asyncio.run(run())


def _source_file(
    path: Path,
    *,
    source_name: str,
    fetched_at: datetime | None,
) -> MarketStructureSourceFile:
    resolved = path.resolve()
    return MarketStructureSourceFile(
        source_name=f"{source_name}: {path.name}",
        source_url=resolved.as_uri(),
        content=resolved.read_text(encoding="utf-8"),
        fetched_at=fetched_at,
    )


def _parse_optional_cli_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise click.ClickException(f"Invalid ISO timestamp for --fetched-at: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
