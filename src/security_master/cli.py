"""CLI commands for security-master datasource maintenance."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import click


@click.command("ingest-nasdaq-trader")
@click.option(
    "--nasdaq-listed-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Local nasdaqlisted.txt file. If omitted, fetches the official Nasdaq Trader URL.",
)
@click.option(
    "--other-listed-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Local otherlisted.txt file. If omitted, fetches the official Nasdaq Trader URL.",
)
@click.option(
    "--observed-at",
    default=None,
    help="Optional ISO timestamp to stamp local-file reconciliation.",
)
def ingest_nasdaq_trader(
    nasdaq_listed_file: str | None,
    other_listed_file: str | None,
    observed_at: str | None,
) -> None:
    """Ingest Nasdaq Trader listed-security reference files."""
    local_files_requested = nasdaq_listed_file is not None or other_listed_file is not None
    if local_files_requested and (nasdaq_listed_file is None or other_listed_file is None):
        raise click.ClickException(
            "--nasdaq-listed-file and --other-listed-file must be provided together",
        )
    observed = _parse_optional_cli_datetime(observed_at)

    async def run() -> None:
        from src.security_master.service import SecurityMasterService
        from src.storage.database import Database

        db = Database()
        await db.connect()
        try:
            service = SecurityMasterService(db)
            if local_files_requested:
                result = await service.ingest_nasdaq_trader_symbol_directory(
                    Path(nasdaq_listed_file or "").read_text(encoding="utf-8"),
                    Path(other_listed_file or "").read_text(encoding="utf-8"),
                    observed_at=observed,
                )
            else:
                result = await service.refresh_nasdaq_trader_symbol_directory()

            click.echo("Nasdaq Trader symbol directory ingested")
            click.echo(f"  Current records: {result.current_record_count}")
            click.echo(f"  Active securities: {result.active_count}")
            click.echo(f"  Test issues: {result.test_issue_count}")
            click.echo(f"  Deactivated missing: {result.deactivated_missing_count}")
            click.echo(f"  Nasdaq-listed rows: {result.nasdaq_listed_count}")
            click.echo(f"  Other-listed rows: {result.other_listed_count}")
        finally:
            await db.close()

    asyncio.run(run())


def _parse_optional_cli_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise click.ClickException(f"Invalid ISO timestamp for --observed-at: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
