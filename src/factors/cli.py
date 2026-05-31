"""CLI commands for factor datasource operations."""

from __future__ import annotations

import asyncio
from datetime import date

import click

from src.factors.refresh import (
    FactorRefreshSummary,
    UnknownFactorSelectorError,
    refresh_curated_factor_series,
    validate_factor_refresh_selectors,
)


@click.group(name="factors")
def factors() -> None:
    """Manage curated macro and supply-chain factor datasources."""


@factors.command("refresh")
@click.option(
    "--provider",
    "providers",
    multiple=True,
    help="Provider to refresh (repeatable: fred, bls, bea, treasury, fed, eia, census)",
)
@click.option("--factor-id", "factor_ids", multiple=True, help="Specific factor_id to refresh")
@click.option("--start", "start_date", help="Observation start date (YYYY-MM-DD)")
@click.option("--end", "end_date", help="Observation end date (YYYY-MM-DD)")
@click.option("--latest/--history", default=True, help="Fetch latest observation only or history")
@click.option("--dry-run", is_flag=True, help="Show selected series without writing observations")
def factors_refresh(
    providers: tuple[str, ...],
    factor_ids: tuple[str, ...],
    start_date: str | None,
    end_date: str | None,
    latest: bool,
    dry_run: bool,
) -> None:
    """Refresh curated factor registry entries and observations."""

    async def run() -> None:
        from src.storage.database import Database

        start = _parse_cli_date(start_date, "--start")
        end = _parse_cli_date(end_date, "--end")
        if start and end and start > end:
            raise click.ClickException("start date must be before end date")

        provider_set = {provider.lower() for provider in providers}
        factor_id_set = set(factor_ids)
        try:
            validate_factor_refresh_selectors(
                providers=provider_set,
                factor_ids=factor_id_set,
            )
        except UnknownFactorSelectorError as exc:
            raise click.ClickException(str(exc)) from exc

        db = Database()
        await db.connect()
        try:
            summary = await refresh_curated_factor_series(
                db,
                providers=provider_set,
                factor_ids=factor_id_set,
                start=start,
                end=end,
                latest=latest,
                dry_run=dry_run,
            )
        finally:
            await db.close()

        _print_refresh_summary(summary)
        if _refresh_failed(summary):
            raise click.ClickException(_refresh_failure_message(summary))

    asyncio.run(run())


def _parse_cli_date(value: str | None, option_name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise click.BadParameter("must use YYYY-MM-DD", param_hint=option_name) from exc


def _print_refresh_summary(summary: FactorRefreshSummary) -> None:
    click.echo("Factor Refresh Results")
    click.echo(f"Series refreshed: {summary.series_refreshed}/{summary.series_seen}")
    click.echo(f"Observations seen: {summary.observations_seen}")
    click.echo(f"Observations written: {summary.observations_written}")
    if summary.skipped_missing_credentials:
        click.echo(f"Skipped missing credentials: {len(summary.skipped_missing_credentials)}")
    if summary.errors:
        click.echo(f"Errors: {len(summary.errors)}")
    if summary.dry_run:
        click.echo("Dry run only")


def _refresh_failed(summary: FactorRefreshSummary) -> bool:
    if summary.dry_run:
        return False
    if summary.errors:
        return True
    return (
        summary.series_seen > 0
        and summary.series_refreshed == 0
        and bool(summary.skipped_missing_credentials)
    )


def _refresh_failure_message(summary: FactorRefreshSummary) -> str:
    if summary.errors:
        return "Factor refresh completed with errors"
    return "No factor series refreshed"
