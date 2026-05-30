"""Operational refresh helpers for curated factor datasources."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from src.factors.ingestion import FactorIngestionService, FactorObservationProvider
from src.factors.macro_catalog import get_curated_macro_factor_series
from src.factors.provider_common import (
    MissingProviderCredentialError,
)
from src.factors.providers import (
    BeaFactorProvider,
    BlsFactorProvider,
    FederalReserveCsvFactorProvider,
    FredFactorProvider,
    MacroProviderCredentials,
    TreasuryFiscalDataProvider,
)
from src.factors.repository import FactorRepository
from src.factors.schemas import FactorSeries
from src.factors.supply_chain_catalog import get_curated_supply_chain_factor_series
from src.factors.supply_chain_providers import (
    CensusTradeFactorProvider,
    EiaFactorProvider,
    SupplyChainProviderCredentials,
)
from src.ingestion.http_client import HTTPClient
from src.storage.database import Database


@dataclass(frozen=True)
class FactorRefreshSummary:
    """Aggregate result for a curated factor refresh run."""

    series_seen: int = 0
    series_refreshed: int = 0
    observations_seen: int = 0
    observations_written: int = 0
    skipped_missing_credentials: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False


class UnknownFactorSelectorError(ValueError):
    """Raised when a refresh selector does not match the curated catalog."""


def curated_factor_series() -> list[FactorSeries]:
    """Return all curated macro and supply-chain factor series."""
    return [
        *get_curated_macro_factor_series(),
        *get_curated_supply_chain_factor_series(),
    ]


def validate_factor_refresh_selectors(
    *,
    providers: set[str] | None = None,
    factor_ids: set[str] | None = None,
    series: Iterable[FactorSeries] | None = None,
) -> None:
    """Validate requested refresh selectors against the curated catalog."""
    selected_providers = {provider.lower() for provider in providers or set()}
    selected_factor_ids = set(factor_ids or set())
    source = list(series) if series is not None else curated_factor_series()

    known_providers = {item.provider.lower() for item in source}
    known_factor_ids = {item.factor_id for item in source}
    unknown_providers = sorted(selected_providers - known_providers)
    unknown_factor_ids = sorted(selected_factor_ids - known_factor_ids)

    messages: list[str] = []
    if unknown_providers:
        label = "provider" if len(unknown_providers) == 1 else "providers"
        messages.append(f"Unknown factor {label}: {', '.join(unknown_providers)}")
    if unknown_factor_ids:
        label = "factor_id" if len(unknown_factor_ids) == 1 else "factor_ids"
        messages.append(f"Unknown {label}: {', '.join(unknown_factor_ids)}")
    if messages:
        raise UnknownFactorSelectorError("; ".join(messages))


async def refresh_curated_factor_series(
    database: Database,
    *,
    providers: set[str] | None = None,
    factor_ids: set[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    latest: bool = True,
    dry_run: bool = False,
) -> FactorRefreshSummary:
    """Register and refresh curated factor observations from their providers."""
    selected_providers = {provider.lower() for provider in providers or set()}
    selected_factor_ids = set(factor_ids or set())
    curated_series = curated_factor_series()
    validate_factor_refresh_selectors(
        providers=selected_providers,
        factor_ids=selected_factor_ids,
        series=curated_series,
    )
    series_list = [
        series
        for series in curated_series
        if (not selected_providers or series.provider.lower() in selected_providers)
        and (not selected_factor_ids or series.factor_id in selected_factor_ids)
    ]
    if dry_run:
        return FactorRefreshSummary(series_seen=len(series_list), dry_run=True)

    repository = FactorRepository(database)
    ingestion = FactorIngestionService(repository)
    skipped: list[str] = []
    errors: dict[str, str] = {}
    refreshed = 0
    observations_seen = 0
    observations_written = 0

    for series in series_list:
        await ingestion.register_series(series)

    async with HTTPClient() as http_client:
        provider_map = _build_provider_map(http_client)
        for series in series_list:
            provider = provider_map.get(series.provider)
            if provider is None:
                errors[series.factor_id] = f"unsupported provider {series.provider!r}"
                continue
            try:
                result = await ingestion.refresh_registered_series(
                    provider,
                    series,
                    start=start,
                    end=end,
                    latest=latest,
                )
            except MissingProviderCredentialError:
                skipped.append(series.factor_id)
                continue
            except Exception as exc:  # noqa: BLE001 - keep one bad source from aborting all refreshes
                errors[series.factor_id] = str(exc)
                continue
            refreshed += 1
            observations_seen += result.observations_seen
            observations_written += result.observations_written

    return FactorRefreshSummary(
        series_seen=len(series_list),
        series_refreshed=refreshed,
        observations_seen=observations_seen,
        observations_written=observations_written,
        skipped_missing_credentials=skipped,
        errors=errors,
        dry_run=False,
    )


def _build_provider_map(http_client: HTTPClient) -> dict[str, FactorObservationProvider]:
    macro_credentials = MacroProviderCredentials.from_env()
    supply_credentials = SupplyChainProviderCredentials.from_env()
    return {
        "fred": FredFactorProvider(http_client, macro_credentials),
        "bls": BlsFactorProvider(http_client, macro_credentials),
        "bea": BeaFactorProvider(http_client, macro_credentials),
        "treasury": TreasuryFiscalDataProvider(http_client),
        "fed": FederalReserveCsvFactorProvider(http_client),
        "eia": EiaFactorProvider(http_client, supply_credentials),
        "census": CensusTradeFactorProvider(http_client, supply_credentials),
    }


def provider_names(series: Iterable[FactorSeries] | None = None) -> list[str]:
    """Return provider names present in the curated catalog."""
    source = list(series) if series is not None else curated_factor_series()
    return sorted({item.provider for item in source})
