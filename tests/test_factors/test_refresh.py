"""Tests for curated factor refresh orchestration."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, cast

import pytest

from src.factors import refresh
from src.factors.providers import MissingProviderCredentialError
from src.factors.schemas import FactorObservation, FactorSeries
from src.storage.database import Database


class MissingCredentialProvider:
    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        raise MissingProviderCredentialError("FRED_API_KEY is required for FRED")


class FakeHTTPClient:
    async def __aenter__(self) -> FakeHTTPClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class InMemoryFactorRepository:
    def __init__(self, database: Any) -> None:
        self.database = database
        self.series: dict[str, FactorSeries] = {}
        self.observations: list[FactorObservation] = []

    async def upsert_series(self, series: FactorSeries) -> FactorSeries:
        self.series[series.factor_id] = series
        return series

    async def upsert_observation(self, observation: FactorObservation) -> FactorObservation:
        self.observations.append(observation)
        return observation

    async def upsert_observations(
        self,
        observations: list[FactorObservation],
    ) -> list[FactorObservation]:
        persisted = []
        for observation in observations:
            persisted.append(await self.upsert_observation(observation))
        return persisted

    async def upsert_series_with_observations(
        self,
        series: FactorSeries,
        observations: list[FactorObservation],
    ) -> tuple[FactorSeries, list[FactorObservation]]:
        persisted_series = await self.upsert_series(series)
        persisted_observations = await self.upsert_observations(observations)
        return persisted_series, persisted_observations


def _series() -> FactorSeries:
    return FactorSeries(
        factor_id="fred:DGS10",
        provider="fred",
        external_id="DGS10",
        name="10-Year Treasury Constant Maturity Rate",
        units="percent",
        cadence="daily",
        relevance_tags=["rates"],
        required_credentials=["FRED_API_KEY"],
    )


@pytest.mark.asyncio
async def test_refresh_curated_registers_catalog_entry_when_credentials_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = _series()
    repository = InMemoryFactorRepository(object())
    monkeypatch.setattr(refresh, "curated_factor_series", lambda: [series])
    monkeypatch.setattr(refresh, "FactorRepository", lambda database: repository)
    monkeypatch.setattr(refresh, "HTTPClient", FakeHTTPClient)
    monkeypatch.setattr(
        refresh,
        "_build_provider_map",
        lambda http_client: {"fred": MissingCredentialProvider()},
    )

    summary = await refresh.refresh_curated_factor_series(
        cast(Database, object()),
        providers={"fred"},
    )

    assert summary.skipped_missing_credentials == ["fred:DGS10"]
    assert repository.series == {"fred:DGS10": series}
    assert repository.observations == []


@pytest.mark.asyncio
async def test_refresh_curated_rejects_unknown_provider_selector() -> None:
    with pytest.raises(refresh.UnknownFactorSelectorError, match="Unknown factor provider"):
        await refresh.refresh_curated_factor_series(cast(Database, object()), providers={"frd"})


@pytest.mark.asyncio
async def test_refresh_curated_rejects_unknown_factor_id_selector() -> None:
    with pytest.raises(refresh.UnknownFactorSelectorError, match="Unknown factor_id"):
        await refresh.refresh_curated_factor_series(
            cast(Database, object()),
            factor_ids={"fred:NOPE"},
        )
