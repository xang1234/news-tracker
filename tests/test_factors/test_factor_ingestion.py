"""Tests for refreshing factor providers into the repository."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.factors.ingestion import FactorIngestionService
from src.factors.providers import MissingProviderCredentialError
from src.factors.schemas import FactorObservation, FactorSeries


class FakeProvider:
    def __init__(self, observations: list[FactorObservation]) -> None:
        self.observations = observations
        self.calls: list[dict[str, object]] = []

    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        self.calls.append(
            {
                "series": series,
                "start": start,
                "end": end,
                "latest": latest,
                "fetched_at": fetched_at,
            }
        )
        return self.observations


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


class InMemoryFactorRepository:
    def __init__(self) -> None:
        self.series: dict[str, FactorSeries] = {}
        self.observations: dict[tuple[str, date, datetime, str], FactorObservation] = {}

    async def upsert_series(self, series: FactorSeries) -> FactorSeries:
        self.series[series.factor_id] = series
        return series

    async def upsert_observation(self, observation: FactorObservation) -> FactorObservation:
        key = (
            observation.factor_id,
            observation.observation_date,
            observation.available_at,
            observation.revision,
        )
        self.observations[key] = observation
        return observation

    async def upsert_observations(
        self,
        observations: list[FactorObservation],
    ) -> list[FactorObservation]:
        return [await self.upsert_observation(observation) for observation in observations]

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
        relevance_tags=["rates", "macro"],
    )


@pytest.mark.asyncio
async def test_refresh_series_upserts_registry_and_observations_idempotently() -> None:
    series = _series()
    fetched_at = datetime(2026, 5, 1, 14, tzinfo=UTC)
    observation = FactorObservation(
        factor_id="fred:DGS10",
        observation_date=date(2026, 4, 30),
        value=4.52,
        units="percent",
        available_at=datetime(2026, 5, 1, tzinfo=UTC),
        fetched_at=fetched_at,
        revision="2026-05-01",
    )
    repo = InMemoryFactorRepository()
    provider = FakeProvider([observation])
    service = FactorIngestionService(repo)

    first = await service.refresh_series(
        provider,
        series,
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
        fetched_at=fetched_at,
    )
    second = await service.refresh_series(
        provider,
        series,
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
        fetched_at=fetched_at,
    )

    assert first.observations_seen == 1
    assert first.observations_written == 1
    assert second.observations_written == 1
    assert len(repo.series) == 1
    assert len(repo.observations) == 1
    assert provider.calls[0]["fetched_at"] == fetched_at


@pytest.mark.asyncio
async def test_refresh_latest_passes_latest_flag_to_provider() -> None:
    series = _series()
    repo = InMemoryFactorRepository()
    provider = FakeProvider([])
    service = FactorIngestionService(repo)

    result = await service.refresh_series(provider, series, latest=True)

    assert result.observations_seen == 0
    assert provider.calls[0]["latest"] is True


@pytest.mark.asyncio
async def test_refresh_rejects_observation_that_does_not_match_registry() -> None:
    repo = InMemoryFactorRepository()
    provider = FakeProvider(
        [
            FactorObservation(
                factor_id="fred:DGS10",
                observation_date=date(2026, 4, 30),
                value=4.52,
                units="basis_points",
                available_at=datetime(2026, 5, 1, tzinfo=UTC),
            )
        ]
    )
    service = FactorIngestionService(repo)

    with pytest.raises(ValueError, match="units mismatch"):
        await service.refresh_series(provider, _series())

    assert repo.series == {}
    assert repo.observations == {}


@pytest.mark.asyncio
async def test_refresh_series_does_not_register_series_when_provider_fetch_fails() -> None:
    series = _series()
    repo = InMemoryFactorRepository()
    service = FactorIngestionService(repo)

    with pytest.raises(MissingProviderCredentialError):
        await service.refresh_series(MissingCredentialProvider(), series)

    assert repo.series == {}
    assert repo.observations == {}
