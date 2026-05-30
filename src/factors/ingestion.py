"""Refresh factor provider observations into factor storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from src.factors.repository import FactorRepository
from src.factors.schemas import (
    FactorObservation,
    FactorSeries,
    validate_observation_for_series,
)


class FactorObservationProvider(Protocol):
    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        """Fetch observations for a registered factor series."""
        ...


class FactorStore(Protocol):
    async def upsert_series(self, series: FactorSeries) -> FactorSeries:
        """Insert or update a factor registry entry."""
        ...

    async def upsert_observation(self, observation: FactorObservation) -> FactorObservation:
        """Insert or update a point-in-time observation."""
        ...


@dataclass(frozen=True)
class FactorIngestionResult:
    """Summary for one factor refresh."""

    factor_id: str
    observations_seen: int
    observations_written: int
    missing_observations: int


class FactorIngestionService:
    """Coordinates provider fetches with repository upserts."""

    def __init__(self, repository: FactorStore | FactorRepository) -> None:
        self._repository = repository

    async def register_series(self, series: FactorSeries) -> FactorSeries:
        """Persist a factor registry entry without fetching observations."""
        return await self._repository.upsert_series(series)

    async def refresh_series(
        self,
        provider: FactorObservationProvider,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> FactorIngestionResult:
        """Fetch and persist one factor series refresh."""
        observations = await self._fetch_validated_observations(
            provider,
            series,
            start=start,
            end=end,
            latest=latest,
            fetched_at=fetched_at,
        )

        await self._repository.upsert_series(series)
        return await self._write_observations(series, observations)

    async def refresh_registered_series(
        self,
        provider: FactorObservationProvider,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> FactorIngestionResult:
        """Fetch and persist observations for a series already in the registry."""
        observations = await self._fetch_validated_observations(
            provider,
            series,
            start=start,
            end=end,
            latest=latest,
            fetched_at=fetched_at,
        )
        return await self._write_observations(series, observations)

    async def _fetch_validated_observations(
        self,
        provider: FactorObservationProvider,
        series: FactorSeries,
        *,
        start: date | None,
        end: date | None,
        latest: bool,
        fetched_at: datetime | None,
    ) -> list[FactorObservation]:
        observations = await provider.fetch_observations(
            series,
            start=start,
            end=end,
            latest=latest,
            fetched_at=fetched_at,
        )
        for observation in observations:
            validate_observation_for_series(series, observation)
        return observations

    async def _write_observations(
        self,
        series: FactorSeries,
        observations: list[FactorObservation],
    ) -> FactorIngestionResult:
        written = 0
        missing = 0
        for observation in observations:
            await self._repository.upsert_observation(observation)
            written += 1
            if observation.is_missing:
                missing += 1

        return FactorIngestionResult(
            factor_id=series.factor_id,
            observations_seen=len(observations),
            observations_written=written,
            missing_observations=missing,
        )
