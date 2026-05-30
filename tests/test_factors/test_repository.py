"""Tests for factor registry and observation repositories."""

import json
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock

import pytest

from src.factors.repository import FactorRepository
from src.factors.schemas import FactorObservation, FactorSeries


@pytest.fixture()
def mock_database() -> AsyncMock:
    return AsyncMock()


def _series_row() -> dict:
    return {
        "factor_id": "fred:DGS10",
        "provider": "fred",
        "external_id": "DGS10",
        "name": "10-Year Treasury Constant Maturity Rate",
        "description": "Daily 10-year Treasury rate.",
        "units": "percent",
        "cadence": "daily",
        "release_lag_days": 1,
        "relevance_tags": ["rates", "macro"],
        "required_credentials": ["FRED_API_KEY"],
        "source_url": "https://fred.stlouisfed.org/series/DGS10",
        "is_active": True,
        "metadata": {"category": "rates"},
        "created_at": datetime(2026, 5, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 5, 1, tzinfo=UTC),
    }


def _observation_row() -> dict:
    return {
        "factor_id": "fred:DGS10",
        "observation_date": date(2026, 4, 30),
        "value": 4.52,
        "units": "percent",
        "available_at": datetime(2026, 5, 1, 13, 30, tzinfo=UTC),
        "fetched_at": datetime(2026, 5, 1, 13, 45, tzinfo=UTC),
        "revision": "initial",
        "missing_reason": None,
        "metadata": {"provider_payload_id": "obs-1"},
    }


class TestFactorSeriesRepository:
    """Registry persistence behavior."""

    @pytest.mark.asyncio
    async def test_upsert_series_persists_registry_metadata(self, mock_database: AsyncMock) -> None:
        mock_database.fetchrow.return_value = _series_row()
        repo = FactorRepository(mock_database)
        series = FactorSeries(
            factor_id="fred:DGS10",
            provider="fred",
            external_id="DGS10",
            name="10-Year Treasury Constant Maturity Rate",
            description="Daily 10-year Treasury rate.",
            units="percent",
            cadence="daily",
            release_lag_days=1,
            relevance_tags=["rates", "macro"],
            required_credentials=["FRED_API_KEY"],
            source_url="https://fred.stlouisfed.org/series/DGS10",
            metadata={"category": "rates"},
        )

        result = await repo.upsert_series(series)

        args = mock_database.fetchrow.call_args[0]
        sql = args[0]
        assert "INSERT INTO factor_series" in sql
        assert "ON CONFLICT (factor_id) DO UPDATE" in sql
        assert args[1] == "fred:DGS10"
        assert args[8] == 1
        assert args[9] == ["rates", "macro"]
        assert args[10] == ["FRED_API_KEY"]
        assert json.loads(args[13]) == {"category": "rates"}
        assert result.factor_id == "fred:DGS10"

    @pytest.mark.asyncio
    async def test_get_series_parses_json_metadata(self, mock_database: AsyncMock) -> None:
        row = _series_row()
        row["metadata"] = json.dumps(row["metadata"])
        mock_database.fetchrow.return_value = row
        repo = FactorRepository(mock_database)

        result = await repo.get_series("fred:DGS10")

        assert result is not None
        assert result.metadata == {"category": "rates"}
        assert result.relevance_tags == ["rates", "macro"]

    @pytest.mark.asyncio
    async def test_list_active_series_filters_by_tag(self, mock_database: AsyncMock) -> None:
        mock_database.fetch.return_value = [_series_row()]
        repo = FactorRepository(mock_database)

        result = await repo.list_series(active_only=True, relevance_tag="rates", provider="fred")

        args = mock_database.fetch.call_args[0]
        sql = args[0]
        assert "is_active = TRUE" in sql
        assert "$1 = ANY(relevance_tags)" in sql
        assert "provider = $2" in sql
        assert args[1] == "rates"
        assert args[2] == "fred"
        assert result[0].factor_id == "fred:DGS10"


class TestFactorObservationRepository:
    """Point-in-time observation persistence behavior."""

    @pytest.mark.asyncio
    async def test_upsert_observation_persists_revision_and_availability(
        self,
        mock_database: AsyncMock,
    ) -> None:
        mock_database.fetchrow.return_value = _observation_row()
        repo = FactorRepository(mock_database)
        observation = FactorObservation(
            factor_id="fred:DGS10",
            observation_date=date(2026, 4, 30),
            value=4.52,
            units="percent",
            available_at=datetime(2026, 5, 1, 13, 30, tzinfo=UTC),
            fetched_at=datetime(2026, 5, 1, 13, 45, tzinfo=UTC),
            revision="initial",
            metadata={"provider_payload_id": "obs-1"},
        )

        result = await repo.upsert_observation(observation)

        args = mock_database.fetchrow.call_args[0]
        sql = args[0]
        assert "INSERT INTO factor_observations" in sql
        assert "ON CONFLICT (factor_id, observation_date, available_at, revision)" in sql
        assert args[1] == "fred:DGS10"
        assert args[2] == date(2026, 4, 30)
        assert args[5] == datetime(2026, 5, 1, 13, 30, tzinfo=UTC)
        assert args[7] == "initial"
        assert result.available_at == datetime(2026, 5, 1, 13, 30, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_get_observations_as_of_uses_latest_available_revision(
        self,
        mock_database: AsyncMock,
    ) -> None:
        mock_database.fetch.return_value = [_observation_row()]
        repo = FactorRepository(mock_database)
        as_of = datetime(2026, 5, 2, tzinfo=UTC)

        result = await repo.get_observations_as_of(
            "fred:DGS10",
            start=date(2026, 4, 1),
            end=date(2026, 4, 30),
            as_of=as_of,
        )

        args = mock_database.fetch.call_args[0]
        sql = args[0]
        assert "DISTINCT ON (observation_date)" in sql
        assert "available_at <= $4" in sql
        assert "ORDER BY observation_date ASC, available_at DESC, fetched_at DESC" in sql
        assert args[4] == as_of
        assert result[0].value == 4.52

    @pytest.mark.asyncio
    async def test_missing_observation_round_trips(self, mock_database: AsyncMock) -> None:
        row = _observation_row()
        row["value"] = None
        row["missing_reason"] = "provider_suppressed"
        mock_database.fetchrow.return_value = row
        repo = FactorRepository(mock_database)

        observation = await repo.upsert_observation(
            FactorObservation(
                factor_id="fred:DGS10",
                observation_date=date(2026, 4, 30),
                value=None,
                units="percent",
                available_at=datetime(2026, 5, 1, 13, 30, tzinfo=UTC),
                missing_reason="provider_suppressed",
            )
        )

        assert observation.is_missing is True
        assert observation.missing_reason == "provider_suppressed"
