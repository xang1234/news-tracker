"""Tests for EIA and Census semiconductor supply-chain providers."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from src.factors.providers import MissingProviderCredentialError, ProviderResponseError
from src.factors.schemas import FactorSeries
from src.factors.supply_chain_providers import (
    CensusTradeFactorProvider,
    EiaFactorProvider,
    SupplyChainProviderCredentials,
)


class FakeHTTPClient:
    def __init__(self, json_payload: Any) -> None:
        self.json_payload = json_payload
        self.get_calls: list[tuple[str, dict[str, Any] | None]] = []

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        **_: Any,
    ) -> httpx.Response:
        self.get_calls.append((url, params))
        return httpx.Response(200, json=self.json_payload)


class FakeRateLimiter:
    def __init__(self) -> None:
        self.acquired = 0

    async def acquire(self) -> None:
        self.acquired += 1


def _series(
    provider: str,
    external_id: str,
    *,
    release_lag_days: int = 0,
    units: str = "index",
    **metadata: Any,
) -> FactorSeries:
    return FactorSeries(
        factor_id=f"{provider}:{external_id}",
        provider=provider,
        external_id=external_id,
        name=f"{provider} {external_id}",
        units=units,
        cadence="monthly",
        release_lag_days=release_lag_days,
        relevance_tags=["semiconductors"],
        metadata=metadata,
    )


def test_supply_chain_credentials_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "eia-env")
    monkeypatch.setenv("CENSUS_API_KEY", "census-env")

    credentials = SupplyChainProviderCredentials.from_env()

    assert credentials.eia_api_key == "eia-env"
    assert credentials.census_api_key == "census-env"


@pytest.mark.asyncio
async def test_eia_provider_requires_key_and_builds_rate_limited_v2_request() -> None:
    rate_limiter = FakeRateLimiter()
    provider = EiaFactorProvider(
        FakeHTTPClient({"response": {"data": []}}),
        SupplyChainProviderCredentials(eia_api_key="eia-key"),
        rate_limiter=rate_limiter,
    )

    await provider.fetch_observations(
        _series(
            "eia",
            "electricity/retail-sales:industrial_price_us",
            route="electricity/retail-sales",
            value_field="price",
            frequency="monthly",
            facets={"stateid": ["US"], "sectorid": ["IND"]},
        ),
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
    )

    url, params = provider.http_client.get_calls[0]
    assert url == "https://api.eia.gov/v2/electricity/retail-sales/data/"
    assert params == {
        "api_key": "eia-key",
        "frequency": "monthly",
        "data[0]": "price",
        "facets[stateid][]": ["US"],
        "facets[sectorid][]": ["IND"],
        "start": "2026-04-01",
        "end": "2026-04-30",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": "0",
        "length": "5000",
    }
    assert rate_limiter.acquired == 1

    with pytest.raises(MissingProviderCredentialError, match="EIA_API_KEY"):
        await EiaFactorProvider(
            FakeHTTPClient({}),
            SupplyChainProviderCredentials(),
        ).fetch_observations(
            _series("eia", "electricity/retail-sales:industrial_price_us"),
        )


@pytest.mark.asyncio
async def test_eia_provider_parses_monthly_values_missing_values_and_revisions() -> None:
    fetched_at = datetime(2026, 6, 20, 12, tzinfo=UTC)
    provider = EiaFactorProvider(
        FakeHTTPClient(
            {
                "response": {
                    "data": [
                        {
                            "period": "2026-04",
                            "price": "8.12",
                            "price-units": "cents per kilowatthour",
                            "last_updated": "2026-06-19T10:30:00Z",
                        },
                        {
                            "period": "2026-05",
                            "price": None,
                            "last_updated": "2026-06-19T10:30:00Z",
                        },
                    ]
                }
            }
        ),
        SupplyChainProviderCredentials(eia_api_key="eia-key"),
        rate_limiter=FakeRateLimiter(),
    )

    observations = await provider.fetch_observations(
        _series(
            "eia",
            "electricity/retail-sales:industrial_price_us",
            release_lag_days=45,
            route="electricity/retail-sales",
            value_field="price",
            frequency="monthly",
        ),
        fetched_at=fetched_at,
    )

    assert observations[0].observation_date == date(2026, 4, 1)
    assert observations[0].available_at == datetime(2026, 5, 16, tzinfo=UTC)
    assert observations[0].fetched_at == fetched_at
    assert observations[0].value == 8.12
    assert observations[0].revision == "2026-06-19T10:30:00Z"
    assert observations[0].metadata["units"] == "cents per kilowatthour"
    assert observations[1].is_missing is True
    assert observations[1].missing_reason == "provider_missing_value"


@pytest.mark.asyncio
async def test_eia_provider_raises_provider_errors() -> None:
    provider = EiaFactorProvider(
        FakeHTTPClient({"error": "invalid route"}),
        SupplyChainProviderCredentials(eia_api_key="eia-key"),
        rate_limiter=FakeRateLimiter(),
    )

    with pytest.raises(ProviderResponseError, match="invalid route"):
        await provider.fetch_observations(_series("eia", "bad"))


@pytest.mark.asyncio
async def test_census_provider_requires_key_and_builds_rate_limited_trade_request() -> None:
    rate_limiter = FakeRateLimiter()
    provider = CensusTradeFactorProvider(
        FakeHTTPClient(
            [
                ["NAME", "I_COMMODITY_LABEL", "I_COMMODITY", "YEAR", "MONTH", "GEN_VAL_MO"],
            ]
        ),
        SupplyChainProviderCredentials(census_api_key="census-key"),
        rate_limiter=rate_limiter,
    )

    await provider.fetch_observations(
        _series(
            "census",
            "intltrade/imports/hsimport:854231:GEN_VAL_MO",
            endpoint="timeseries/intltrade/imports/hsimport",
            value_field="GEN_VAL_MO",
            commodity="854231",
            commodity_field="I_COMMODITY",
            geography="world:*",
        ),
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
    )

    url, params = provider.http_client.get_calls[0]
    assert url == "https://api.census.gov/data/timeseries/intltrade/imports/hsimport"
    assert params == {
        "get": "NAME,I_COMMODITY_LABEL,I_COMMODITY,YEAR,MONTH,GEN_VAL_MO",
        "for": "world:*",
        "I_COMMODITY": "854231",
        "YEAR": "2026",
        "MONTH": "04",
        "key": "census-key",
    }
    assert rate_limiter.acquired == 1

    with pytest.raises(MissingProviderCredentialError, match="CENSUS_API_KEY"):
        await CensusTradeFactorProvider(
            FakeHTTPClient([]),
            SupplyChainProviderCredentials(),
        ).fetch_observations(_series("census", "intltrade/imports/hsimport:854231"))


@pytest.mark.asyncio
async def test_census_provider_parses_monthly_hs_rows() -> None:
    provider = CensusTradeFactorProvider(
        FakeHTTPClient(
            [
                ["NAME", "I_COMMODITY_LABEL", "I_COMMODITY", "YEAR", "MONTH", "GEN_VAL_MO"],
                [
                    "WORLD TOTAL",
                    "Processors and controllers",
                    "854231",
                    "2026",
                    "04",
                    "1250000000",
                ],
            ]
        ),
        SupplyChainProviderCredentials(census_api_key="census-key"),
        rate_limiter=FakeRateLimiter(),
    )

    observations = await provider.fetch_observations(
        _series(
            "census",
            "intltrade/imports/hsimport:854231:GEN_VAL_MO",
            release_lag_days=35,
            units="usd",
            endpoint="timeseries/intltrade/imports/hsimport",
            value_field="GEN_VAL_MO",
            commodity="854231",
            commodity_field="I_COMMODITY",
            geography="world:*",
        ),
        fetched_at=datetime(2026, 6, 15, 12, tzinfo=UTC),
    )

    assert observations[0].observation_date == date(2026, 4, 1)
    assert observations[0].available_at == datetime(2026, 5, 6, tzinfo=UTC)
    assert observations[0].value == 1_250_000_000
    assert observations[0].revision == "2026-04"
    assert observations[0].metadata["commodity_label"] == "Processors and controllers"


@pytest.mark.asyncio
async def test_census_provider_raises_api_key_errors() -> None:
    provider = CensusTradeFactorProvider(
        FakeHTTPClient({"error": "invalid key"}),
        SupplyChainProviderCredentials(census_api_key="bad-key"),
        rate_limiter=FakeRateLimiter(),
    )

    with pytest.raises(ProviderResponseError, match="invalid key"):
        await provider.fetch_observations(
            _series(
                "census",
                "intltrade/imports/hsimport:854231:GEN_VAL_MO",
                endpoint="timeseries/intltrade/imports/hsimport",
                value_field="GEN_VAL_MO",
                commodity="854231",
            )
        )
