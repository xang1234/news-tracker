"""EIA and Census factor providers for semiconductor supply-chain context."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

from src.factors.provider_common import (
    FactorHTTPGetClient,
    MissingProviderCredentialError,
    ProviderResponseError,
    latest_only,
    make_observation,
    response_json,
    utc_now,
)
from src.factors.schemas import FactorObservation, FactorSeries
from src.ingestion.base_adapter import RateLimiter

EIA_V2_BASE_URL = "https://api.eia.gov/v2"
CENSUS_DATA_BASE_URL = "https://api.census.gov/data"


class AsyncRateLimiter(Protocol):
    async def acquire(self) -> None:
        """Wait until this provider can issue another request."""
        ...


@dataclass(frozen=True)
class SupplyChainProviderCredentials:
    """Free API keys for EIA and Census supply-chain data."""

    eia_api_key: str | None = None
    census_api_key: str | None = None

    @classmethod
    def from_env(cls) -> SupplyChainProviderCredentials:
        """Load provider keys from conventional environment variables."""
        return cls(
            eia_api_key=os.getenv("EIA_API_KEY") or None,
            census_api_key=os.getenv("CENSUS_API_KEY") or None,
        )


def _month_start(value: str) -> date:
    if len(value) == 7:
        return date.fromisoformat(f"{value}-01")
    return date.fromisoformat(value)


def _credentials_required(value: str | None, env_name: str, provider: str) -> str:
    if not value:
        raise MissingProviderCredentialError(f"{env_name} is required for {provider}")
    return value


def _rate_limiter(
    rate_limit_per_minute: int,
    rate_limiter: AsyncRateLimiter | None,
) -> AsyncRateLimiter:
    return rate_limiter or RateLimiter(rate=rate_limit_per_minute)


class EiaFactorProvider:
    """EIA v2 factor provider for energy inputs."""

    def __init__(
        self,
        http_client: FactorHTTPGetClient,
        credentials: SupplyChainProviderCredentials,
        *,
        rate_limit_per_minute: int = 60,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self.http_client = http_client
        self._credentials = credentials
        self._rate_limiter = _rate_limiter(rate_limit_per_minute, rate_limiter)

    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        api_key = _credentials_required(
            self._credentials.eia_api_key,
            "EIA_API_KEY",
            "EIA",
        )
        await self._rate_limiter.acquire()

        value_field = str(series.metadata.get("value_field", "value"))
        route = str(series.metadata.get("route") or series.external_id.split(":", 1)[0])
        params: dict[str, Any] = {
            "api_key": api_key,
            "frequency": series.metadata.get("frequency", "monthly"),
            "data[0]": value_field,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc" if latest else "asc",
            "offset": "0",
            "length": "1" if latest else "5000",
        }
        for facet, values in dict(series.metadata.get("facets") or {}).items():
            params[f"facets[{facet}][]"] = list(values)
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()

        payload = response_json(
            await self.http_client.get(f"{EIA_V2_BASE_URL}/{route}/data/", params=params)
        )
        if isinstance(payload, dict) and "error" in payload:
            raise ProviderResponseError(str(payload["error"]))
        rows = payload.get("response", {}).get("data", []) if isinstance(payload, dict) else []

        observed_at = fetched_at or utc_now()
        observations = [
            make_observation(
                series,
                observation_date=_month_start(row["period"]),
                value=row.get(value_field),
                fetched_at=observed_at,
                revision=str(row.get("last_updated") or row.get("updated") or row["period"]),
                metadata={
                    "provider": "eia",
                    "units": row.get(f"{value_field}-units"),
                    "route": route,
                },
            )
            for row in rows
        ]
        return latest_only(observations, latest=latest)


class CensusTradeFactorProvider:
    """Census international trade provider for HS/end-use monthly data."""

    def __init__(
        self,
        http_client: FactorHTTPGetClient,
        credentials: SupplyChainProviderCredentials,
        *,
        rate_limit_per_minute: int = 60,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self.http_client = http_client
        self._credentials = credentials
        self._rate_limiter = _rate_limiter(rate_limit_per_minute, rate_limiter)

    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        api_key = _credentials_required(
            self._credentials.census_api_key,
            "CENSUS_API_KEY",
            "Census",
        )
        await self._rate_limiter.acquire()

        value_field = str(series.metadata["value_field"])
        commodity_field = str(series.metadata.get("commodity_field", "I_COMMODITY"))
        fields = ["NAME", "I_COMMODITY_LABEL", commodity_field, "YEAR", "MONTH", value_field]
        params: dict[str, Any] = {
            "get": ",".join(fields),
            "for": series.metadata.get("geography", "world:*"),
            commodity_field: series.metadata.get("commodity") or series.metadata.get("end_use"),
            "key": api_key,
        }
        _add_census_time_params(params, start=start, end=end, latest=latest)

        endpoint = str(series.metadata["endpoint"]).strip("/")
        payload = response_json(
            await self.http_client.get(f"{CENSUS_DATA_BASE_URL}/{endpoint}", params=params)
        )
        if isinstance(payload, dict) and "error" in payload:
            raise ProviderResponseError(str(payload["error"]))
        if not isinstance(payload, list):
            raise ProviderResponseError("Census provider returned an unexpected payload")

        observed_at = fetched_at or utc_now()
        observations: list[FactorObservation] = []
        for row in _census_rows(payload):
            obs_date = date(int(row["YEAR"]), int(row["MONTH"]), 1)
            if start is not None and obs_date < date(start.year, start.month, 1):
                continue
            if end is not None and obs_date > date(end.year, end.month, 1):
                continue
            observations.append(
                make_observation(
                    series,
                    observation_date=obs_date,
                    value=row.get(value_field),
                    fetched_at=observed_at,
                    revision=f"{row['YEAR']}-{row['MONTH']}",
                    metadata={
                        "provider": "census",
                        "name": row.get("NAME"),
                        "commodity": row.get(commodity_field),
                        "commodity_label": row.get("I_COMMODITY_LABEL"),
                    },
                )
            )
        return latest_only(observations, latest=latest)


def _add_census_time_params(
    params: dict[str, Any],
    *,
    start: date | None,
    end: date | None,
    latest: bool,
) -> None:
    if latest:
        params["time"] = "latest"
        return
    if start is not None and (end is None or (start.year, start.month) == (end.year, end.month)):
        params["YEAR"] = str(start.year)
        params["MONTH"] = f"{start.month:02d}"
    elif start is not None:
        params["time"] = f"from {start.year}-{start.month:02d}"


def _census_rows(payload: list[Any]) -> list[dict[str, str]]:
    if not payload:
        return []
    headers = payload[0]
    if not isinstance(headers, list):
        raise ProviderResponseError("Census payload did not include a header row")
    return [
        dict(zip(headers, row, strict=False))
        for row in payload[1:]
        if isinstance(row, list)
    ]
