"""Macro factor datasource providers."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from typing import Any, Protocol

import httpx

from src.factors.schemas import FactorObservation, FactorSeries

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
BLS_TIMESERIES_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
BEA_DATA_URL = "https://apps.bea.gov/api/data/"
FED_DDP_URL = "https://www.federalreserve.gov/datadownload/Output.aspx"
TREASURY_AVG_INTEREST_RATES_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
    "v2/accounting/od/avg_interest_rates"
)


class FactorHTTPClient(Protocol):
    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Fetch a URL with optional query parameters."""
        ...

    async def post(
        self,
        url: str,
        json_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Post a JSON body to a URL."""
        ...


@dataclass(frozen=True)
class MacroProviderCredentials:
    """Optional free API credentials for macro data providers."""

    fred_api_key: str | None = None
    bls_registration_key: str | None = None
    bea_api_key: str | None = None

    @classmethod
    def from_env(cls) -> MacroProviderCredentials:
        """Load provider keys from conventional environment variables."""
        return cls(
            fred_api_key=os.getenv("FRED_API_KEY") or None,
            bls_registration_key=os.getenv("BLS_REGISTRATION_KEY") or None,
            bea_api_key=os.getenv("BEA_API_KEY") or None,
        )


class MacroProviderError(Exception):
    """Base exception for macro provider failures."""


class MissingProviderCredentialError(MacroProviderError):
    """Raised when a provider cannot run without a configured free key."""


class ProviderResponseError(MacroProviderError):
    """Raised when an upstream provider returns a domain-level error."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _date_to_utc(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), datetime.min.time(), tzinfo=UTC)


def _lagged_available_at(series: FactorSeries, observation_date: date) -> datetime:
    release_date = observation_date + timedelta(days=series.release_lag_days)
    return datetime.combine(release_date, datetime.min.time(), tzinfo=UTC)


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text == ".":
        return None
    return float(text)


def _observation(
    series: FactorSeries,
    *,
    observation_date: date,
    value: Any,
    available_at: datetime,
    fetched_at: datetime,
    revision: str = "",
    metadata: dict[str, Any] | None = None,
) -> FactorObservation:
    parsed_value = _parse_number(value)
    return FactorObservation(
        factor_id=series.factor_id,
        observation_date=observation_date,
        value=parsed_value,
        units=series.units,
        available_at=available_at,
        fetched_at=fetched_at,
        revision=revision,
        missing_reason="provider_missing_value" if parsed_value is None else None,
        metadata=metadata or {},
    )


def _parse_bls_period(year: str, period: str) -> date:
    if period.startswith("M") and period != "M13":
        return date(int(year), int(period[1:]), 1)
    if period.startswith("Q"):
        return date(int(year), (int(period[1:]) - 1) * 3 + 1, 1)
    return date(int(year), 1, 1)


def _parse_bea_period(value: str) -> date:
    if "Q" in value:
        year, quarter = value.split("Q", 1)
        return date(int(year), (int(quarter) - 1) * 3 + 1, 1)
    return date(int(value), 1, 1)


def _date_in_range(value: date, start: date | None, end: date | None) -> bool:
    if start is not None and value < start:
        return False
    return not (end is not None and value > end)


def _latest_only(
    observations: list[FactorObservation],
    *,
    latest: bool,
) -> list[FactorObservation]:
    if not latest or not observations:
        return observations
    return [max(observations, key=lambda observation: observation.observation_date)]


def _response_json(response: httpx.Response) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise ProviderResponseError("provider returned non-object JSON payload")
    return payload


class FredFactorProvider:
    """FRED series observation provider."""

    def __init__(
        self,
        http_client: FactorHTTPClient,
        credentials: MacroProviderCredentials,
    ) -> None:
        self.http_client = http_client
        self._credentials = credentials

    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        if not self._credentials.fred_api_key:
            raise MissingProviderCredentialError("FRED_API_KEY is required for FRED")

        params = {
            "series_id": series.external_id,
            "api_key": self._credentials.fred_api_key,
            "file_type": "json",
            "sort_order": "desc" if latest else "asc",
        }
        if latest:
            params["limit"] = "1"
        if start is not None:
            params["observation_start"] = start.isoformat()
        if end is not None:
            params["observation_end"] = end.isoformat()

        payload = _response_json(await self.http_client.get(FRED_OBSERVATIONS_URL, params=params))
        if "error_code" in payload:
            raise ProviderResponseError(str(payload.get("error_message") or payload["error_code"]))

        observed_at = fetched_at or _utc_now()
        observations: list[FactorObservation] = []
        for row in payload.get("observations", []):
            release_date = row.get("realtime_start") or row["date"]
            obs_date = date.fromisoformat(row["date"])
            observations.append(
                _observation(
                    series,
                    observation_date=obs_date,
                    value=row.get("value"),
                    available_at=_date_to_utc(release_date),
                    fetched_at=observed_at,
                    revision=release_date,
                    metadata={"realtime_end": row.get("realtime_end")},
                )
            )
        return observations


class BlsFactorProvider:
    """BLS public data API provider."""

    def __init__(
        self,
        http_client: FactorHTTPClient,
        credentials: MacroProviderCredentials,
    ) -> None:
        self.http_client = http_client
        self._credentials = credentials

    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        if latest:
            response = await self.http_client.get(
                f"{BLS_TIMESERIES_URL}/{series.external_id}",
                params={"latest": "true"},
            )
        else:
            body: dict[str, Any] = {"seriesid": [series.external_id]}
            if start is not None:
                body["startyear"] = str(start.year)
            if end is not None:
                body["endyear"] = str(end.year)
            if self._credentials.bls_registration_key:
                body["registrationkey"] = self._credentials.bls_registration_key
            response = await self.http_client.post(BLS_TIMESERIES_URL, json_body=body)

        payload = _response_json(response)
        if payload.get("status") != "REQUEST_SUCCEEDED":
            message = "; ".join(payload.get("message") or ["BLS request failed"])
            raise ProviderResponseError(message)

        observed_at = fetched_at or _utc_now()
        observations: list[FactorObservation] = []
        for provider_series in payload.get("Results", {}).get("series", []):
            for row in provider_series.get("data", []):
                if row.get("period") == "M13":
                    continue
                obs_date = _parse_bls_period(row["year"], row["period"])
                if not _date_in_range(obs_date, start, end):
                    continue
                observations.append(
                    _observation(
                        series,
                        observation_date=obs_date,
                        value=row.get("value"),
                        available_at=_lagged_available_at(series, obs_date),
                        fetched_at=observed_at,
                        revision="latest" if row.get("latest") == "true" else "",
                        metadata={
                            "period": row.get("period"),
                            "period_name": row.get("periodName"),
                            "footnotes": row.get("footnotes", []),
                        },
                    )
                )
        return observations


class BeaFactorProvider:
    """BEA API provider for NIPA macro series."""

    def __init__(
        self,
        http_client: FactorHTTPClient,
        credentials: MacroProviderCredentials,
    ) -> None:
        self.http_client = http_client
        self._credentials = credentials

    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        if not self._credentials.bea_api_key:
            raise MissingProviderCredentialError("BEA_API_KEY is required for BEA")

        params = {
            "UserID": self._credentials.bea_api_key,
            "method": "GetData",
            "DataSetName": series.metadata.get("dataset", "NIPA"),
            "TableName": series.metadata["table_name"],
            "LineNumber": str(series.metadata["line_number"]),
            "Frequency": series.metadata.get("frequency", "Q"),
            "Year": str(end.year if latest and end else "ALL"),
            "ResultFormat": "JSON",
        }

        payload = _response_json(await self.http_client.get(BEA_DATA_URL, params=params))
        if "Error" in payload.get("BEAAPI", {}):
            raise ProviderResponseError(str(payload["BEAAPI"]["Error"]))

        observed_at = fetched_at or _utc_now()
        observations: list[FactorObservation] = []
        rows = payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
        for row in rows:
            obs_date = _parse_bea_period(row["TimePeriod"])
            if not _date_in_range(obs_date, start, end):
                continue
            observations.append(
                _observation(
                    series,
                    observation_date=obs_date,
                    value=row.get("DataValue"),
                    available_at=_lagged_available_at(series, obs_date),
                    fetched_at=observed_at,
                    revision=str(row.get("TimePeriod", "")),
                    metadata={
                        "unit": row.get("CL_UNIT"),
                        "line_description": row.get("LineDescription"),
                    },
                )
            )
        return _latest_only(observations, latest=latest)


class TreasuryFiscalDataProvider:
    """No-key Treasury Fiscal Data provider."""

    def __init__(self, http_client: FactorHTTPClient) -> None:
        self.http_client = http_client

    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        date_field = str(series.metadata.get("date_field", "record_date"))
        value_field = str(series.metadata["value_field"])
        filters = [str(series.metadata["filter"])] if series.metadata.get("filter") else []
        if start is not None:
            filters.append(f"{date_field}:gte:{start.isoformat()}")
        if end is not None:
            filters.append(f"{date_field}:lte:{end.isoformat()}")

        params = {
            "fields": f"{date_field},{value_field}",
            "sort": f"-{date_field}" if latest else date_field,
            "filter": ",".join(filters),
        }
        if latest:
            params["page[size]"] = "1"

        url = str(series.metadata.get("endpoint") or TREASURY_AVG_INTEREST_RATES_URL)
        payload = _response_json(await self.http_client.get(url, params=params))
        observed_at = fetched_at or _utc_now()
        observations: list[FactorObservation] = []
        for row in payload.get("data", []):
            obs_date = date.fromisoformat(row[date_field])
            observations.append(
                _observation(
                    series,
                    observation_date=obs_date,
                    value=row.get(value_field),
                    available_at=_lagged_available_at(series, obs_date),
                    fetched_at=observed_at,
                    revision=str(row.get(date_field, "")),
                    metadata={"provider": "treasury"},
                )
            )
        return observations


class FederalReserveCsvFactorProvider:
    """No-key Federal Reserve Data Download Program CSV provider."""

    def __init__(self, http_client: FactorHTTPClient) -> None:
        self.http_client = http_client

    async def fetch_observations(
        self,
        series: FactorSeries,
        *,
        start: date | None = None,
        end: date | None = None,
        latest: bool = False,
        fetched_at: datetime | None = None,
    ) -> list[FactorObservation]:
        params = {
            "rel": series.metadata["rel"],
            "series": series.metadata["series"],
            "filetype": "csv",
            "label": "include",
            "layout": "seriescolumn",
        }
        if start is not None:
            params["from"] = start.isoformat()
        if end is not None:
            params["to"] = end.isoformat()
        if latest:
            params["lastobs"] = "1"

        response = await self.http_client.get(FED_DDP_URL, params=params)
        value_column = str(series.metadata.get("value_column") or series.external_id)
        observed_at = fetched_at or _utc_now()
        observations: list[FactorObservation] = []

        for row in _iter_csv_data_rows(response.text):
            raw_date = row.get("Time Period") or row.get("Date")
            if not raw_date or value_column not in row:
                continue
            obs_date = date.fromisoformat(raw_date)
            if not _date_in_range(obs_date, start, end):
                continue
            observations.append(
                _observation(
                    series,
                    observation_date=obs_date,
                    value=row.get(value_column),
                    available_at=_lagged_available_at(series, obs_date),
                    fetched_at=observed_at,
                    revision=raw_date,
                    metadata={"provider": "fed", "value_column": value_column},
                )
            )
        return _latest_only(observations, latest=latest)


def _iter_csv_data_rows(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("Time Period,") or line.startswith("Date,")
        ),
        None,
    )
    if header_index is None:
        raise ProviderResponseError("Federal Reserve CSV response did not include a data header")
    return list(csv.DictReader(StringIO("\n".join(lines[header_index:]))))
