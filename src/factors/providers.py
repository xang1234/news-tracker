"""Macro factor datasource providers."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from typing import Any

from src.factors.provider_common import (
    FactorHTTPClient,
    MacroProviderError,
    MissingProviderCredentialError,
    ProviderResponseError,
    date_in_range,
    latest_only,
    make_observation,
    response_json,
    utc_now,
)
from src.factors.schemas import FactorObservation, FactorSeries

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
BLS_TIMESERIES_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
BEA_DATA_URL = "https://apps.bea.gov/api/data/"
FED_DDP_URL = "https://www.federalreserve.gov/datadownload/Output.aspx"
TREASURY_AVG_INTEREST_RATES_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
    "v2/accounting/od/avg_interest_rates"
)

__all__ = [
    "BeaFactorProvider",
    "BlsFactorProvider",
    "FederalReserveCsvFactorProvider",
    "FredFactorProvider",
    "MacroProviderCredentials",
    "MacroProviderError",
    "MissingProviderCredentialError",
    "ProviderResponseError",
    "TreasuryFiscalDataProvider",
]


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


def _response_json_dict(response: Any) -> dict[str, Any]:
    payload = response_json(response)
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

        payload = _response_json_dict(
            await self.http_client.get(FRED_OBSERVATIONS_URL, params=params)
        )
        if "error_code" in payload:
            raise ProviderResponseError(str(payload.get("error_message") or payload["error_code"]))

        observed_at = fetched_at or utc_now()
        observations: list[FactorObservation] = []
        for row in payload.get("observations", []):
            realtime_start = row.get("realtime_start") or row["date"]
            obs_date = date.fromisoformat(row["date"])
            observations.append(
                make_observation(
                    series,
                    observation_date=obs_date,
                    value=row.get("value"),
                    fetched_at=observed_at,
                    revision=realtime_start,
                    metadata={
                        "realtime_start": realtime_start,
                        "realtime_end": row.get("realtime_end"),
                    },
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

        payload = _response_json_dict(response)
        if payload.get("status") != "REQUEST_SUCCEEDED":
            message = "; ".join(payload.get("message") or ["BLS request failed"])
            raise ProviderResponseError(message)

        observed_at = fetched_at or utc_now()
        observations: list[FactorObservation] = []
        for provider_series in payload.get("Results", {}).get("series", []):
            for row in provider_series.get("data", []):
                if row.get("period") == "M13":
                    continue
                obs_date = _parse_bls_period(row["year"], row["period"])
                if not date_in_range(obs_date, start, end):
                    continue
                observations.append(
                    make_observation(
                        series,
                        observation_date=obs_date,
                        value=row.get("value"),
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

        payload = _response_json_dict(await self.http_client.get(BEA_DATA_URL, params=params))
        if "Error" in payload.get("BEAAPI", {}):
            raise ProviderResponseError(str(payload["BEAAPI"]["Error"]))

        observed_at = fetched_at or utc_now()
        observations: list[FactorObservation] = []
        rows = payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
        for row in rows:
            obs_date = _parse_bea_period(row["TimePeriod"])
            if not date_in_range(obs_date, start, end):
                continue
            observations.append(
                make_observation(
                    series,
                    observation_date=obs_date,
                    value=row.get("DataValue"),
                    fetched_at=observed_at,
                    revision=str(row.get("TimePeriod", "")),
                    metadata={
                        "unit": row.get("CL_UNIT"),
                        "line_description": row.get("LineDescription"),
                    },
                )
            )
        return latest_only(observations, latest=latest)


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
        payload = _response_json_dict(await self.http_client.get(url, params=params))
        observed_at = fetched_at or utc_now()
        observations: list[FactorObservation] = []
        for row in payload.get("data", []):
            obs_date = date.fromisoformat(row[date_field])
            observations.append(
                make_observation(
                    series,
                    observation_date=obs_date,
                    value=row.get(value_field),
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
        observed_at = fetched_at or utc_now()
        observations: list[FactorObservation] = []

        for row in _iter_csv_data_rows(response.text):
            raw_date = row.get("Time Period") or row.get("Date")
            if not raw_date or value_column not in row:
                continue
            obs_date = date.fromisoformat(raw_date)
            if not date_in_range(obs_date, start, end):
                continue
            observations.append(
                make_observation(
                    series,
                    observation_date=obs_date,
                    value=row.get(value_column),
                    fetched_at=observed_at,
                    revision=raw_date,
                    metadata={"provider": "fed", "value_column": value_column},
                )
            )
        return latest_only(observations, latest=latest)


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
