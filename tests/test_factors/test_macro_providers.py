"""Tests for macro factor datasource providers."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from src.factors.provider_common import parse_number
from src.factors.providers import (
    BeaFactorProvider,
    BlsFactorProvider,
    FederalReserveCsvFactorProvider,
    FredFactorProvider,
    MacroProviderCredentials,
    MacroProviderError,
    MissingProviderCredentialError,
    ProviderResponseError,
    TreasuryFiscalDataProvider,
)
from src.factors.schemas import FactorSeries


def test_macro_provider_credentials_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "fred-env")
    monkeypatch.setenv("BLS_REGISTRATION_KEY", "bls-env")
    monkeypatch.setenv("BEA_API_KEY", "bea-env")

    credentials = MacroProviderCredentials.from_env()

    assert credentials.fred_api_key == "fred-env"
    assert credentials.bls_registration_key == "bls-env"
    assert credentials.bea_api_key == "bea-env"


def test_provider_specific_errors_keep_macro_error_base() -> None:
    assert issubclass(MissingProviderCredentialError, MacroProviderError)
    assert issubclass(ProviderResponseError, MacroProviderError)


class FakeHTTPClient:
    def __init__(self, *, json_payload: dict[str, Any] | None = None, text: str = "") -> None:
        self.json_payload = json_payload or {}
        self.text = text
        self.get_calls: list[tuple[str, dict[str, Any] | None]] = []
        self.post_calls: list[tuple[str, dict[str, Any] | None]] = []

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        **_: Any,
    ) -> httpx.Response:
        self.get_calls.append((url, params))
        if self.text:
            return httpx.Response(200, text=self.text)
        return httpx.Response(200, json=self.json_payload)

    async def post(
        self,
        url: str,
        json_body: dict[str, Any] | None = None,
        **_: Any,
    ) -> httpx.Response:
        self.post_calls.append((url, json_body))
        return httpx.Response(200, json=self.json_payload)


def _series(
    provider: str,
    external_id: str,
    *,
    units: str = "percent",
    cadence: str = "monthly",
    release_lag_days: int = 0,
    **metadata: Any,
) -> FactorSeries:
    return FactorSeries(
        factor_id=f"{provider}:{external_id}",
        provider=provider,
        external_id=external_id,
        name=f"{provider} {external_id}",
        units=units,
        cadence=cadence,
        release_lag_days=release_lag_days,
        relevance_tags=["macro"],
        metadata=metadata,
    )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("4.52", 4.52),
        ("1,234.5", 1234.5),
        ("(1,234.5)", -1234.5),
        ("4.5%", 4.5),
        ("n/a", None),
        ("footnote", None),
        ("--", None),
    ],
)
def test_parse_number_treats_provider_tokens_without_raising(
    raw_value: Any,
    expected: float | None,
) -> None:
    assert parse_number(raw_value) == expected


@pytest.mark.asyncio
async def test_fred_provider_requires_api_key_and_builds_observation_request() -> None:
    provider = FredFactorProvider(
        FakeHTTPClient(json_payload={"observations": []}),
        MacroProviderCredentials(fred_api_key="fred-key"),
    )

    await provider.fetch_observations(
        _series("fred", "DGS10"),
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
        fetched_at=datetime(2026, 5, 1, 14, tzinfo=UTC),
    )

    url, params = provider.http_client.get_calls[0]
    assert url == "https://api.stlouisfed.org/fred/series/observations"
    assert params == {
        "series_id": "DGS10",
        "api_key": "fred-key",
        "file_type": "json",
        "sort_order": "asc",
        "observation_start": "2026-04-01",
        "observation_end": "2026-04-30",
    }

    with pytest.raises(MissingProviderCredentialError, match="FRED_API_KEY"):
        await FredFactorProvider(FakeHTTPClient(), MacroProviderCredentials()).fetch_observations(
            _series("fred", "DGS10")
        )


@pytest.mark.asyncio
async def test_fred_provider_preserves_realtime_release_and_sparse_values() -> None:
    fetched_at = datetime(2026, 5, 2, 15, 30, tzinfo=UTC)
    provider = FredFactorProvider(
        FakeHTTPClient(
            json_payload={
                "observations": [
                    {
                        "realtime_start": "2026-05-01",
                        "realtime_end": "2026-05-01",
                        "date": "2026-04-30",
                        "value": "4.52",
                    },
                    {
                        "realtime_start": "2026-05-02",
                        "realtime_end": "2026-05-02",
                        "date": "2026-05-01",
                        "value": ".",
                    },
                ]
            }
        ),
        MacroProviderCredentials(fred_api_key="fred-key"),
    )

    observations = await provider.fetch_observations(
        _series("fred", "DGS10"),
        fetched_at=fetched_at,
    )

    assert observations[0].observation_date == date(2026, 4, 30)
    assert observations[0].value == 4.52
    assert observations[0].available_at == datetime(2026, 5, 1, tzinfo=UTC)
    assert observations[0].fetched_at == fetched_at
    assert observations[0].revision == "2026-05-01"
    assert observations[0].metadata["estimated_release_at"] == "2026-04-30T00:00:00+00:00"
    assert observations[0].metadata["realtime_start"] == "2026-05-01"
    assert observations[1].is_missing is True
    assert observations[1].missing_reason == "provider_missing_value"


@pytest.mark.asyncio
async def test_bls_provider_uses_no_key_mode_and_latest_refresh() -> None:
    provider = BlsFactorProvider(
        FakeHTTPClient(
            json_payload={
                "status": "REQUEST_SUCCEEDED",
                "Results": {
                    "series": [
                        {
                            "seriesID": "CUSR0000SA0",
                            "data": [
                                {
                                    "year": "2026",
                                    "period": "M04",
                                    "periodName": "April",
                                    "latest": "true",
                                    "value": "321.5",
                                    "footnotes": [{"code": "P", "text": "Preliminary."}],
                                }
                            ],
                        }
                    ]
                },
            }
        ),
        MacroProviderCredentials(),
    )

    observations = await provider.fetch_observations(
        _series("bls", "CUSR0000SA0", units="index", release_lag_days=7),
        latest=True,
        fetched_at=datetime(2026, 5, 15, 12, tzinfo=UTC),
    )

    url, params = provider.http_client.get_calls[0]
    assert url == "https://api.bls.gov/publicAPI/v2/timeseries/data/CUSR0000SA0"
    assert params == {"latest": "true"}
    assert observations[0].observation_date == date(2026, 4, 1)
    assert observations[0].available_at == datetime(2026, 5, 7, tzinfo=UTC)
    assert observations[0].metadata["estimated_release_at"] == "2026-05-07T00:00:00+00:00"
    assert observations[0].fetched_at == datetime(2026, 5, 15, 12, tzinfo=UTC)
    assert observations[0].value == 321.5
    assert observations[0].revision == "latest"
    assert observations[0].metadata["footnotes"] == [{"code": "P", "text": "Preliminary."}]


@pytest.mark.asyncio
async def test_bls_provider_posts_ranges_with_optional_registration_key() -> None:
    provider = BlsFactorProvider(
        FakeHTTPClient(
            json_payload={
                "status": "REQUEST_SUCCEEDED",
                "Results": {"series": [{"seriesID": "CES0000000001", "data": []}]},
            }
        ),
        MacroProviderCredentials(bls_registration_key="bls-key"),
    )

    await provider.fetch_observations(
        _series("bls", "CES0000000001"),
        start=date(2024, 1, 1),
        end=date(2026, 12, 31),
    )

    _, body = provider.http_client.post_calls[0]
    assert body == {
        "seriesid": ["CES0000000001"],
        "startyear": "2024",
        "endyear": "2026",
        "registrationkey": "bls-key",
    }


@pytest.mark.asyncio
async def test_bls_provider_raises_provider_errors() -> None:
    provider = BlsFactorProvider(
        FakeHTTPClient(json_payload={"status": "REQUEST_FAILED", "message": ["bad series"]}),
        MacroProviderCredentials(),
    )

    with pytest.raises(ProviderResponseError, match="bad series"):
        await provider.fetch_observations(_series("bls", "BAD"), latest=True)


@pytest.mark.asyncio
async def test_bea_provider_requires_key_and_parses_quarterly_data() -> None:
    provider = BeaFactorProvider(
        FakeHTTPClient(
            json_payload={
                "BEAAPI": {
                    "Results": {
                        "Data": [
                            {
                                "TimePeriod": "2026Q1",
                                "DataValue": "2.4",
                                "CL_UNIT": "Percent",
                                "LineDescription": "Real GDP",
                            }
                        ]
                    }
                }
            }
        ),
        MacroProviderCredentials(bea_api_key="bea-key"),
    )

    observations = await provider.fetch_observations(
        _series(
            "bea",
            "NIPA:T10101:A191RL:Q",
            cadence="quarterly",
            release_lag_days=30,
            dataset="NIPA",
            table_name="T10101",
            line_number="1",
            frequency="Q",
        ),
        start=date(2026, 1, 1),
        end=date(2026, 3, 31),
        fetched_at=datetime(2026, 4, 30, 12, tzinfo=UTC),
    )

    _, params = provider.http_client.get_calls[0]
    assert params["UserID"] == "bea-key"
    assert params["method"] == "GetData"
    assert params["DataSetName"] == "NIPA"
    assert params["TableName"] == "T10101"
    assert params["LineNumber"] == "1"
    assert observations[0].observation_date == date(2026, 1, 1)
    assert observations[0].available_at == datetime(2026, 4, 30, tzinfo=UTC)
    assert observations[0].metadata["estimated_release_at"] == "2026-04-30T00:00:00+00:00"
    assert observations[0].fetched_at == datetime(2026, 4, 30, 12, tzinfo=UTC)
    assert observations[0].value == 2.4

    with pytest.raises(MissingProviderCredentialError, match="BEA_API_KEY"):
        await BeaFactorProvider(FakeHTTPClient(), MacroProviderCredentials()).fetch_observations(
            _series("bea", "NIPA:T10101:A191RL:Q")
        )


@pytest.mark.asyncio
async def test_bea_latest_refresh_selects_latest_period_not_response_tail() -> None:
    provider = BeaFactorProvider(
        FakeHTTPClient(
            json_payload={
                "BEAAPI": {
                    "Results": {
                        "Data": [
                            {"TimePeriod": "2026Q2", "DataValue": "3.0"},
                            {"TimePeriod": "2026Q1", "DataValue": "2.0"},
                        ]
                    }
                }
            }
        ),
        MacroProviderCredentials(bea_api_key="bea-key"),
    )

    observations = await provider.fetch_observations(
        _series(
            "bea",
            "NIPA:T10101:A191RL:Q",
            cadence="quarterly",
            dataset="NIPA",
            table_name="T10101",
            line_number="1",
            frequency="Q",
        ),
        latest=True,
    )

    assert len(observations) == 1
    assert observations[0].observation_date == date(2026, 4, 1)
    assert observations[0].value == 3.0


@pytest.mark.asyncio
async def test_treasury_provider_fetches_no_key_fiscal_data() -> None:
    provider = TreasuryFiscalDataProvider(
        FakeHTTPClient(
            json_payload={
                "data": [
                    {
                        "record_date": "2026-04-30",
                        "security_type_desc": "Treasury Notes",
                        "avg_interest_rate_amt": "4.52",
                    }
                ]
            }
        )
    )

    observations = await provider.fetch_observations(
        _series(
            "treasury",
            "avg_interest_rates:notes",
            release_lag_days=1,
            value_field="avg_interest_rate_amt",
            date_field="record_date",
            filter="security_type_desc:eq:Treasury Notes",
        ),
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
    )

    _, params = provider.http_client.get_calls[0]
    assert "api_key" not in params
    assert params["filter"] == (
        "security_type_desc:eq:Treasury Notes,record_date:gte:2026-04-01,record_date:lte:2026-04-30"
    )
    assert observations[0].observation_date == date(2026, 4, 30)
    assert observations[0].metadata["estimated_release_at"] == "2026-05-01T00:00:00+00:00"
    assert observations[0].value == 4.52


@pytest.mark.asyncio
async def test_federal_reserve_csv_provider_fetches_no_key_ddp_series() -> None:
    provider = FederalReserveCsvFactorProvider(
        FakeHTTPClient(text="Header line\nTime Period,RIFLGFCY10_N.B\n2026-04-30,4.52\n")
    )

    observations = await provider.fetch_observations(
        _series(
            "fed",
            "RIFLGFCY10_N.B",
            release_lag_days=1,
            rel="H15",
            series="bf17364827e38702b42a58cf8eaa3f78",
            value_column="RIFLGFCY10_N.B",
        ),
        start=date(2026, 4, 1),
        end=date(2026, 4, 30),
    )

    _, params = provider.http_client.get_calls[0]
    assert params == {
        "rel": "H15",
        "series": "bf17364827e38702b42a58cf8eaa3f78",
        "filetype": "csv",
        "label": "include",
        "layout": "seriescolumn",
        "from": "2026-04-01",
        "to": "2026-04-30",
    }
    assert observations[0].observation_date == date(2026, 4, 30)
    assert "estimated_release_at" in observations[0].metadata
    assert observations[0].value == 4.52
