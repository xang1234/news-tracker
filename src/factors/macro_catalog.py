"""Curated macro factor series for stock and theme interpretation."""

from __future__ import annotations

from src.factors.schemas import FactorSeries

CATALOG_VERSION = "2026-05-30"


def _series(
    *,
    factor_id: str,
    provider: str,
    external_id: str,
    name: str,
    description: str,
    units: str,
    cadence: str,
    release_lag_days: int,
    relevance_tags: list[str],
    required_credentials: list[str] | None = None,
    source_url: str | None = None,
    metadata: dict[str, object] | None = None,
) -> FactorSeries:
    return FactorSeries(
        factor_id=factor_id,
        provider=provider,
        external_id=external_id,
        name=name,
        description=description,
        units=units,
        cadence=cadence,
        release_lag_days=release_lag_days,
        relevance_tags=relevance_tags,
        required_credentials=required_credentials or [],
        source_url=source_url,
        metadata={"catalog_version": CATALOG_VERSION, **(metadata or {})},
    )


_CURATED_MACRO_FACTOR_SERIES = (
    _series(
        factor_id="fred:DGS10",
        provider="fred",
        external_id="DGS10",
        name="10-Year Treasury Constant Maturity Rate",
        description="Daily 10-year Treasury rate used as a discount-rate backdrop.",
        units="percent",
        cadence="daily",
        release_lag_days=1,
        relevance_tags=["rates", "yield_curve", "macro", "stocks"],
        required_credentials=["FRED_API_KEY"],
        source_url="https://fred.stlouisfed.org/series/DGS10",
    ),
    _series(
        factor_id="fred:T10Y2Y",
        provider="fred",
        external_id="T10Y2Y",
        name="10-Year Minus 2-Year Treasury Spread",
        description="Yield-curve slope for recession and duration-sensitive equity context.",
        units="percent",
        cadence="daily",
        release_lag_days=1,
        relevance_tags=["rates", "yield_curve", "macro", "stocks"],
        required_credentials=["FRED_API_KEY"],
        source_url="https://fred.stlouisfed.org/series/T10Y2Y",
    ),
    _series(
        factor_id="fred:CPIAUCSL",
        provider="fred",
        external_id="CPIAUCSL",
        name="Consumer Price Index for All Urban Consumers",
        description="Monthly CPI level for inflation-regime interpretation.",
        units="index",
        cadence="monthly",
        release_lag_days=15,
        relevance_tags=["inflation", "macro", "consumer", "stocks"],
        required_credentials=["FRED_API_KEY"],
        source_url="https://fred.stlouisfed.org/series/CPIAUCSL",
    ),
    _series(
        factor_id="fred:INDPRO",
        provider="fred",
        external_id="INDPRO",
        name="Industrial Production Index",
        description="Monthly production backdrop for cyclicals and semiconductor demand.",
        units="index",
        cadence="monthly",
        release_lag_days=17,
        relevance_tags=["industry", "macro", "demand", "stocks"],
        required_credentials=["FRED_API_KEY"],
        source_url="https://fred.stlouisfed.org/series/INDPRO",
    ),
    _series(
        factor_id="bls:CES0000000001",
        provider="bls",
        external_id="CES0000000001",
        name="All Employees, Total Nonfarm",
        description="Payroll employment level from BLS establishment survey.",
        units="thousands",
        cadence="monthly",
        release_lag_days=7,
        relevance_tags=["labor", "macro", "demand", "stocks"],
        source_url="https://data.bls.gov/timeseries/CES0000000001",
    ),
    _series(
        factor_id="bls:CUSR0000SA0",
        provider="bls",
        external_id="CUSR0000SA0",
        name="Consumer Price Index for All Urban Consumers",
        description="BLS CPI level as the primary inflation source.",
        units="index",
        cadence="monthly",
        release_lag_days=15,
        relevance_tags=["inflation", "macro", "consumer", "stocks"],
        source_url="https://data.bls.gov/timeseries/CUSR0000SA0",
    ),
    _series(
        factor_id="bea:NIPA:T10101:A191RL:Q",
        provider="bea",
        external_id="NIPA:T10101:A191RL:Q",
        name="Real Gross Domestic Product",
        description="Quarterly real GDP growth for broad macro regime context.",
        units="percent",
        cadence="quarterly",
        release_lag_days=30,
        relevance_tags=["growth", "macro", "stocks"],
        required_credentials=["BEA_API_KEY"],
        source_url="https://apps.bea.gov/iTable/?ReqID=19&step=2",
        metadata={
            "dataset": "NIPA",
            "table_name": "T10101",
            "line_number": "1",
            "frequency": "Q",
        },
    ),
    _series(
        factor_id="bea:NIPA:T61900A:A055RC:Q",
        provider="bea",
        external_id="NIPA:T61900A:A055RC:Q",
        name="Corporate Profits After Tax",
        description=(
            "Corporate profits after tax without IVA and CCAdj for margin "
            "and earnings-cycle interpretation."
        ),
        units="billions_usd",
        cadence="quarterly",
        release_lag_days=60,
        relevance_tags=["profits", "macro", "earnings", "stocks"],
        required_credentials=["BEA_API_KEY"],
        source_url="https://apps.bea.gov/iTable/?ReqID=19&step=2",
        metadata={
            "dataset": "NIPA",
            "table_name": "T61900A",
            "line_number": "1",
            "frequency": "Q",
        },
    ),
    _series(
        factor_id="treasury:avg_interest_rates:notes",
        provider="treasury",
        external_id="avg_interest_rates:notes",
        name="Average Interest Rate on Treasury Notes",
        description="No-key Treasury Fiscal Data series for government borrowing costs.",
        units="percent",
        cadence="monthly",
        release_lag_days=1,
        relevance_tags=["rates", "treasury", "macro", "stocks"],
        source_url=(
            "https://fiscaldata.treasury.gov/datasets/average-interest-rates/"
            "average-interest-rates-on-u-s-treasury-securities"
        ),
        metadata={
            "endpoint": (
                "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
                "v2/accounting/od/avg_interest_rates"
            ),
            "date_field": "record_date",
            "value_field": "avg_interest_rate_amt",
            "filter": "security_type_desc:eq:Treasury Notes",
        },
    ),
    _series(
        factor_id="fed:h15:RIFLGFCY10_N.B",
        provider="fed",
        external_id="RIFLGFCY10_N.B",
        name="Federal Reserve H.15 10-Year Treasury Yield",
        description="No-key Federal Reserve DDP CSV series for H.15 daily rates.",
        units="percent",
        cadence="daily",
        release_lag_days=1,
        relevance_tags=["rates", "fed", "yield_curve", "macro", "stocks"],
        source_url="https://www.federalreserve.gov/releases/h15/data.htm",
        metadata={
            "rel": "H15",
            "series": "bf17364827e38702b42a58cf8eaa3f78",
            "value_column": "RIFLGFCY10_N.B",
        },
    ),
)


def get_curated_macro_factor_series() -> list[FactorSeries]:
    """Return versioned macro factor registry entries."""
    return list(_CURATED_MACRO_FACTOR_SERIES)
