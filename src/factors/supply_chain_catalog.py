"""Curated energy and trade factors for semiconductor supply-chain context."""

from __future__ import annotations

from src.factors.schemas import FactorSeries

SUPPLY_CHAIN_CATALOG_VERSION = "2026-05-30"


def _series(
    *,
    factor_id: str,
    provider: str,
    external_id: str,
    name: str,
    description: str,
    units: str,
    release_lag_days: int,
    relevance_tags: list[str],
    required_credentials: list[str],
    source_url: str,
    metadata: dict[str, object],
) -> FactorSeries:
    return FactorSeries(
        factor_id=factor_id,
        provider=provider,
        external_id=external_id,
        name=name,
        description=description,
        units=units,
        cadence="monthly",
        release_lag_days=release_lag_days,
        relevance_tags=relevance_tags,
        required_credentials=required_credentials,
        source_url=source_url,
        metadata={"catalog_version": SUPPLY_CHAIN_CATALOG_VERSION, **metadata},
    )


_CURATED_SUPPLY_CHAIN_FACTOR_SERIES = (
    _series(
        factor_id="eia:electricity:retail_sales:industrial_price_us",
        provider="eia",
        external_id="electricity/retail-sales:industrial_price_us",
        name="U.S. Industrial Electricity Price",
        description="Monthly industrial electricity price from EIA retail-sales data.",
        units="cents_per_kwh",
        release_lag_days=45,
        relevance_tags=["energy", "ai_infrastructure", "semiconductors", "macro", "stocks"],
        required_credentials=["EIA_API_KEY"],
        source_url="https://www.eia.gov/opendata/browser/electricity/retail-sales",
        metadata={
            "route": "electricity/retail-sales",
            "value_field": "price",
            "frequency": "monthly",
            "facets": {"stateid": ["US"], "sectorid": ["IND"]},
            "rationale": (
                "Industrial power prices pressure fab operating costs and data-center "
                "economics for AI infrastructure themes."
            ),
        },
    ),
    _series(
        factor_id="eia:electricity:retail_sales:industrial_sales_us",
        provider="eia",
        external_id="electricity/retail-sales:industrial_sales_us",
        name="U.S. Industrial Electricity Sales",
        description="Monthly industrial electricity sales as an energy-demand proxy.",
        units="megawatthours",
        release_lag_days=45,
        relevance_tags=["energy", "ai_infrastructure", "industry", "semiconductors", "stocks"],
        required_credentials=["EIA_API_KEY"],
        source_url="https://www.eia.gov/opendata/browser/electricity/retail-sales",
        metadata={
            "route": "electricity/retail-sales",
            "value_field": "sales",
            "frequency": "monthly",
            "facets": {"stateid": ["US"], "sectorid": ["IND"]},
            "rationale": (
                "Industrial electricity sales help separate broad power-demand regimes "
                "from issuer-specific AI infrastructure signals."
            ),
        },
    ),
    _series(
        factor_id="census:imports:hs854231:value",
        provider="census",
        external_id="intltrade/imports/hsimport:854231:GEN_VAL_MO",
        name="U.S. Imports of Processors and Controllers",
        description="Monthly import value for HS 854231 integrated circuits.",
        units="usd",
        release_lag_days=35,
        relevance_tags=["trade", "semiconductors", "supply_chain", "stocks"],
        required_credentials=["CENSUS_API_KEY"],
        source_url="https://api.census.gov/data/timeseries/intltrade/imports/hsimport.html",
        metadata={
            "endpoint": "timeseries/intltrade/imports/hsimport",
            "value_field": "GEN_VAL_MO",
            "commodity": "854231",
            "commodity_field": "I_COMMODITY",
            "geography": "world:*",
            "rationale": (
                "Processor/controller import value is a direct monthly proxy for chip "
                "flows into U.S. electronics and AI supply chains."
            ),
        },
    ),
    _series(
        factor_id="census:imports:hs854232:value",
        provider="census",
        external_id="intltrade/imports/hsimport:854232:GEN_VAL_MO",
        name="U.S. Imports of Memory Integrated Circuits",
        description="Monthly import value for HS 854232 memory integrated circuits.",
        units="usd",
        release_lag_days=35,
        relevance_tags=["trade", "semiconductors", "memory", "supply_chain", "stocks"],
        required_credentials=["CENSUS_API_KEY"],
        source_url="https://api.census.gov/data/timeseries/intltrade/imports/hsimport.html",
        metadata={
            "endpoint": "timeseries/intltrade/imports/hsimport",
            "value_field": "GEN_VAL_MO",
            "commodity": "854232",
            "commodity_field": "I_COMMODITY",
            "geography": "world:*",
            "rationale": (
                "Memory IC import value gives a free monthly signal for DRAM/NAND "
                "supply-chain cycles."
            ),
        },
    ),
    _series(
        factor_id="census:imports:hs848620:value",
        provider="census",
        external_id="intltrade/imports/hsimport:848620:GEN_VAL_MO",
        name="U.S. Imports of Semiconductor Manufacturing Equipment",
        description="Monthly import value for HS 848620 semiconductor manufacturing equipment.",
        units="usd",
        release_lag_days=35,
        relevance_tags=["trade", "semiconductors", "capex", "supply_chain", "stocks"],
        required_credentials=["CENSUS_API_KEY"],
        source_url="https://api.census.gov/data/timeseries/intltrade/imports/hsimport.html",
        metadata={
            "endpoint": "timeseries/intltrade/imports/hsimport",
            "value_field": "GEN_VAL_MO",
            "commodity": "848620",
            "commodity_field": "I_COMMODITY",
            "geography": "world:*",
            "rationale": (
                "Semiconductor-equipment import value tracks capex and domestic fab "
                "buildout pressure before it appears in company commentary."
            ),
        },
    ),
)


def get_curated_supply_chain_factor_series() -> list[FactorSeries]:
    """Return versioned energy and trade factor registry entries."""
    return list(_CURATED_SUPPLY_CHAIN_FACTOR_SERIES)
