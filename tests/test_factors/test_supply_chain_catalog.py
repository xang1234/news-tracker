"""Tests for curated semiconductor supply-chain factor entries."""

from src.factors.supply_chain_catalog import (
    SUPPLY_CHAIN_CATALOG_VERSION,
    get_curated_supply_chain_factor_series,
)


def test_supply_chain_catalog_covers_eia_and_census() -> None:
    series = get_curated_supply_chain_factor_series()

    providers = {entry.provider for entry in series}

    assert providers == {"eia", "census"}


def test_supply_chain_catalog_has_semiconductor_rationale_and_tags() -> None:
    series = get_curated_supply_chain_factor_series()

    assert all(
        entry.metadata["catalog_version"] == SUPPLY_CHAIN_CATALOG_VERSION for entry in series
    )
    assert all(entry.metadata["rationale"] for entry in series)
    assert any("energy" in entry.relevance_tags for entry in series)
    assert any("ai_infrastructure" in entry.relevance_tags for entry in series)
    assert any("semiconductors" in entry.relevance_tags for entry in series)
    assert any("trade" in entry.relevance_tags for entry in series)


def test_supply_chain_catalog_declares_api_key_requirements() -> None:
    by_provider = {
        entry.provider: entry.required_credentials
        for entry in get_curated_supply_chain_factor_series()
    }

    assert by_provider["eia"] == ["EIA_API_KEY"]
    assert by_provider["census"] == ["CENSUS_API_KEY"]


def test_census_entries_identify_hs_or_end_use_codes() -> None:
    census_entries = [
        entry for entry in get_curated_supply_chain_factor_series() if entry.provider == "census"
    ]

    assert census_entries
    assert all(
        entry.metadata.get("commodity") or entry.metadata.get("end_use") for entry in census_entries
    )
