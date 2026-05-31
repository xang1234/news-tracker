"""Tests for curated macro factor registry entries."""

from src.factors.macro_catalog import (
    CATALOG_VERSION,
    get_curated_macro_factor_series,
)


def test_macro_catalog_covers_required_provider_families() -> None:
    series = get_curated_macro_factor_series()

    providers = {entry.provider for entry in series}

    assert {"fred", "bls", "bea", "treasury", "fed"} <= providers


def test_macro_catalog_tags_series_for_stock_relevance() -> None:
    series = get_curated_macro_factor_series()

    assert all(entry.relevance_tags for entry in series)
    assert all(entry.metadata["catalog_version"] == CATALOG_VERSION for entry in series)
    assert any("inflation" in entry.relevance_tags for entry in series)
    assert any("rates" in entry.relevance_tags for entry in series)
    assert any("labor" in entry.relevance_tags for entry in series)
    assert any("industry" in entry.relevance_tags for entry in series)
    assert any("profits" in entry.relevance_tags for entry in series)


def test_macro_catalog_uses_verified_bea_corporate_profit_series() -> None:
    profits = next(
        entry
        for entry in get_curated_macro_factor_series()
        if entry.name == "Corporate Profits After Tax"
    )

    assert profits.factor_id == "bea:NIPA:T61900A:A055RC:Q"
    assert profits.external_id == "NIPA:T61900A:A055RC:Q"
    assert profits.metadata["table_name"] == "T61900A"
    assert profits.metadata["line_number"] == "1"


def test_macro_catalog_declares_credentials_only_where_needed() -> None:
    by_provider = {
        entry.provider: entry.required_credentials for entry in get_curated_macro_factor_series()
    }

    assert by_provider["fred"] == ["FRED_API_KEY"]
    assert by_provider["bea"] == ["BEA_API_KEY"]
    assert by_provider["bls"] == []
    assert by_provider["treasury"] == []
    assert by_provider["fed"] == []
