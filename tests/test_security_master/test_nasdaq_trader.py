"""Tests for Nasdaq Trader symbol-directory parsing and reconciliation."""

from datetime import UTC, datetime

import pytest

from src.security_master.nasdaq_trader import (
    NASDAQ_TRADER_EXTERNAL_KEY,
    build_nasdaq_trader_reconciliation,
    parse_nasdaq_trader_symbol_directories,
)
from src.security_master.schemas import Security

NASDAQ_HEADER = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
    "Round Lot Size|ETF|NextShares"
)
OTHER_HEADER = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
    "Test Issue|NASDAQ Symbol"
)
NASDAQ_LISTED_TEXT = f"""{NASDAQ_HEADER}
NVDA|NVIDIA Corporation|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
TST|Nasdaq Test Issue|S|Y|N|100|N|N
File Creation Time: 0601202616:01|||||||
"""

OTHER_LISTED_TEXT = f"""{OTHER_HEADER}
IBM|International Business Machines Corporation|N|IBM|N|100|N|IBM
ARKK|ARK Innovation ETF|P|ARKK|Y|100|N|ARKK
ZVZZT|NYSE Test Issue|N|ZVZZT|N|100|Y|ZVZZT
File Creation Time: 0601202616:02|||||||
"""


def test_parse_symbol_directories_preserves_listing_metadata() -> None:
    directory = parse_nasdaq_trader_symbol_directories(
        NASDAQ_LISTED_TEXT,
        OTHER_LISTED_TEXT,
    )

    by_symbol = {record.symbol: record for record in directory.records}

    assert directory.nasdaq_listed.file_creation_time == datetime(
        2026,
        6,
        1,
        16,
        1,
        tzinfo=UTC,
    )
    assert by_symbol["NVDA"].listing_exchange == "Nasdaq"
    assert by_symbol["NVDA"].market_category_name == "Nasdaq Global Select Market"
    assert by_symbol["QQQ"].is_etf is True
    assert by_symbol["TST"].is_test_issue is True
    assert by_symbol["IBM"].listing_exchange == "New York Stock Exchange"
    assert by_symbol["ARKK"].listing_exchange_code == "P"
    assert by_symbol["ARKK"].round_lot_size == 100


def test_parser_accepts_extra_columns_without_losing_shape_metadata() -> None:
    nasdaq_text = f"""{NASDAQ_HEADER}|Odd Lot Eligible
NVDA|NVIDIA Corporation|Q|N|N|100|N|N|Y
File Creation Time: 0601202616:01||||||||
"""
    directory = parse_nasdaq_trader_symbol_directories(nasdaq_text, OTHER_LISTED_TEXT)

    record = next(record for record in directory.records if record.symbol == "NVDA")

    assert record.extra_fields == {"Odd Lot Eligible": "Y"}
    assert "Odd Lot Eligible" in directory.nasdaq_listed.headers


def test_parser_rejects_missing_required_columns() -> None:
    bad_text = """Symbol|Security Name|Test Issue
NVDA|NVIDIA Corporation|N
"""

    with pytest.raises(ValueError, match="Market Category"):
        parse_nasdaq_trader_symbol_directories(bad_text, OTHER_LISTED_TEXT)


def test_reconciliation_preserves_existing_sec_identifiers_and_marks_flags() -> None:
    directory = parse_nasdaq_trader_symbol_directories(
        NASDAQ_LISTED_TEXT,
        OTHER_LISTED_TEXT,
    )
    existing = Security(
        ticker="NVDA",
        exchange="US",
        name="NVIDIA Corp",
        aliases=["nvidia"],
        sec_cik="1045810",
        issuer_name="NVIDIA Corporation",
        external_identifiers={"sec_ticker": "NVDA"},
        identifier_lineage=[
            {
                "identifier_type": "sec_cik",
                "value": "1045810",
                "source": "sec_company_tickers",
            },
        ],
    )

    result = build_nasdaq_trader_reconciliation(
        directory,
        existing_by_key={("NVDA", "US"): existing},
        previously_sourced_by_key={},
        observed_at=datetime(2026, 6, 1, 17, tzinfo=UTC),
    )
    securities = {security.ticker: security for security in result.securities}

    nvda = securities["NVDA"]
    assert nvda.sec_cik == "0001045810"
    assert nvda.external_identifiers["sec_ticker"] == "NVDA"
    assert nvda.name == "NVIDIA Corporation"
    assert "NVIDIA Corp" in nvda.aliases
    assert nvda.external_identifiers[NASDAQ_TRADER_EXTERNAL_KEY]["market_category"] == "Q"
    assert nvda.external_identifiers[NASDAQ_TRADER_EXTERNAL_KEY]["is_test_issue"] is False
    assert nvda.is_active is True

    qqq = securities["QQQ"]
    assert qqq.external_identifiers[NASDAQ_TRADER_EXTERNAL_KEY]["is_etf"] is True
    assert qqq.is_active is True

    test_issue = securities["TST"]
    assert test_issue.is_active is False
    assert test_issue.external_identifiers[NASDAQ_TRADER_EXTERNAL_KEY]["status"] == "test_issue"

    arkk = securities["ARKK"]
    assert arkk.external_identifiers[NASDAQ_TRADER_EXTERNAL_KEY]["listing_exchange"] == (
        "NYSE ARCA"
    )


def test_reconciliation_deactivates_previously_sourced_symbols_missing_from_latest() -> None:
    directory = parse_nasdaq_trader_symbol_directories(
        f"""{NASDAQ_HEADER}
META|Meta Platforms Inc|Q|N|N|100|N|N
File Creation Time: 0601202616:01|||||||
""",
        OTHER_LISTED_TEXT,
    )
    previously_sourced = {
        ("FB", "US"): Security(
            ticker="FB",
            exchange="US",
            name="Facebook Inc",
            is_active=True,
            external_identifiers={
                NASDAQ_TRADER_EXTERNAL_KEY: {
                    "status": "active",
                    "last_seen_at": "2026-05-31T17:00:00+00:00",
                },
            },
        ),
    }

    result = build_nasdaq_trader_reconciliation(
        directory,
        existing_by_key={},
        previously_sourced_by_key=previously_sourced,
        observed_at=datetime(2026, 6, 1, 17, tzinfo=UTC),
    )
    securities = {security.ticker: security for security in result.securities}

    assert securities["META"].is_active is True
    assert securities["FB"].is_active is False
    assert securities["FB"].external_identifiers[NASDAQ_TRADER_EXTERNAL_KEY]["status"] == (
        "missing_from_latest"
    )
    assert result.deactivated_missing_count == 1
