"""Tests for market-structure datasource parsing and signal derivation."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from src.market_structure import (
    MarketStructureSourceFile,
    apply_market_structure_signals,
    parse_finra_short_volume_file,
    parse_sec_fails_to_deliver_file,
)
from src.security_master.schemas import Security

FINRA_TEXT = """Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
20260601|NVDA|600|5|1000|Q
20260601|ZERO|0|0|0|N
2
"""

SEC_FTD_TEXT = """SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
20260601|67066G104|NVDA|12000|NVIDIA CORP|123.45
20260602|000000000|PENNY|50|PENNY ISSUER|.
"""


def test_parse_finra_short_volume_rows_with_ratio_and_lineage() -> None:
    events = parse_finra_short_volume_file(
        MarketStructureSourceFile(
            source_name="FINRA CNMS short volume",
            source_url="https://example.test/CNMSshvol20260601.txt",
            content=FINRA_TEXT,
            fetched_at=datetime(2026, 6, 1, 23, tzinfo=UTC),
        ),
        securities_by_symbol={
            "NVDA": Security(
                ticker="NVDA",
                exchange="US",
                name="NVIDIA Corporation",
                sec_cik="1045810",
                issuer_name="NVIDIA Corporation",
            )
        },
    )

    nvda, zero = events
    assert nvda.event_type == "finra_short_volume"
    assert nvda.source_date == date(2026, 6, 1)
    assert nvda.trade_date == date(2026, 6, 1)
    assert nvda.settlement_date is None
    assert nvda.symbol == "NVDA"
    assert nvda.issuer_cik == "0001045810"
    assert nvda.issuer_name == "NVIDIA Corporation"
    assert nvda.short_volume == 600
    assert nvda.short_exempt_volume == 5
    assert nvda.total_volume == 1000
    assert nvda.short_volume_ratio == Decimal("0.6")
    assert nvda.market_code == "Q"
    assert nvda.market_name == "NASDAQ TRF Carteret"
    assert nvda.available_at == datetime(2026, 6, 1, 22, tzinfo=UTC)
    assert nvda.fetched_at == datetime(2026, 6, 1, 23, tzinfo=UTC)
    assert nvda.source_url == "https://example.test/CNMSshvol20260601.txt"
    assert nvda.metadata["source_row_number"] == 2
    assert nvda.metadata["mapping_status"] == "resolved"

    assert zero.short_volume_ratio is None
    assert zero.total_volume == 0
    assert zero.metadata["mapping_status"] == "unresolved"


def test_finra_header_and_zero_count_trailer_produce_empty_snapshot() -> None:
    events = parse_finra_short_volume_file(
        MarketStructureSourceFile(
            source_name="FINRA empty day",
            source_url="https://example.test/CNMSshvol20260602.txt",
            content="Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n0\n",
        )
    )

    assert events == []


def test_parse_sec_fails_to_deliver_rows_with_notional_and_lineage() -> None:
    events = parse_sec_fails_to_deliver_file(
        MarketStructureSourceFile(
            source_name="SEC FTD first half June 2026",
            source_url="https://www.sec.gov/files/cnsfails202606a.zip",
            content=SEC_FTD_TEXT,
            fetched_at=datetime(2026, 6, 15, tzinfo=UTC),
        ),
        securities_by_symbol={
            "NVDA": Security(
                ticker="NVDA",
                exchange="US",
                name="NVIDIA Corporation",
                sec_cik="1045810",
                issuer_name="NVIDIA Corporation",
            )
        },
    )

    nvda, penny = events
    assert nvda.event_type == "sec_fail_to_deliver"
    assert nvda.source_date == date(2026, 6, 1)
    assert nvda.trade_date is None
    assert nvda.settlement_date == date(2026, 6, 1)
    assert nvda.cusip == "67066G104"
    assert nvda.symbol == "NVDA"
    assert nvda.fail_quantity == 12000
    assert nvda.fail_price == Decimal("123.45")
    assert nvda.fail_notional == Decimal("1481400.00")
    assert nvda.issuer_cik == "0001045810"
    assert nvda.metadata["mapping_status"] == "resolved"
    assert nvda.metadata["raw_fields"]["DESCRIPTION"] == "NVIDIA CORP"

    assert penny.fail_price is None
    assert penny.fail_notional is None
    assert penny.metadata["mapping_status"] == "unresolved"


def test_signals_distinguish_ratios_persistence_and_anomaly_thresholds() -> None:
    day_1 = parse_finra_short_volume_file(
        MarketStructureSourceFile(
            source_name="FINRA",
            source_url="https://example.test/CNMSshvol20260601.txt",
            content="Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20260601|NVDA|600|0|1000|Q\n"
            "1\n",
        )
    )[0]
    day_2 = parse_finra_short_volume_file(
        MarketStructureSourceFile(
            source_name="FINRA",
            source_url="https://example.test/CNMSshvol20260602.txt",
            content="Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20260602|NVDA|700|0|1000|Q\n"
            "1\n",
        )
    )[0]
    missing_gap = parse_finra_short_volume_file(
        MarketStructureSourceFile(
            source_name="FINRA",
            source_url="https://example.test/CNMSshvol20260605.txt",
            content="Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20260605|NVDA|800|0|1000|Q\n"
            "1\n",
        )
    )[0]
    ftd = parse_sec_fails_to_deliver_file(
        MarketStructureSourceFile(
            source_name="SEC",
            source_url="https://example.test/cnsfails202606a.zip",
            content=SEC_FTD_TEXT,
        )
    )[0]

    events = apply_market_structure_signals([day_1, day_2, missing_gap, ftd])
    first, second, gap, ftd_signal = events

    assert first.signal_type == "short_volume_ratio"
    assert first.anomaly_level == "elevated"
    assert first.persistence_count == 1
    assert second.persistence_count == 2
    assert gap.persistence_count == 1
    assert gap.metadata["persistence_reset_reason"] == "missing_trading_days"
    assert ftd_signal.signal_type == "fails_to_deliver_notional"
    assert ftd_signal.anomaly_level == "watch"

    for event in events:
        payload = event.to_payload()
        assert "short_interest" not in payload
        assert "short_interest" not in event.metadata


def test_persistence_treats_weekends_as_non_trading_days() -> None:
    friday = parse_finra_short_volume_file(
        MarketStructureSourceFile(
            source_name="FINRA",
            source_url="file:///first-path.txt",
            content="Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20260605|NVDA|600|0|1000|Q\n"
            "1\n",
        )
    )[0]
    monday = parse_finra_short_volume_file(
        MarketStructureSourceFile(
            source_name="FINRA",
            source_url="file:///different-path.txt",
            content="Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20260608|NVDA|700|0|1000|Q\n"
            "1\n",
        )
    )[0]

    events = apply_market_structure_signals([friday, monday])

    assert friday.event_id != monday.event_id
    assert "first-path" not in friday.event_id
    assert events[1].persistence_count == 2
    assert "persistence_reset_reason" not in events[1].metadata


def test_malformed_records_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="line 2"):
        parse_finra_short_volume_file(
            MarketStructureSourceFile(
                source_name="FINRA",
                source_url="https://example.test/bad.txt",
                content="Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
                "20260601|NVDA|bad|0|1000|Q\n"
                "1\n",
            )
        )

    with pytest.raises(ValueError, match="line 2"):
        parse_sec_fails_to_deliver_file(
            MarketStructureSourceFile(
                source_name="SEC",
                source_url="https://example.test/bad.zip",
                content="SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
                "20260601|67066G104|NVDA|not-number|NVIDIA CORP|123.45\n",
            )
        )
