"""Tests for publishing market-plumbing events with guardrails."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from src.filing.sec_ownership_events import SECOwnershipEvent
from src.market_plumbing import (
    build_market_plumbing_alerts,
    build_market_plumbing_read_model,
)
from src.market_structure import MarketStructureEvent

NOW = datetime(2026, 6, 1, 22, tzinfo=UTC)


def _ownership_event(event_type: str) -> SECOwnershipEvent:
    return SECOwnershipEvent(
        event_id=f"ownership:{event_type}",
        event_type=event_type,
        accession_number=f"0001045810-26-{event_type}",
        filing_type={
            "form4_non_derivative_transaction": "4",
            "schedule_13d_ownership": "SC 13D",
            "schedule_13g_ownership": "SC 13G",
            "13f_position": "13F-HR",
        }[event_type],
        filed_date=date(2026, 6, 1),
        available_at=NOW,
        issuer_cik="1045810",
        issuer_name="NVIDIA Corporation",
        issuer_ticker="NVDA",
        filer_cik="1999999",
        filer_name="Example Filer",
        transaction_code="P" if event_type.startswith("form4") else None,
        transaction_shares=Decimal("125") if event_type.startswith("form4") else None,
        ownership_percent=Decimal("7.2") if event_type.startswith("schedule") else None,
        position_cusip="67066G104" if event_type == "13f_position" else None,
        position_delta_shares=Decimal("200") if event_type == "13f_position" else None,
        source_url="https://www.sec.gov/Archives/example.txt",
        metadata={"mapping_status": "resolved"},
    )


def _market_event(event_type: str) -> MarketStructureEvent:
    if event_type == "finra_short_volume":
        return MarketStructureEvent(
            event_id="market:short-volume",
            event_type="finra_short_volume",
            source_name="FINRA daily short-volume",
            source_url="https://example.test/CNMSshvol20260601.txt",
            source_date=date(2026, 6, 1),
            trade_date=date(2026, 6, 1),
            symbol="NVDA",
            security_ticker="NVDA",
            issuer_cik="1045810",
            issuer_name="NVIDIA Corporation",
            market_code="Q",
            market_name="NASDAQ TRF Carteret",
            short_volume=800,
            total_volume=1000,
            short_volume_ratio=Decimal("0.8"),
            signal_type="short_volume_ratio",
            anomaly_level="extreme",
            persistence_count=3,
            available_at=NOW,
            fetched_at=NOW,
            metadata={"mapping_status": "resolved"},
        )
    return MarketStructureEvent(
        event_id="market:ftd",
        event_type="sec_fail_to_deliver",
        source_name="SEC fails-to-deliver",
        source_url="https://www.sec.gov/files/cnsfails202606a.zip",
        source_date=date(2026, 6, 1),
        settlement_date=date(2026, 6, 1),
        symbol="NVDA",
        security_ticker="NVDA",
        issuer_cik="1045810",
        issuer_name="NVIDIA Corporation",
        cusip="67066G104",
        fail_quantity=12000,
        fail_price=Decimal("123.45"),
        fail_notional=Decimal("1481400.00"),
        signal_type="fails_to_deliver_notional",
        anomaly_level="watch",
        persistence_count=1,
        available_at=NOW,
        fetched_at=NOW,
        metadata={"mapping_status": "resolved"},
    )


def test_alerts_distinguish_ownership_and_market_plumbing_signal_classes() -> None:
    alerts = build_market_plumbing_alerts(
        theme_id="theme_market_plumbing",
        ownership_events=[
            _ownership_event("form4_non_derivative_transaction"),
            _ownership_event("schedule_13d_ownership"),
            _ownership_event("13f_position"),
        ],
        market_structure_events=[
            _market_event("finra_short_volume"),
            _market_event("sec_fail_to_deliver"),
        ],
    )

    by_type = {alert.trigger_type: alert for alert in alerts}

    assert set(by_type) == {
        "insider_ownership",
        "activist_ownership",
        "institutional_holdings",
        "short_volume_anomaly",
        "fails_to_deliver_anomaly",
    }
    assert by_type["insider_ownership"].trigger_data["signal_label"] == (
        "Insider ownership transaction"
    )
    assert by_type["activist_ownership"].trigger_data["signal_label"] == (
        "Activist ownership disclosure"
    )
    assert by_type["institutional_holdings"].trigger_data["signal_label"] == (
        "Institutional holdings update"
    )
    assert by_type["short_volume_anomaly"].trigger_data["signal_label"] == (
        "Short-volume anomaly"
    )
    assert by_type["fails_to_deliver_anomaly"].trigger_data["signal_label"] == (
        "Fails-to-deliver anomaly"
    )


def test_market_structure_alert_payloads_include_visible_caveats() -> None:
    alerts = build_market_plumbing_alerts(
        theme_id="theme_market_plumbing",
        market_structure_events=[
            _market_event("finra_short_volume"),
            _market_event("sec_fail_to_deliver"),
        ],
    )

    short_volume = alerts[0].to_dict()["trigger_data"]
    ftd = alerts[1].to_dict()["trigger_data"]

    assert "caveats" in short_volume
    assert "short interest" in " ".join(short_volume["caveats"]).lower()
    assert "position" in " ".join(short_volume["caveats"]).lower()
    assert "caveats" in ftd
    assert "open short position" in " ".join(ftd["caveats"]).lower()
    assert short_volume["available_at"] == NOW.isoformat()
    assert ftd["available_at"] == NOW.isoformat()


def test_read_model_preserves_lineage_available_at_and_guardrails() -> None:
    payloads = build_market_plumbing_read_model(
        ownership_events=[_ownership_event("schedule_13d_ownership")],
        market_structure_events=[_market_event("sec_fail_to_deliver")],
    )

    ownership, ftd = payloads

    assert ownership["object_type"] == "market_plumbing_signal"
    assert ownership["signal_class"] == "activist_ownership"
    assert ownership["lineage"]["available_at"] == NOW.isoformat()
    assert ownership["lineage"]["source_url"] == "https://www.sec.gov/Archives/example.txt"
    assert ownership["guardrails"] == []
    assert ftd["signal_class"] == "fails_to_deliver_anomaly"
    assert ftd["lineage"]["available_at"] == NOW.isoformat()
    assert ftd["guardrails"]


def test_alert_builder_requires_existing_theme_context() -> None:
    try:
        build_market_plumbing_alerts(
            theme_id="",
            ownership_events=[_ownership_event("schedule_13d_ownership")],
        )
    except ValueError as exc:
        assert "theme_id" in str(exc)
    else:
        raise AssertionError("empty theme_id should be rejected")


def test_schedule_13g_is_not_labeled_as_activist_ownership() -> None:
    payloads = build_market_plumbing_read_model(
        ownership_events=[_ownership_event("schedule_13g_ownership")]
    )

    assert payloads[0]["signal_class"] == "institutional_holdings"
    assert payloads[0]["signal_label"] == "Institutional holdings update"


def test_non_anomalous_market_structure_rows_do_not_publish_as_anomalies() -> None:
    event = _market_event("finra_short_volume")
    event.anomaly_level = "none"

    alerts = build_market_plumbing_alerts(
        theme_id="theme_market_plumbing",
        market_structure_events=[event],
    )
    payloads = build_market_plumbing_read_model(market_structure_events=[event])

    assert alerts == []
    assert payloads == []
