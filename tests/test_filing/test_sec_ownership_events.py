"""Tests for SEC ownership filing event parsing."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from src.filing.schemas import FilingIdentity, FilingResult, FilingSection
from src.filing.sec_ownership_events import (
    SECOwnershipEvent,
    parse_sec_ownership_events,
)

FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <periodOfReport>2026-05-29</periodOfReport>
  <issuer>
    <issuerCik>0001045810</issuerCik>
    <issuerName>NVIDIA Corporation</issuerName>
    <issuerTradingSymbol>NVDA</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001999999</rptOwnerCik>
      <rptOwnerName>Example Insider</rptOwnerName>
    </reportingOwnerId>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-05-28</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>125</value></transactionShares>
        <transactionPricePerShare><value>123.45</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>875</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <derivativeTable>
    <derivativeTransaction>
      <securityTitle><value>Employee Stock Option</value></securityTitle>
      <conversionOrExercisePrice><value>10.00</value></conversionOrExercisePrice>
      <transactionDate><value>2026-05-28</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>50</value></transactionShares>
        <transactionPricePerShare><value>10.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <underlyingSecurity>
        <underlyingSecurityTitle><value>Common Stock</value></underlyingSecurityTitle>
        <underlyingSecurityShares><value>50</value></underlyingSecurityShares>
      </underlyingSecurity>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>
"""


FORM13F_XML = """<?xml version="1.0"?>
<informationTable>
  <infoTable>
    <nameOfIssuer>NVIDIA CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>67066G104</cusip>
    <value>125000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>1000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
</informationTable>
"""


def _filing(form: str, content: str, metadata: dict | None = None) -> FilingResult:
    return FilingResult(
        identity=FilingIdentity(
            cik="1999999",
            accession_number=f"0001999999-26-{form.replace('/', '')}",
            filing_type=form,
            filed_date=date(2026, 6, 1),
            company_name="Example Filer",
        ),
        sections=[
            FilingSection(
                section_id="section-1",
                section_name="Ownership XML",
                section_type="xml",
                content=content,
                word_count=len(content.split()),
            )
        ],
        raw_url="https://www.sec.gov/Archives/example.txt",
        provider="sec_api",
        fetched_at=datetime(2026, 6, 1, 17, tzinfo=UTC),
        metadata=metadata or {},
    )


def test_form4_parses_non_derivative_and_derivative_transactions() -> None:
    result = parse_sec_ownership_events(_filing("4", FORM4_XML))

    assert result.errors == []
    assert [event.event_type for event in result.events] == [
        "form4_non_derivative_transaction",
        "form4_derivative_transaction",
    ]
    non_derivative, derivative = result.events
    assert non_derivative.issuer_cik == "0001045810"
    assert non_derivative.issuer_ticker == "NVDA"
    assert non_derivative.filer_cik == "0001999999"
    assert non_derivative.transaction_code == "S"
    assert non_derivative.transaction_shares == Decimal("125")
    assert non_derivative.transaction_acquired_disposed_code == "D"
    assert non_derivative.shares_owned_following == Decimal("875")
    assert derivative.transaction_code == "M"
    assert derivative.derivative_underlying_shares == Decimal("50")


def test_form4_amendment_and_missing_tables_are_explicit() -> None:
    amendment = parse_sec_ownership_events(_filing("4/A", FORM4_XML))
    missing_tables = parse_sec_ownership_events(
        _filing(
            "4",
            "<ownershipDocument><issuer><issuerCik>1045810</issuerCik></issuer></ownershipDocument>",
        )
    )

    assert all(event.is_amendment for event in amendment.events)
    assert missing_tables.events == []
    assert missing_tables.errors == ["missing_form4_transaction_tables"]


def test_form4_filer_mapping_failure_is_reported_without_dropping_transaction() -> None:
    result = parse_sec_ownership_events(
        _filing(
            "4",
            FORM4_XML.replace(
                "<reportingOwner>\n"
                "    <reportingOwnerId>\n"
                "      <rptOwnerCik>0001999999</rptOwnerCik>\n"
                "      <rptOwnerName>Example Insider</rptOwnerName>\n"
                "    </reportingOwnerId>\n"
                "  </reportingOwner>",
                "",
            ),
        )
    )

    assert result.errors == ["filer_mapping_failed:form4"]
    assert result.events[0].filer_cik == ""
    assert result.events[0].metadata["filer_mapping_status"] == "unresolved"
    assert result.events[0].transaction_code == "S"


def test_schedule_13d_threshold_event_parses_percent_and_mapping_status() -> None:
    result = parse_sec_ownership_events(
        _filing(
            "SC 13D",
            "CUSIP No. 67066G104\n"
            "Aggregate Amount Beneficially Owned by Each Reporting Person 12,500,000\n"
            "Percent of Class Represented by Amount in Row (11) 7.2%",
            metadata={
                "issuer_cik": "1045810",
                "issuer_name": "NVIDIA Corporation",
                "issuer_ticker": "NVDA",
                "filer_cik": "1999999",
            },
        )
    )

    assert result.errors == []
    event = result.events[0]
    assert event.event_type == "schedule_13d_ownership"
    assert event.ownership_percent == Decimal("7.2")
    assert event.position_shares == Decimal("12500000")
    assert event.metadata["mapping_status"] == "resolved"


def test_schedule_13g_amendment_does_not_use_filer_identity_as_issuer() -> None:
    result = parse_sec_ownership_events(
        _filing(
            "SC 13G/A",
            "CUSIP No. 67066G104\n"
            "Aggregate Amount Beneficially Owned by Each Reporting Person 12,500,000\n"
            "Percent of Class Represented by Amount in Row (11) 7.2%",
        )
    )

    assert result.errors == ["issuer_mapping_failed:schedule"]
    event = result.events[0]
    assert event.event_type == "schedule_13g_ownership"
    assert event.is_amendment is True
    assert event.issuer_cik == ""
    assert event.filer_cik == "0001999999"
    assert event.metadata["issuer_mapping_status"] == "unresolved"
    assert event.metadata["filer_mapping_status"] == "resolved"
    assert event.metadata["mapping_status"] == "partial"


def test_13f_positions_include_quarterly_position_change() -> None:
    previous = SECOwnershipEvent(
        event_id="previous",
        event_type="13f_position",
        accession_number="0001999999-26-old",
        filing_type="13F-HR",
        filed_date=date(2026, 3, 1),
        issuer_cik="1045810",
        issuer_name="NVIDIA Corporation",
        issuer_ticker="NVDA",
        filer_cik="1999999",
        filer_name="Example Filer",
        position_cusip="67066G104",
        position_shares=Decimal("800"),
        position_value_usd=Decimal("100000000"),
        available_at=datetime(2026, 3, 1, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 1, tzinfo=UTC),
    )

    result = parse_sec_ownership_events(
        _filing(
            "13F-HR",
            FORM13F_XML,
            metadata={
                "issuer_mappings": {
                    "67066G104": {
                        "issuer_cik": "1045810",
                        "issuer_name": "NVIDIA Corporation",
                        "issuer_ticker": "NVDA",
                    }
                }
            },
        ),
        previous_13f_events=[previous],
    )

    event = result.events[0]
    assert event.event_type == "13f_position"
    assert event.position_cusip == "67066G104"
    assert event.position_value_usd == Decimal("125000000")
    assert event.position_shares == Decimal("1000")
    assert event.previous_position_shares == Decimal("800")
    assert event.position_delta_shares == Decimal("200")


def test_13f_missing_xml_table_and_mapping_failure_are_nonfatal() -> None:
    missing = parse_sec_ownership_events(_filing("13F-HR", "<informationTable />"))
    unresolved = parse_sec_ownership_events(_filing("13F-HR", FORM13F_XML))

    assert missing.events == []
    assert missing.errors == ["missing_13f_info_table"]
    assert unresolved.events[0].metadata["mapping_status"] == "unresolved"
    assert unresolved.errors == ["issuer_mapping_failed:67066G104"]


def test_malformed_xml_returns_parse_error() -> None:
    result = parse_sec_ownership_events(_filing("4", "<ownershipDocument>"))

    assert result.events == []
    assert result.errors == ["malformed_xml"]
