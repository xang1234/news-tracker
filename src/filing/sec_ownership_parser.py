"""Parse SEC ownership filings into normalized structured events."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any

from src.filing.schemas import FilingResult
from src.filing.sec_ownership_models import (
    SECOwnershipEvent,
    make_sec_ownership_event_id,
)


@dataclass(frozen=True)
class SECOwnershipParseResult:
    """Ownership parsing output with nonfatal data-quality errors."""

    events: list[SECOwnershipEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_sec_ownership_events(
    filing: FilingResult,
    *,
    previous_13f_events: list[SECOwnershipEvent] | None = None,
) -> SECOwnershipParseResult:
    """Parse supported ownership filings into normalized event records."""
    form = filing.identity.filing_type.upper()
    try:
        if form.startswith("4"):
            return _parse_form4(filing)
        if form.startswith("SC 13D") or form.startswith("SC 13G"):
            return _parse_schedule_13d_g(filing)
        if form.startswith("13F-HR"):
            return _parse_13f(filing, previous_13f_events or [])
    except ET.ParseError:
        return SECOwnershipParseResult(errors=["malformed_xml"])
    return SECOwnershipParseResult(errors=[f"unsupported_ownership_form:{form}"])


def _parse_form4(filing: FilingResult) -> SECOwnershipParseResult:
    root = _parse_xml(_content(filing))
    issuer = root.find("issuer")
    owner = root.find("reportingOwner/reportingOwnerId")
    events: list[SECOwnershipEvent] = []
    errors = _form4_mapping_errors(issuer, owner)

    non_derivative = list(root.findall(".//nonDerivativeTransaction"))
    derivative = list(root.findall(".//derivativeTransaction"))
    if not non_derivative and not derivative:
        return SECOwnershipParseResult(errors=["missing_form4_transaction_tables"])

    for index, node in enumerate(non_derivative):
        events.append(
            _form4_event(
                filing,
                node,
                issuer=issuer,
                owner=owner,
                event_type="form4_non_derivative_transaction",
                table="non_derivative",
                index=index,
            )
        )
    for index, node in enumerate(derivative):
        events.append(
            _form4_event(
                filing,
                node,
                issuer=issuer,
                owner=owner,
                event_type="form4_derivative_transaction",
                table="derivative",
                index=index,
            )
        )
    return SECOwnershipParseResult(events=events, errors=errors)


def _form4_event(
    filing: FilingResult,
    node: ET.Element,
    *,
    issuer: ET.Element | None,
    owner: ET.Element | None,
    event_type: str,
    table: str,
    index: int,
) -> SECOwnershipEvent:
    transaction_code = _xml_text(node, "transactionCoding/transactionCode")
    transaction_date = _parse_date(_xml_text(node, "transactionDate/value"))
    security_title = _xml_text(node, "securityTitle/value")
    issuer_mapping_status = _mapping_status(
        _xml_text(issuer, "issuerCik"),
        _xml_text(issuer, "issuerTradingSymbol"),
    )
    filer_mapping_status = _mapping_status(_xml_text(owner, "rptOwnerCik"))
    metadata = {
        "source_table": table,
        "transaction_index": index,
        "issuer_mapping_status": issuer_mapping_status,
        "filer_mapping_status": filer_mapping_status,
        "mapping_status": _combined_mapping_status(
            issuer_mapping_status,
            filer_mapping_status,
        ),
    }
    return SECOwnershipEvent(
        event_id=make_sec_ownership_event_id(
            [
                filing.identity.accession_number,
                event_type,
                table,
                index,
                transaction_code,
                transaction_date,
                security_title,
            ]
        ),
        event_type=event_type,
        accession_number=filing.identity.accession_number,
        filing_type=filing.identity.filing_type,
        filed_date=filing.identity.filed_date,
        issuer_cik=_xml_text(issuer, "issuerCik") or filing.identity.cik,
        issuer_name=_xml_text(issuer, "issuerName") or filing.identity.company_name,
        issuer_ticker=_xml_text(issuer, "issuerTradingSymbol") or filing.identity.ticker,
        filer_cik=_xml_text(owner, "rptOwnerCik"),
        filer_name=_xml_text(owner, "rptOwnerName"),
        security_title=security_title,
        transaction_code=transaction_code or None,
        transaction_date=transaction_date,
        transaction_shares=_parse_decimal(
            _xml_text(node, "transactionAmounts/transactionShares/value")
        ),
        transaction_price_per_share=_parse_decimal(
            _xml_text(node, "transactionAmounts/transactionPricePerShare/value")
        ),
        transaction_acquired_disposed_code=_xml_text(
            node,
            "transactionAmounts/transactionAcquiredDisposedCode/value",
        )
        or None,
        shares_owned_following=_parse_decimal(
            _xml_text(node, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")
        ),
        derivative_underlying_shares=_parse_decimal(
            _xml_text(node, "underlyingSecurity/underlyingSecurityShares/value")
        ),
        is_amendment=_is_amendment(filing.identity.filing_type),
        available_at=_available_at(filing),
        fetched_at=filing.fetched_at,
        source_url=filing.raw_url,
        metadata=metadata,
    )


def _parse_schedule_13d_g(filing: FilingResult) -> SECOwnershipParseResult:
    content = _content(filing)
    is_13g = filing.identity.filing_type.upper().startswith("SC 13G")
    metadata = dict(filing.metadata)
    issuer_cik = _metadata_str(metadata, "issuer_cik")
    issuer_name = _metadata_str(metadata, "issuer_name")
    issuer_ticker = _metadata_str(metadata, "issuer_ticker")
    if not _mapping_has_identifier(issuer_cik, issuer_ticker) and filing.identity.ticker:
        issuer_cik = filing.identity.cik
        issuer_name = issuer_name or filing.identity.company_name
        issuer_ticker = filing.identity.ticker
    filer_cik = _metadata_str(metadata, "filer_cik") or filing.identity.cik
    filer_name = _metadata_str(metadata, "filer_name") or filing.identity.company_name
    issuer_mapping_status = _mapping_status(issuer_cik, issuer_ticker)
    filer_mapping_status = _mapping_status(filer_cik)
    errors = []
    if issuer_mapping_status == "unresolved":
        errors.append("issuer_mapping_failed:schedule")
    if filer_mapping_status == "unresolved":
        errors.append("filer_mapping_failed:schedule")
    event_type = "schedule_13g_ownership" if is_13g else "schedule_13d_ownership"
    event = SECOwnershipEvent(
        event_id=make_sec_ownership_event_id(
            [filing.identity.accession_number, event_type, _extract_cusip(content)]
        ),
        event_type=event_type,
        accession_number=filing.identity.accession_number,
        filing_type=filing.identity.filing_type,
        filed_date=filing.identity.filed_date,
        issuer_cik=issuer_cik,
        issuer_name=issuer_name,
        issuer_ticker=issuer_ticker,
        filer_cik=filer_cik,
        filer_name=filer_name,
        ownership_percent=_extract_percent(content),
        position_cusip=_extract_cusip(content),
        position_shares=_extract_beneficial_shares(content),
        is_amendment=_is_amendment(filing.identity.filing_type),
        available_at=_available_at(filing),
        fetched_at=filing.fetched_at,
        source_url=filing.raw_url,
        metadata={
            "issuer_mapping_status": issuer_mapping_status,
            "filer_mapping_status": filer_mapping_status,
            "mapping_status": _combined_mapping_status(
                issuer_mapping_status,
                filer_mapping_status,
            ),
        },
    )
    return SECOwnershipParseResult(events=[event], errors=errors)


def _parse_13f(
    filing: FilingResult,
    previous_events: list[SECOwnershipEvent],
) -> SECOwnershipParseResult:
    root = _parse_xml(_content(filing))
    rows = list(root.findall(".//infoTable"))
    if not rows:
        return SECOwnershipParseResult(errors=["missing_13f_info_table"])

    previous_by_cusip = {
        event.position_cusip: event for event in previous_events if event.position_cusip is not None
    }
    issuer_mappings = _issuer_mappings(filing.metadata)
    filer_cik = _metadata_str(filing.metadata, "filer_cik") or filing.identity.cik
    filer_name = _metadata_str(filing.metadata, "filer_name") or filing.identity.company_name
    filer_mapping_status = _mapping_status(filer_cik)
    errors: list[str] = []
    if filer_mapping_status == "unresolved":
        errors.append("filer_mapping_failed:13f")
    events: list[SECOwnershipEvent] = []
    for index, row in enumerate(rows):
        cusip = _xml_text(row, "cusip").upper()
        mapping = issuer_mappings.get(cusip, {})
        issuer_mapping_status = "resolved" if mapping else "unresolved"
        if issuer_mapping_status == "unresolved":
            errors.append(f"issuer_mapping_failed:{cusip}")
        shares = _parse_decimal(_xml_text(row, "shrsOrPrnAmt/sshPrnamt"))
        previous = previous_by_cusip.get(cusip)
        previous_shares = previous.position_shares if previous else None
        events.append(
            SECOwnershipEvent(
                event_id=make_sec_ownership_event_id(
                    [filing.identity.accession_number, "13f_position", cusip]
                ),
                event_type="13f_position",
                accession_number=filing.identity.accession_number,
                filing_type=filing.identity.filing_type,
                filed_date=filing.identity.filed_date,
                issuer_cik=_metadata_str(mapping, "issuer_cik"),
                issuer_name=_metadata_str(mapping, "issuer_name") or _xml_text(row, "nameOfIssuer"),
                issuer_ticker=_metadata_str(mapping, "issuer_ticker") or None,
                filer_cik=filer_cik,
                filer_name=filer_name,
                security_title=_xml_text(row, "titleOfClass"),
                position_cusip=cusip,
                position_shares=shares,
                position_value_usd=_parse_13f_value_usd(_xml_text(row, "value")),
                previous_position_shares=previous_shares,
                position_delta_shares=(
                    shares - previous_shares
                    if shares is not None and previous_shares is not None
                    else None
                ),
                is_amendment=_is_amendment(filing.identity.filing_type),
                available_at=_available_at(filing),
                fetched_at=filing.fetched_at,
                source_url=filing.raw_url,
                metadata={
                    "mapping_status": issuer_mapping_status,
                    "issuer_mapping_status": issuer_mapping_status,
                    "filer_mapping_status": filer_mapping_status,
                    "info_table_index": index,
                    "share_type": _xml_text(row, "shrsOrPrnAmt/sshPrnamtType"),
                    "investment_discretion": _xml_text(row, "investmentDiscretion"),
                },
            )
        )
    return SECOwnershipParseResult(events=events, errors=errors)


def _parse_xml(content: str) -> ET.Element:
    root = ET.fromstring(content)
    _strip_namespaces(root)
    return root


def _strip_namespaces(node: ET.Element) -> None:
    for element in node.iter():
        if "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]


def _content(filing: FilingResult) -> str:
    return "\n".join(section.content for section in filing.sections if section.content)


def _xml_text(node: ET.Element | None, path: str) -> str:
    current = node
    if current is None:
        return ""
    for part in path.split("/"):
        current = current.find(part)
        if current is None:
            return ""
    return (current.text or "").strip()


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_decimal(value: str) -> Decimal | None:
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _parse_13f_value_usd(value: str) -> Decimal | None:
    parsed = _parse_decimal(value)
    return parsed * Decimal("1000") if parsed is not None else None


def _extract_percent(content: str) -> Decimal | None:
    match = re.search(r"Percent of Class.*?(\d+(?:\.\d+)?)\s*%", content, re.I | re.S)
    return _parse_decimal(match.group(1)) if match else None


def _extract_beneficial_shares(content: str) -> Decimal | None:
    match = re.search(
        r"Aggregate Amount Beneficially Owned.*?(\d[\d,]*)",
        content,
        re.I | re.S,
    )
    return _parse_decimal(match.group(1)) if match else None


def _extract_cusip(content: str) -> str | None:
    match = re.search(r"CUSIP(?: No\.)?\s*([A-Z0-9]{6,12})", content, re.I)
    return match.group(1).upper() if match else None


def _metadata_str(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return str(value).strip() if value is not None else ""


def _mapping_has_identifier(*values: str | None) -> bool:
    return any(value.strip() for value in values if value is not None)


def _mapping_status(*values: str | None) -> str:
    return "resolved" if _mapping_has_identifier(*values) else "unresolved"


def _combined_mapping_status(*statuses: str) -> str:
    if all(status == "resolved" for status in statuses):
        return "resolved"
    if all(status == "unresolved" for status in statuses):
        return "unresolved"
    return "partial"


def _form4_mapping_errors(
    issuer: ET.Element | None,
    owner: ET.Element | None,
) -> list[str]:
    errors = []
    issuer_status = _mapping_status(
        _xml_text(issuer, "issuerCik"),
        _xml_text(issuer, "issuerTradingSymbol"),
    )
    if issuer_status == "unresolved":
        errors.append("issuer_mapping_failed:form4")
    if _mapping_status(_xml_text(owner, "rptOwnerCik")) == "unresolved":
        errors.append("filer_mapping_failed:form4")
    return errors


def _issuer_mappings(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = metadata.get("issuer_mappings")
    if not isinstance(raw, dict):
        return {}
    return {
        str(cusip).upper(): dict(mapping)
        for cusip, mapping in raw.items()
        if isinstance(mapping, dict)
    }


def _available_at(filing: FilingResult) -> datetime:
    return datetime.combine(filing.identity.filed_date, time.min, tzinfo=UTC)


def _is_amendment(filing_type: str) -> bool:
    return filing_type.upper().endswith("/A")
