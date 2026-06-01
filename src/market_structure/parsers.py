"""Parsers for free market-structure datasource files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

from src.market_structure.models import (
    MarketStructureEvent,
    MarketStructureSourceFile,
    make_market_structure_event_id,
)
from src.security_master.schemas import Security

_FINRA_HEADERS = (
    "Date",
    "Symbol",
    "ShortVolume",
    "ShortExemptVolume",
    "TotalVolume",
    "Market",
)
_SEC_FTD_HEADERS = (
    "SETTLEMENT DATE",
    "CUSIP",
    "SYMBOL",
    "QUANTITY (FAILS)",
    "DESCRIPTION",
    "PRICE",
)
_FINRA_MARKET_NAMES = {
    "N": "NYSE TRF",
    "Q": "NASDAQ TRF Carteret",
    "B": "NASDAQ TRF Chicago",
    "D": "ADF",
    "O": "ORF",
}
_NEW_YORK = ZoneInfo("America/New_York")


def parse_finra_short_volume_file(
    source: MarketStructureSourceFile,
    *,
    securities_by_symbol: Mapping[str, Security] | None = None,
) -> list[MarketStructureEvent]:
    """Parse a FINRA Reg SHO daily short-sale volume file."""
    rows = _parse_pipe_rows(
        source.content,
        required_headers=_FINRA_HEADERS,
        source_name=source.source_name,
    )
    securities = _normalize_security_map(securities_by_symbol)
    events: list[MarketStructureEvent] = []
    for parsed_row in rows:
        row = parsed_row.values
        trade_date = _parse_yyyymmdd(row["Date"], parsed_row.line_number, "Date")
        symbol = _required_symbol(row["Symbol"], parsed_row.line_number, "Symbol")
        short_volume = _parse_int(row["ShortVolume"], parsed_row.line_number, "ShortVolume")
        short_exempt_volume = _parse_int(
            row["ShortExemptVolume"],
            parsed_row.line_number,
            "ShortExemptVolume",
        )
        total_volume = _parse_int(row["TotalVolume"], parsed_row.line_number, "TotalVolume")
        market_code = row["Market"].upper()
        security = securities.get(symbol)
        mapping = _security_mapping(symbol, security, row.get("DESCRIPTION", ""))
        events.append(
            MarketStructureEvent(
                event_id=make_market_structure_event_id(
                    [
                        "finra_short_volume",
                        trade_date,
                        symbol,
                        market_code,
                    ]
                ),
                event_type="finra_short_volume",
                source_name=source.source_name,
                source_url=source.source_url,
                source_date=trade_date,
                trade_date=trade_date,
                symbol=symbol,
                security_ticker=mapping.security_ticker,
                security_exchange=mapping.security_exchange,
                issuer_cik=mapping.issuer_cik,
                issuer_name=mapping.issuer_name,
                market_code=market_code,
                market_name=_FINRA_MARKET_NAMES.get(market_code, market_code),
                short_volume=short_volume,
                short_exempt_volume=short_exempt_volume,
                total_volume=total_volume,
                short_volume_ratio=_safe_ratio(short_volume, total_volume),
                short_exempt_ratio=_safe_ratio(short_exempt_volume, total_volume),
                available_at=_finra_available_at(trade_date),
                fetched_at=source.fetched_at,
                metadata={
                    "source_row_number": parsed_row.line_number,
                    "mapping_status": mapping.status,
                    "raw_fields": dict(row),
                },
            )
        )
    return events


def parse_sec_fails_to_deliver_file(
    source: MarketStructureSourceFile,
    *,
    securities_by_symbol: Mapping[str, Security] | None = None,
) -> list[MarketStructureEvent]:
    """Parse an SEC fails-to-deliver pipe-delimited text file."""
    rows = _parse_pipe_rows(
        source.content,
        required_headers=_SEC_FTD_HEADERS,
        source_name=source.source_name,
    )
    securities = _normalize_security_map(securities_by_symbol)
    events: list[MarketStructureEvent] = []
    for parsed_row in rows:
        row = parsed_row.values
        settlement_date = _parse_yyyymmdd(
            row["SETTLEMENT DATE"],
            parsed_row.line_number,
            "SETTLEMENT DATE",
        )
        symbol = _required_symbol(row["SYMBOL"], parsed_row.line_number, "SYMBOL")
        quantity = _parse_int(
            row["QUANTITY (FAILS)"],
            parsed_row.line_number,
            "QUANTITY (FAILS)",
        )
        price = _parse_optional_decimal(row["PRICE"], parsed_row.line_number, "PRICE")
        security = securities.get(symbol)
        mapping = _security_mapping(symbol, security, row["DESCRIPTION"])
        events.append(
            MarketStructureEvent(
                event_id=make_market_structure_event_id(
                    [
                        "sec_fail_to_deliver",
                        settlement_date,
                        row["CUSIP"].upper(),
                        symbol,
                    ]
                ),
                event_type="sec_fail_to_deliver",
                source_name=source.source_name,
                source_url=source.source_url,
                source_date=settlement_date,
                settlement_date=settlement_date,
                symbol=symbol,
                security_ticker=mapping.security_ticker,
                security_exchange=mapping.security_exchange,
                issuer_cik=mapping.issuer_cik,
                issuer_name=mapping.issuer_name or row["DESCRIPTION"],
                cusip=row["CUSIP"].upper(),
                fail_quantity=quantity,
                fail_price=price,
                fail_notional=price * quantity if price is not None else None,
                available_at=_sec_ftd_available_at(source.fetched_at, settlement_date),
                fetched_at=source.fetched_at,
                metadata={
                    "source_row_number": parsed_row.line_number,
                    "mapping_status": mapping.status,
                    "raw_fields": dict(row),
                },
            )
        )
    return events


def apply_market_structure_signals(
    events: list[MarketStructureEvent],
    *,
    short_ratio_watch: Decimal = Decimal("0.5"),
    short_ratio_extreme: Decimal = Decimal("0.75"),
    ftd_notional_watch: Decimal = Decimal("1000000"),
    ftd_notional_extreme: Decimal = Decimal("10000000"),
) -> list[MarketStructureEvent]:
    """Attach ratio, persistence, and threshold fields without position claims."""
    ordered = sorted(
        enumerate(events),
        key=lambda item: (item[1].symbol, item[1].source_date, item[1].event_id),
    )
    previous_by_key: dict[tuple[str, str], MarketStructureEvent] = {}
    enriched_by_index: dict[int, MarketStructureEvent] = {}
    for original_index, event in ordered:
        if event.event_type == "finra_short_volume":
            signal_type = "short_volume_ratio"
            anomaly_level = _short_ratio_level(
                event.short_volume_ratio,
                watch=short_ratio_watch,
                extreme=short_ratio_extreme,
            )
        else:
            signal_type = "fails_to_deliver_notional"
            anomaly_level = _ftd_notional_level(
                event.fail_notional,
                watch=ftd_notional_watch,
                extreme=ftd_notional_extreme,
            )

        key = (event.event_type, event.symbol)
        previous = previous_by_key.get(key)
        persistence_count, reset_reason = _next_persistence(event, previous, anomaly_level)
        metadata = dict(event.metadata)
        if reset_reason is not None:
            metadata["persistence_reset_reason"] = reset_reason
        updated = replace(
            event,
            signal_type=signal_type,
            anomaly_level=anomaly_level,
            persistence_count=persistence_count,
            metadata=metadata,
        )
        previous_by_key[key] = updated
        enriched_by_index[original_index] = updated
    return [enriched_by_index[index] for index in range(len(events))]


def apply_security_mappings(
    events: list[MarketStructureEvent],
    *,
    securities_by_symbol: Mapping[str, Security] | None = None,
) -> list[MarketStructureEvent]:
    """Attach current security-master mapping to already parsed events."""
    securities = _normalize_security_map(securities_by_symbol)
    mapped: list[MarketStructureEvent] = []
    for event in events:
        raw_fields = event.metadata.get("raw_fields")
        fallback_issuer_name = (
            str(raw_fields.get("DESCRIPTION", ""))
            if isinstance(raw_fields, dict)
            else event.issuer_name
        )
        mapping = _security_mapping(
            event.symbol,
            securities.get(event.symbol),
            fallback_issuer_name,
        )
        metadata = dict(event.metadata)
        metadata["mapping_status"] = mapping.status
        mapped.append(
            replace(
                event,
                security_ticker=mapping.security_ticker,
                security_exchange=mapping.security_exchange,
                issuer_cik=mapping.issuer_cik,
                issuer_name=mapping.issuer_name or event.issuer_name,
                metadata=metadata,
            )
        )
    return mapped


@dataclass(frozen=True)
class _ParsedRow:
    values: dict[str, str]
    line_number: int


@dataclass(frozen=True)
class _SecurityMapping:
    security_ticker: str
    security_exchange: str
    issuer_cik: str
    issuer_name: str
    status: str


def _parse_pipe_rows(
    content: str,
    *,
    required_headers: tuple[str, ...],
    source_name: str,
) -> list[_ParsedRow]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{source_name} file is empty")
    raw_headers = tuple(header.strip() for header in lines[0].split("|"))
    headers = _canonical_headers(raw_headers, required_headers, source_name)
    rows: list[_ParsedRow] = []
    trailer_count: int | None = None
    for line_number, line in enumerate(lines[1:], start=2):
        values = [value.strip() for value in line.split("|")]
        if len(values) == 1 and values[0].isdigit():
            trailer_count = int(values[0])
            continue
        if len(values) < len(headers):
            raise ValueError(
                f"{source_name} line {line_number} has {len(values)} fields; "
                f"expected at least {len(headers)}"
            )
        rows.append(
            _ParsedRow(
                {header: values[index] for index, header in enumerate(headers)},
                line_number,
            )
        )
    if trailer_count is not None and trailer_count != len(rows):
        raise ValueError(
            f"{source_name} trailer count {trailer_count} does not match "
            f"{len(rows)} parsed rows"
        )
    return rows


def _canonical_headers(
    raw_headers: tuple[str, ...],
    required_headers: tuple[str, ...],
    source_name: str,
) -> tuple[str, ...]:
    required_by_key = {_header_key(header): header for header in required_headers}
    canonical = tuple(
        required_by_key.get(_header_key(header), header.strip()) for header in raw_headers
    )
    missing = [header for header in required_headers if header not in canonical]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{source_name} missing required columns: {joined}")
    return canonical


def _header_key(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _normalize_security_map(
    securities_by_symbol: Mapping[str, Security] | None,
) -> dict[str, Security]:
    if securities_by_symbol is None:
        return {}
    return {symbol.upper(): security for symbol, security in securities_by_symbol.items()}


def _security_mapping(
    symbol: str,
    security: Security | None,
    fallback_issuer_name: str,
) -> _SecurityMapping:
    if security is None:
        return _SecurityMapping(
            security_ticker=symbol,
            security_exchange="US",
            issuer_cik="",
            issuer_name=fallback_issuer_name,
            status="unresolved",
        )
    return _SecurityMapping(
        security_ticker=security.ticker,
        security_exchange=security.exchange,
        issuer_cik=security.sec_cik or "",
        issuer_name=security.issuer_name or security.name,
        status="resolved",
    )


def _parse_yyyymmdd(value: str, line_number: int, field_name: str) -> date:
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"line {line_number}: {field_name} must be YYYYMMDD")
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except ValueError as exc:
        raise ValueError(f"line {line_number}: {field_name} must be a valid date") from exc


def _parse_int(value: str, line_number: int, field_name: str) -> int:
    if not value.isdigit():
        raise ValueError(f"line {line_number}: {field_name} must be a non-negative integer")
    return int(value)


def _required_symbol(value: str, line_number: int, field_name: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError(f"line {line_number}: {field_name} must be non-empty")
    return symbol


def _parse_optional_decimal(value: str, line_number: int, field_name: str) -> Decimal | None:
    if not value or value == ".":
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"line {line_number}: {field_name} must be a decimal") from exc


def _safe_ratio(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    return Decimal(numerator) / Decimal(denominator)


def _finra_available_at(trade_date: date) -> datetime:
    eastern_post_time = datetime.combine(trade_date, time(18), tzinfo=_NEW_YORK)
    return eastern_post_time.astimezone(UTC)


def _sec_ftd_available_at(fetched_at: datetime | None, settlement_date: date) -> datetime:
    if fetched_at is not None:
        return fetched_at.astimezone(UTC) if fetched_at.tzinfo else fetched_at.replace(tzinfo=UTC)
    return datetime.combine(settlement_date, time.min, tzinfo=UTC)


def _short_ratio_level(
    ratio: Decimal | None,
    *,
    watch: Decimal,
    extreme: Decimal,
) -> str:
    if ratio is None:
        return "none"
    if ratio >= extreme:
        return "extreme"
    if ratio >= watch:
        return "elevated"
    return "none"


def _ftd_notional_level(
    notional: Decimal | None,
    *,
    watch: Decimal,
    extreme: Decimal,
) -> str:
    if notional is None:
        return "none"
    if notional >= extreme:
        return "extreme"
    if notional >= watch:
        return "watch"
    return "none"


def _next_persistence(
    event: MarketStructureEvent,
    previous: MarketStructureEvent | None,
    anomaly_level: str,
) -> tuple[int, str | None]:
    if anomaly_level == "none":
        return 0, None
    if previous is None or previous.anomaly_level == "none":
        return 1, None
    if _has_missing_trading_days(previous.source_date, event.source_date):
        return 1, "missing_trading_days"
    return previous.persistence_count + 1, None


def _has_missing_trading_days(previous_date: date, current_date: date) -> bool:
    next_expected = previous_date + timedelta(days=1)
    while next_expected.weekday() >= 5:
        next_expected += timedelta(days=1)
    return current_date > next_expected
