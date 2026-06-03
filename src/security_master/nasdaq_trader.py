"""Nasdaq Trader symbol-directory parsing and security-master reconciliation."""

from __future__ import annotations

from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from src.security_master.schemas import Security, SecurityIdentifierLineage

NasdaqTraderSourceFile = Literal["nasdaqlisted", "otherlisted"]

NASDAQ_TRADER_EXTERNAL_KEY = "nasdaq_trader"
NASDAQ_TRADER_LINEAGE_SOURCE = "nasdaq_trader_symbol_directory"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SECURITY_MASTER_US_EXCHANGE = "US"

_NASDAQ_REQUIRED_COLUMNS = (
    "Symbol",
    "Security Name",
    "Market Category",
    "Test Issue",
    "Financial Status",
    "Round Lot Size",
)
_OTHER_LISTED_REQUIRED_COLUMNS = (
    "ACT Symbol",
    "Security Name",
    "Exchange",
    "CQS Symbol",
    "ETF",
    "Round Lot Size",
    "Test Issue",
    "NASDAQ Symbol",
)
_MARKET_CATEGORY_NAMES = {
    "Q": "Nasdaq Global Select Market",
    "G": "Nasdaq Global Market",
    "S": "Nasdaq Capital Market",
}
_FINANCIAL_STATUS_NAMES = {
    "D": "Deficient",
    "E": "Delinquent",
    "Q": "Bankrupt",
    "N": "Normal",
    "G": "Deficient and Bankrupt",
    "H": "Deficient and Delinquent",
    "J": "Delinquent and Bankrupt",
    "K": "Deficient, Delinquent, and Bankrupt",
}
_OTHER_EXCHANGE_NAMES = {
    "A": "NYSE American",
    "N": "New York Stock Exchange",
    "P": "NYSE ARCA",
    "Z": "BATS Global Markets",
    "V": "Investors' Exchange",
}


class HttpTextResponse(Protocol):
    """Minimal HTTP response contract needed by the Nasdaq Trader fetcher."""

    text: str

    def raise_for_status(self) -> None: ...


class HttpTextClient(Protocol):
    """Minimal async HTTP client contract needed by the Nasdaq Trader fetcher."""

    def get(self, url: str) -> Awaitable[HttpTextResponse]: ...


@dataclass(frozen=True)
class NasdaqTraderSymbolRecord:
    """One symbol-directory row normalized across Nasdaq and other-listed files."""

    source_file: NasdaqTraderSourceFile
    symbol: str
    security_name: str
    listing_exchange_code: str
    listing_exchange: str
    market_category: str = ""
    market_category_name: str = ""
    financial_status: str = ""
    financial_status_name: str = ""
    round_lot_size: int | None = None
    is_etf: bool | None = None
    is_test_issue: bool = False
    act_symbol: str = ""
    cqs_symbol: str = ""
    nasdaq_symbol: str = ""
    raw_fields: dict[str, str] = field(default_factory=dict)
    extra_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NasdaqTraderFileSnapshot:
    """Parsed representation of one Nasdaq Trader symbol-directory file."""

    source_file: NasdaqTraderSourceFile
    headers: tuple[str, ...]
    records: tuple[NasdaqTraderSymbolRecord, ...]
    file_creation_time_raw: str = ""
    file_creation_time: datetime | None = None


@dataclass(frozen=True)
class NasdaqTraderSymbolDirectory:
    """Paired Nasdaq-listed and other-listed symbol directory snapshots."""

    nasdaq_listed: NasdaqTraderFileSnapshot
    other_listed: NasdaqTraderFileSnapshot

    @property
    def records(self) -> tuple[NasdaqTraderSymbolRecord, ...]:
        """All directory records in deterministic source-file order."""
        return self.nasdaq_listed.records + self.other_listed.records


@dataclass(frozen=True)
class NasdaqTraderReconciliationResult:
    """Securities to upsert after reconciling a symbol directory snapshot."""

    securities: tuple[Security, ...]
    current_record_count: int
    active_count: int
    test_issue_count: int
    deactivated_missing_count: int
    nasdaq_listed_count: int
    other_listed_count: int


def parse_nasdaq_trader_symbol_directories(
    nasdaq_listed_text: str,
    other_listed_text: str,
) -> NasdaqTraderSymbolDirectory:
    """Parse the official Nasdaq Trader listed-security reference files."""
    nasdaq_listed = _parse_symbol_file(
        text=nasdaq_listed_text,
        source_file="nasdaqlisted",
        required_columns=_NASDAQ_REQUIRED_COLUMNS,
    )
    other_listed = _parse_symbol_file(
        text=other_listed_text,
        source_file="otherlisted",
        required_columns=_OTHER_LISTED_REQUIRED_COLUMNS,
    )
    return NasdaqTraderSymbolDirectory(
        nasdaq_listed=nasdaq_listed,
        other_listed=other_listed,
    )


def build_nasdaq_trader_reconciliation(
    directory: NasdaqTraderSymbolDirectory,
    *,
    existing_by_key: Mapping[tuple[str, str], Security],
    previously_sourced_by_key: Mapping[tuple[str, str], Security],
    observed_at: datetime | None = None,
) -> NasdaqTraderReconciliationResult:
    """Build security-master upserts without losing existing curated identifiers."""
    observed = _normalize_observed_at(observed_at)
    records_by_key = _dedupe_records_by_security_key(directory.records)
    securities: list[Security] = []

    for key, record in records_by_key.items():
        existing = existing_by_key.get(key) or previously_sourced_by_key.get(key)
        securities.append(_merge_record_into_security(record, existing, directory, observed))

    missing_source_records = [
        security
        for key, security in previously_sourced_by_key.items()
        if key not in records_by_key and security.is_active
    ]
    for security in missing_source_records:
        securities.append(_deactivate_missing_security(security, observed))

    return NasdaqTraderReconciliationResult(
        securities=tuple(securities),
        current_record_count=len(records_by_key),
        active_count=sum(1 for security in securities if security.is_active),
        test_issue_count=sum(1 for record in records_by_key.values() if record.is_test_issue),
        deactivated_missing_count=len(missing_source_records),
        nasdaq_listed_count=len(directory.nasdaq_listed.records),
        other_listed_count=len(directory.other_listed.records),
    )


async def fetch_nasdaq_trader_symbol_directories(
    *,
    nasdaq_listed_url: str = NASDAQ_LISTED_URL,
    other_listed_url: str = OTHER_LISTED_URL,
    timeout_seconds: float = 30.0,
) -> NasdaqTraderSymbolDirectory:
    """Fetch and parse the free Nasdaq Trader symbol-directory files."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
        nasdaq_response, other_response = await _fetch_symbol_directory_texts(
            client,
            nasdaq_listed_url,
            other_listed_url,
        )
    return parse_nasdaq_trader_symbol_directories(nasdaq_response, other_response)


async def _fetch_symbol_directory_texts(
    client: HttpTextClient,
    nasdaq_listed_url: str,
    other_listed_url: str,
) -> tuple[str, str]:
    import asyncio

    nasdaq_response, other_response = await asyncio.gather(
        client.get(nasdaq_listed_url),
        client.get(other_listed_url),
    )
    nasdaq_response.raise_for_status()
    other_response.raise_for_status()
    return nasdaq_response.text, other_response.text


def _parse_symbol_file(
    *,
    text: str,
    source_file: NasdaqTraderSourceFile,
    required_columns: tuple[str, ...],
) -> NasdaqTraderFileSnapshot:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{source_file} symbol directory is empty")

    headers = tuple(column.strip() for column in lines[0].split("|"))
    _require_columns(headers, required_columns, source_file)

    records: list[NasdaqTraderSymbolRecord] = []
    file_creation_time_raw = ""
    file_creation_time: datetime | None = None
    for line_number, line in enumerate(lines[1:], start=2):
        first_field = line.split("|", 1)[0].strip()
        if first_field.startswith("File Creation Time:"):
            file_creation_time_raw = first_field.split(":", 1)[1].strip()
            file_creation_time = _parse_file_creation_time(file_creation_time_raw)
            continue
        values = line.split("|")
        if len(values) < len(headers):
            raise ValueError(
                f"{source_file} line {line_number} has {len(values)} fields; "
                f"expected at least {len(headers)}"
            )
        row = {header: values[index].strip() for index, header in enumerate(headers)}
        records.append(_parse_record(row, source_file, headers, required_columns))

    return NasdaqTraderFileSnapshot(
        source_file=source_file,
        headers=headers,
        records=tuple(records),
        file_creation_time_raw=file_creation_time_raw,
        file_creation_time=file_creation_time,
    )


def _require_columns(
    headers: tuple[str, ...],
    required_columns: tuple[str, ...],
    source_file: NasdaqTraderSourceFile,
) -> None:
    missing = [column for column in required_columns if column not in headers]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{source_file} missing required columns: {joined}")


def _parse_record(
    row: dict[str, str],
    source_file: NasdaqTraderSourceFile,
    headers: tuple[str, ...],
    required_columns: tuple[str, ...],
) -> NasdaqTraderSymbolRecord:
    if source_file == "nasdaqlisted":
        return _parse_nasdaq_listed_record(row, headers, required_columns)
    return _parse_other_listed_record(row, headers, required_columns)


def _parse_nasdaq_listed_record(
    row: dict[str, str],
    headers: tuple[str, ...],
    required_columns: tuple[str, ...],
) -> NasdaqTraderSymbolRecord:
    symbol = _required_symbol(row["Symbol"], "Symbol")
    market_category = row["Market Category"].strip().upper()
    financial_status = row["Financial Status"].strip().upper()
    return NasdaqTraderSymbolRecord(
        source_file="nasdaqlisted",
        symbol=symbol,
        security_name=row["Security Name"].strip(),
        listing_exchange_code="Q",
        listing_exchange="Nasdaq",
        market_category=market_category,
        market_category_name=_MARKET_CATEGORY_NAMES.get(market_category, market_category),
        financial_status=financial_status,
        financial_status_name=_FINANCIAL_STATUS_NAMES.get(financial_status, financial_status),
        round_lot_size=_parse_int(row["Round Lot Size"]),
        is_etf=_parse_bool(row.get("ETF", "")),
        is_test_issue=_parse_required_bool(row["Test Issue"], "Test Issue"),
        nasdaq_symbol=symbol,
        raw_fields=row,
        extra_fields=_extra_fields(row, headers, required_columns),
    )


def _parse_other_listed_record(
    row: dict[str, str],
    headers: tuple[str, ...],
    required_columns: tuple[str, ...],
) -> NasdaqTraderSymbolRecord:
    nasdaq_symbol = _clean_symbol(row["NASDAQ Symbol"])
    act_symbol = _required_symbol(row["ACT Symbol"], "ACT Symbol")
    symbol = nasdaq_symbol or act_symbol
    exchange_code = row["Exchange"].strip().upper()
    return NasdaqTraderSymbolRecord(
        source_file="otherlisted",
        symbol=symbol,
        security_name=row["Security Name"].strip(),
        listing_exchange_code=exchange_code,
        listing_exchange=_OTHER_EXCHANGE_NAMES.get(exchange_code, exchange_code),
        round_lot_size=_parse_int(row["Round Lot Size"]),
        is_etf=_parse_bool(row["ETF"]),
        is_test_issue=_parse_required_bool(row["Test Issue"], "Test Issue"),
        act_symbol=act_symbol,
        cqs_symbol=_clean_symbol(row["CQS Symbol"]),
        nasdaq_symbol=nasdaq_symbol,
        raw_fields=row,
        extra_fields=_extra_fields(row, headers, required_columns),
    )


def _extra_fields(
    row: dict[str, str],
    headers: tuple[str, ...],
    required_columns: tuple[str, ...],
) -> dict[str, str]:
    optional_known = {"ETF", "NextShares"}
    known = set(required_columns) | optional_known
    return {header: row[header] for header in headers if header not in known}


def _parse_file_creation_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%m%d%Y%H:%M").replace(tzinfo=UTC)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


def _parse_bool(value: str) -> bool | None:
    stripped = value.strip().upper()
    if not stripped:
        return None
    if stripped == "Y":
        return True
    if stripped == "N":
        return False
    raise ValueError(f"expected Y/N value, got {value!r}")


def _parse_required_bool(value: str, field_name: str) -> bool:
    parsed = _parse_bool(value)
    if parsed is None:
        raise ValueError(f"{field_name} must be Y or N")
    return parsed


def _required_symbol(value: str, field_name: str) -> str:
    symbol = _clean_symbol(value)
    if not symbol:
        raise ValueError(f"{field_name} must be non-empty")
    return symbol


def _clean_symbol(value: str) -> str:
    return value.strip().upper()


def _dedupe_records_by_security_key(
    records: Iterable[NasdaqTraderSymbolRecord],
) -> dict[tuple[str, str], NasdaqTraderSymbolRecord]:
    by_key: dict[tuple[str, str], NasdaqTraderSymbolRecord] = {}
    for record in records:
        by_key.setdefault(_security_key(record.symbol), record)
    return by_key


def _security_key(symbol: str) -> tuple[str, str]:
    return (symbol, SECURITY_MASTER_US_EXCHANGE)


def _merge_record_into_security(
    record: NasdaqTraderSymbolRecord,
    existing: Security | None,
    directory: NasdaqTraderSymbolDirectory,
    observed_at: datetime,
) -> Security:
    metadata = _record_metadata(record, directory, observed_at)
    is_active = not record.is_test_issue
    previous_name = existing.name if existing and existing.name != record.security_name else ""
    aliases = _merge_unique(
        existing.aliases if existing else [],
        [previous_name, record.act_symbol, record.cqs_symbol, record.nasdaq_symbol],
    )
    external_identifiers = dict(existing.external_identifiers if existing else {})
    external_identifiers[NASDAQ_TRADER_EXTERNAL_KEY] = metadata

    return Security(
        ticker=record.symbol,
        exchange=SECURITY_MASTER_US_EXCHANGE,
        name=record.security_name,
        aliases=aliases,
        sector=existing.sector if existing else "",
        country=existing.country if existing else "US",
        currency=existing.currency if existing else "USD",
        figi=existing.figi if existing else None,
        sec_cik=existing.sec_cik if existing else None,
        issuer_name=existing.issuer_name if existing else record.security_name,
        former_names=list(existing.former_names if existing else []),
        external_identifiers=external_identifiers,
        identifier_lineage=_replace_nasdaq_trader_lineage(
            existing.identifier_lineage if existing else [],
            _record_lineage(record, metadata, observed_at),
        ),
        is_active=is_active,
        created_at=existing.created_at if existing else None,
        updated_at=existing.updated_at if existing else None,
    )


def _record_metadata(
    record: NasdaqTraderSymbolRecord,
    directory: NasdaqTraderSymbolDirectory,
    observed_at: datetime,
) -> dict[str, Any]:
    snapshot = (
        directory.nasdaq_listed if record.source_file == "nasdaqlisted" else directory.other_listed
    )
    return {
        "source_file": record.source_file,
        "source_url": _source_url(record.source_file),
        "status": "test_issue" if record.is_test_issue else "active",
        "symbol": record.symbol,
        "act_symbol": record.act_symbol,
        "cqs_symbol": record.cqs_symbol,
        "nasdaq_symbol": record.nasdaq_symbol,
        "listing_exchange_code": record.listing_exchange_code,
        "listing_exchange": record.listing_exchange,
        "market_category": record.market_category,
        "market_category_name": record.market_category_name,
        "financial_status": record.financial_status,
        "financial_status_name": record.financial_status_name,
        "round_lot_size": record.round_lot_size,
        "is_etf": record.is_etf,
        "is_test_issue": record.is_test_issue,
        "last_seen_at": observed_at.isoformat(),
        "file_creation_time": (
            snapshot.file_creation_time.isoformat() if snapshot.file_creation_time else None
        ),
        "file_creation_time_raw": snapshot.file_creation_time_raw,
        "raw_fields": dict(record.raw_fields),
        "extra_fields": dict(record.extra_fields),
    }


def _source_url(source_file: NasdaqTraderSourceFile) -> str:
    if source_file == "nasdaqlisted":
        return NASDAQ_LISTED_URL
    return OTHER_LISTED_URL


def _record_lineage(
    record: NasdaqTraderSymbolRecord,
    metadata: dict[str, Any],
    observed_at: datetime,
) -> SecurityIdentifierLineage:
    return SecurityIdentifierLineage(
        identifier_type="ticker",
        value=record.symbol,
        source=NASDAQ_TRADER_LINEAGE_SOURCE,
        observed_at=observed_at.isoformat(),
        confidence=1.0,
        metadata={
            "source_file": record.source_file,
            "listing_exchange": metadata["listing_exchange"],
            "is_etf": metadata["is_etf"],
            "is_test_issue": metadata["is_test_issue"],
        },
    )


def _replace_nasdaq_trader_lineage(
    existing: Iterable[SecurityIdentifierLineage],
    current: SecurityIdentifierLineage,
) -> list[SecurityIdentifierLineage]:
    return [
        record
        for record in existing
        if not (
            record.source == NASDAQ_TRADER_LINEAGE_SOURCE
            and record.identifier_type == current.identifier_type
        )
    ] + [current]


def _deactivate_missing_security(security: Security, observed_at: datetime) -> Security:
    external_identifiers = dict(security.external_identifiers)
    metadata = dict(external_identifiers.get(NASDAQ_TRADER_EXTERNAL_KEY) or {})
    metadata["status"] = "missing_from_latest"
    metadata["last_missing_at"] = observed_at.isoformat()
    external_identifiers[NASDAQ_TRADER_EXTERNAL_KEY] = metadata
    return replace(
        security,
        external_identifiers=external_identifiers,
        identifier_lineage=[
            _expire_nasdaq_trader_lineage(record, observed_at)
            for record in security.identifier_lineage
        ],
        is_active=False,
    )


def _expire_nasdaq_trader_lineage(
    record: SecurityIdentifierLineage,
    observed_at: datetime,
) -> SecurityIdentifierLineage:
    if record.source != NASDAQ_TRADER_LINEAGE_SOURCE or record.valid_to is not None:
        return record
    return replace(record, valid_to=observed_at.isoformat())


def _normalize_observed_at(observed_at: datetime | None) -> datetime:
    observed = observed_at or datetime.now(UTC)
    if observed.tzinfo is None:
        return observed.replace(tzinfo=UTC)
    return observed.astimezone(UTC)


def _merge_unique(existing: Iterable[str], additions: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *additions]:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged
