"""Ingestion orchestration for market-structure datasource files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from src.market_structure.models import MarketStructureEvent, MarketStructureSourceFile
from src.market_structure.parsers import (
    apply_market_structure_signals,
    apply_security_mappings,
    parse_finra_short_volume_file,
    parse_sec_fails_to_deliver_file,
)
from src.market_structure.repository import MarketStructureEventRepository
from src.security_master.repository import SecurityMasterRepository
from src.security_master.schemas import Security
from src.storage.database import Database


@dataclass(frozen=True)
class MarketStructureIngestionResult:
    """Summary of one market-structure ingestion run."""

    total_events: int
    finra_short_volume_count: int
    sec_fails_to_deliver_count: int
    upserted_count: int
    unresolved_symbol_count: int


class _SecurityRepository(Protocol):
    async def get_by_keys(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> Mapping[tuple[str, str], Security]: ...


class _EventRepository(Protocol):
    async def upsert_events(
        self,
        events: list[MarketStructureEvent],
    ) -> list[MarketStructureEvent]: ...


class MarketStructureIngestionService:
    """Parse, map, signal, and persist market-structure source files."""

    def __init__(
        self,
        database: Database | None = None,
        *,
        security_repository: _SecurityRepository | None = None,
        event_repository: _EventRepository | None = None,
    ) -> None:
        if database is None and (security_repository is None or event_repository is None):
            raise ValueError("database or both repositories must be provided")
        self._security_repository: _SecurityRepository = (
            security_repository or SecurityMasterRepository(database)
        )
        self._event_repository: _EventRepository = (
            event_repository or MarketStructureEventRepository(database)
        )

    async def ingest_source_files(
        self,
        *,
        finra_short_volume_files: list[MarketStructureSourceFile] | None = None,
        sec_fails_to_deliver_files: list[MarketStructureSourceFile] | None = None,
    ) -> MarketStructureIngestionResult:
        """Ingest local or fetched FINRA short-volume and SEC FTD files."""
        finra_files = finra_short_volume_files or []
        sec_files = sec_fails_to_deliver_files or []
        events = _parse_files(finra_files, sec_files)
        securities_by_symbol = await self._load_securities_by_symbol(events)
        mapped_events = apply_security_mappings(
            events,
            securities_by_symbol=securities_by_symbol,
        )
        signaled_events = apply_market_structure_signals(mapped_events)
        persisted = await self._event_repository.upsert_events(signaled_events)
        unresolved_symbols = {
            event.symbol
            for event in signaled_events
            if event.metadata.get("mapping_status") == "unresolved"
        }
        return MarketStructureIngestionResult(
            total_events=len(signaled_events),
            finra_short_volume_count=sum(
                1 for event in signaled_events if event.event_type == "finra_short_volume"
            ),
            sec_fails_to_deliver_count=sum(
                1 for event in signaled_events if event.event_type == "sec_fail_to_deliver"
            ),
            upserted_count=len(persisted),
            unresolved_symbol_count=len(unresolved_symbols),
        )

    async def _load_securities_by_symbol(
        self,
        events: list[MarketStructureEvent],
    ) -> dict[str, Security]:
        keys = sorted({(event.symbol, "US") for event in events if event.symbol})
        if not keys:
            return {}
        securities_by_key = await self._security_repository.get_by_keys(keys)
        return {
            security.ticker.upper(): security
            for security in securities_by_key.values()
        }


def _parse_files(
    finra_files: list[MarketStructureSourceFile],
    sec_files: list[MarketStructureSourceFile],
) -> list[MarketStructureEvent]:
    events: list[MarketStructureEvent] = []
    for source in finra_files:
        events.extend(parse_finra_short_volume_file(source))
    for source in sec_files:
        events.extend(parse_sec_fails_to_deliver_file(source))
    return events
