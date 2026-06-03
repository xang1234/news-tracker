"""Persistence for point-in-time market-structure events."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.market_structure.models import MarketStructureEvent
from src.storage.database import Database


def _parse_json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _row_to_event(row: Any) -> MarketStructureEvent:
    return MarketStructureEvent(
        event_id=row["event_id"],
        event_type=row["event_type"],
        source_name=row["source_name"],
        source_url=row["source_url"],
        source_date=row["source_date"],
        trade_date=row["trade_date"],
        settlement_date=row["settlement_date"],
        symbol=row["symbol"],
        security_ticker=row["security_ticker"],
        security_exchange=row["security_exchange"],
        issuer_cik=row["issuer_cik"],
        issuer_name=row["issuer_name"],
        cusip=row["cusip"],
        market_code=row["market_code"],
        market_name=row["market_name"],
        short_volume=row["short_volume"],
        short_exempt_volume=row["short_exempt_volume"],
        total_volume=row["total_volume"],
        short_volume_ratio=row["short_volume_ratio"],
        short_exempt_ratio=row["short_exempt_ratio"],
        fail_quantity=row["fail_quantity"],
        fail_price=row["fail_price"],
        fail_notional=row["fail_notional"],
        signal_type=row["signal_type"],
        anomaly_level=row["anomaly_level"],
        persistence_count=row["persistence_count"],
        available_at=row["available_at"],
        fetched_at=row["fetched_at"],
        metadata=_parse_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class MarketStructureEventRepository:
    """Repository for idempotent market-structure event persistence."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_event(self, event: MarketStructureEvent) -> MarketStructureEvent:
        row = await self._db.fetchrow(
            """
            INSERT INTO market_structure_events (
                event_id, event_type, source_name, source_url, source_date,
                trade_date, settlement_date, symbol, security_ticker,
                security_exchange, issuer_cik, issuer_name, cusip, market_code,
                market_name, short_volume, short_exempt_volume, total_volume,
                short_volume_ratio, short_exempt_ratio, fail_quantity, fail_price,
                fail_notional, signal_type, anomaly_level, persistence_count,
                available_at, fetched_at, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28, $29
            )
            ON CONFLICT (event_id) DO UPDATE SET
                event_type = $2,
                source_name = $3,
                source_url = $4,
                source_date = $5,
                trade_date = $6,
                settlement_date = $7,
                symbol = $8,
                security_ticker = $9,
                security_exchange = $10,
                issuer_cik = $11,
                issuer_name = $12,
                cusip = $13,
                market_code = $14,
                market_name = $15,
                short_volume = $16,
                short_exempt_volume = $17,
                total_volume = $18,
                short_volume_ratio = $19,
                short_exempt_ratio = $20,
                fail_quantity = $21,
                fail_price = $22,
                fail_notional = $23,
                signal_type = $24,
                anomaly_level = $25,
                persistence_count = $26,
                available_at = $27,
                fetched_at = $28,
                metadata = $29,
                updated_at = NOW()
            RETURNING *
            """,
            event.event_id,
            event.event_type,
            event.source_name,
            event.source_url,
            event.source_date,
            event.trade_date,
            event.settlement_date,
            event.symbol,
            event.security_ticker,
            event.security_exchange,
            event.issuer_cik,
            event.issuer_name,
            event.cusip,
            event.market_code,
            event.market_name,
            event.short_volume,
            event.short_exempt_volume,
            event.total_volume,
            event.short_volume_ratio,
            event.short_exempt_ratio,
            event.fail_quantity,
            event.fail_price,
            event.fail_notional,
            event.signal_type,
            event.anomaly_level,
            event.persistence_count,
            event.available_at,
            event.fetched_at,
            json.dumps(event.metadata),
        )
        return _row_to_event(row)

    async def upsert_events(
        self,
        events: list[MarketStructureEvent],
    ) -> list[MarketStructureEvent]:
        return [await self.upsert_event(event) for event in events]

    async def list_events(
        self,
        *,
        symbol: str | None = None,
        cusip: str | None = None,
        as_of: datetime | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[MarketStructureEvent]:
        params: list[Any] = []
        conditions: list[str] = []
        if symbol is not None:
            params.append(symbol.upper())
            conditions.append(f"symbol = ${len(params)}")
        if cusip is not None:
            params.append(cusip.upper())
            conditions.append(f"cusip = ${len(params)}")
        if as_of is not None:
            params.append(as_of)
            conditions.append(f"available_at <= ${len(params)}")
        if event_type is not None:
            params.append(event_type)
            conditions.append(f"event_type = ${len(params)}")

        params.append(limit)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT *
            FROM market_structure_events
            {where_clause}
            ORDER BY available_at DESC, source_date DESC, event_id DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_event(row) for row in rows]
