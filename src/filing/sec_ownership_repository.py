"""Persistence for SEC ownership filing events."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.filing.sec_ownership_models import SECOwnershipEvent
from src.security_master.schemas import normalize_sec_cik
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
    return dict(value) if isinstance(value, dict) else {}


def _row_to_event(row: Any) -> SECOwnershipEvent:
    return SECOwnershipEvent(
        event_id=row["event_id"],
        event_type=row["event_type"],
        accession_number=row["accession_number"],
        filing_type=row["filing_type"],
        filed_date=row["filed_date"],
        issuer_cik=row["issuer_cik"],
        issuer_name=row["issuer_name"],
        issuer_ticker=row["issuer_ticker"],
        filer_cik=row["filer_cik"],
        filer_name=row["filer_name"],
        security_title=row["security_title"],
        transaction_code=row["transaction_code"],
        transaction_date=row["transaction_date"],
        transaction_shares=row["transaction_shares"],
        transaction_price_per_share=row["transaction_price_per_share"],
        transaction_acquired_disposed_code=row["transaction_acquired_disposed_code"],
        shares_owned_following=row["shares_owned_following"],
        derivative_underlying_shares=row["derivative_underlying_shares"],
        ownership_percent=row["ownership_percent"],
        position_cusip=row["position_cusip"],
        position_shares=row["position_shares"],
        position_value_usd=row["position_value_usd"],
        previous_position_shares=row["previous_position_shares"],
        position_delta_shares=row["position_delta_shares"],
        is_amendment=row["is_amendment"],
        available_at=row["available_at"],
        fetched_at=row["fetched_at"],
        source_url=row["source_url"],
        metadata=_parse_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class SECOwnershipEventRepository:
    """Repository for idempotent SEC ownership event persistence."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_event(self, event: SECOwnershipEvent) -> SECOwnershipEvent:
        row = await self._db.fetchrow(
            """
            INSERT INTO sec_ownership_events (
                event_id, event_type, accession_number, filing_type, filed_date,
                issuer_cik, issuer_name, issuer_ticker, filer_cik, filer_name,
                security_title, transaction_code, transaction_date, transaction_shares,
                transaction_price_per_share, transaction_acquired_disposed_code,
                shares_owned_following, derivative_underlying_shares,
                ownership_percent, position_cusip, position_shares, position_value_usd,
                previous_position_shares, position_delta_shares, is_amendment,
                available_at, fetched_at, source_url, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28, $29
            )
            ON CONFLICT (event_id) DO UPDATE SET
                event_type = $2,
                accession_number = $3,
                filing_type = $4,
                filed_date = $5,
                issuer_cik = $6,
                issuer_name = $7,
                issuer_ticker = $8,
                filer_cik = $9,
                filer_name = $10,
                security_title = $11,
                transaction_code = $12,
                transaction_date = $13,
                transaction_shares = $14,
                transaction_price_per_share = $15,
                transaction_acquired_disposed_code = $16,
                shares_owned_following = $17,
                derivative_underlying_shares = $18,
                ownership_percent = $19,
                position_cusip = $20,
                position_shares = $21,
                position_value_usd = $22,
                previous_position_shares = $23,
                position_delta_shares = $24,
                is_amendment = $25,
                available_at = $26,
                fetched_at = $27,
                source_url = $28,
                metadata = $29,
                updated_at = NOW()
            RETURNING *
            """,
            event.event_id,
            event.event_type,
            event.accession_number,
            event.filing_type,
            event.filed_date,
            event.issuer_cik,
            event.issuer_name,
            event.issuer_ticker,
            event.filer_cik,
            event.filer_name,
            event.security_title,
            event.transaction_code,
            event.transaction_date,
            event.transaction_shares,
            event.transaction_price_per_share,
            event.transaction_acquired_disposed_code,
            event.shares_owned_following,
            event.derivative_underlying_shares,
            event.ownership_percent,
            event.position_cusip,
            event.position_shares,
            event.position_value_usd,
            event.previous_position_shares,
            event.position_delta_shares,
            event.is_amendment,
            event.available_at,
            event.fetched_at,
            event.source_url,
            json.dumps(event.metadata),
        )
        return _row_to_event(row)

    async def upsert_events(self, events: list[SECOwnershipEvent]) -> list[SECOwnershipEvent]:
        return [await self.upsert_event(event) for event in events]

    async def list_events(
        self,
        *,
        issuer_cik: str | None = None,
        filer_cik: str | None = None,
        as_of: datetime | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[SECOwnershipEvent]:
        params: list[Any] = []
        conditions: list[str] = []
        if issuer_cik is not None:
            normalized = normalize_sec_cik(issuer_cik)
            if normalized is None:
                raise ValueError("issuer_cik must be a non-empty SEC CIK")
            params.append(normalized)
            conditions.append(f"issuer_cik = ${len(params)}")
        if filer_cik is not None:
            normalized = normalize_sec_cik(filer_cik)
            if normalized is None:
                raise ValueError("filer_cik must be a non-empty SEC CIK")
            params.append(normalized)
            conditions.append(f"filer_cik = ${len(params)}")
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
            FROM sec_ownership_events
            {where_clause}
            ORDER BY available_at DESC, filed_date DESC, event_id DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_event(row) for row in rows]
