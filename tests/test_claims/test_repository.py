"""Tests for ClaimRepository numeric-fact persistence and comparison queries.

Uses an AsyncMock Database (the project convention for repository tests) so
no live Postgres is required — verifies Python-side mapping and that the
generated SQL references the new typed columns.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from src.claims.repository import ClaimRepository
from src.claims.schemas import EvidenceClaim, make_claim_id, make_claim_key


def _make_claim(**overrides: Any) -> EvidenceClaim:
    key = make_claim_key("narrative", "doc_1", "TSMC", "revises_guidance", "capex")
    base: dict[str, Any] = {
        "claim_id": make_claim_id(key),
        "claim_key": key,
        "lane": "narrative",
        "source_id": "doc_1",
        "predicate": "revises_guidance",
        "subject_text": "TSMC",
        "contract_version": "v1",
    }
    base.update(overrides)
    return EvidenceClaim(**base)


def _full_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "claim_id": "claim_x",
        "claim_key": "clk_x",
        "lane": "narrative",
        "run_id": None,
        "source_id": "doc_1",
        "source_type": "document",
        "source_span_start": None,
        "source_span_end": None,
        "source_text": None,
        "subject_text": "TSMC",
        "subject_concept_id": "concept_tsmc",
        "predicate": "revises_guidance",
        "object_text": "capex",
        "object_concept_id": None,
        "confidence": 0.8,
        "extraction_method": "rule",
        "metric": "capex",
        "numeric_value": 42_000_000_000.0,
        "unit": "USD",
        "period": "2026",
        "modality": "guided",
        "claim_valid_from": None,
        "claim_valid_to": None,
        "source_published_at": None,
        "contract_version": "v1",
        "status": "active",
        "metadata": {},
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


class TestUpsertNumericFields:
    async def test_round_trips_typed_numeric_fields(self) -> None:
        db = AsyncMock()
        db.fetchrow = AsyncMock(return_value=_full_row())
        repo = ClaimRepository(db)

        claim = _make_claim(
            metric="capex",
            numeric_value=42_000_000_000.0,
            unit="USD",
            period="2026",
            modality="guided",
        )
        result = await repo.upsert_claim(claim)

        assert result.metric == "capex"
        assert result.numeric_value == 42_000_000_000.0
        assert result.unit == "USD"
        assert result.period == "2026"
        assert result.modality == "guided"

    async def test_insert_sql_references_new_columns(self) -> None:
        db = AsyncMock()
        db.fetchrow = AsyncMock(return_value=_full_row())
        repo = ClaimRepository(db)

        await repo.upsert_claim(
            _make_claim(
                metric="capex",
                numeric_value=42e9,
                unit="USD",
                period="2026",
                modality="guided",
            )
        )

        sql = db.fetchrow.call_args.args[0]
        for column in ("metric", "numeric_value", "unit", "period", "modality"):
            assert column in sql
        # The typed values must be passed as bind params.
        passed = db.fetchrow.call_args.args
        assert "capex" in passed
        assert 42e9 in passed
