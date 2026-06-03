"""Tests for persisted innovation patent signal records."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock

import pytest

from src.innovation.patents import PatentSignal
from src.innovation.repository import PatentSignalRepository


@pytest.fixture
def mock_database() -> AsyncMock:
    db = AsyncMock()
    db.fetchrow = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    return db


def _signal() -> PatentSignal:
    return PatentSignal(
        patent_id="US999",
        patent_family_id="fam-999",
        event_type="grant",
        event_date=date(2025, 8, 1),
        title="Neural network chiplet package",
        issuer_concept_id="issuer_amd",
        security_concept_id="security_amd",
        theme_id="theme_ai_accelerators",
        confidence=0.82,
        confidence_reasons=["assignee_alias:0.92", "classification:G06N"],
        source_lineage={
            "source": "uspto_odp_patentsview_transition",
            "assignee_alias": "Advanced Micro Devices",
            "classes": ["G06N 3/04"],
        },
        metadata={"requires_review": False},
        source_url="https://api.uspto.gov/api/v1/patent/applications/search",
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )


def _row() -> dict:
    return {
        "patent_id": "US999",
        "patent_family_id": "fam-999",
        "event_type": "grant",
        "event_date": date(2025, 8, 1),
        "title": "Neural network chiplet package",
        "issuer_concept_id": "issuer_amd",
        "security_concept_id": "security_amd",
        "theme_id": "theme_ai_accelerators",
        "confidence": 0.82,
        "confidence_reasons": ["assignee_alias:0.92", "classification:G06N"],
        "source_lineage": {
            "source": "uspto_odp_patentsview_transition",
            "assignee_alias": "Advanced Micro Devices",
            "classes": ["G06N 3/04"],
        },
        "metadata": {"requires_review": False},
        "source_url": "https://api.uspto.gov/api/v1/patent/applications/search",
        "fetched_at": datetime(2026, 6, 1, tzinfo=UTC),
        "created_at": datetime(2026, 6, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 1, tzinfo=UTC),
    }


@pytest.mark.asyncio
async def test_upsert_signals_persists_confidence_and_lineage_json(
    mock_database: AsyncMock,
) -> None:
    mock_database.fetchrow.return_value = _row()
    repository = PatentSignalRepository(mock_database)

    written = await repository.upsert_signals([_signal()])

    sql, *params = mock_database.fetchrow.call_args.args
    assert "INSERT INTO innovation_patent_signals" in sql
    assert "ON CONFLICT" in sql
    assert params[:10] == [
        "US999",
        "fam-999",
        "grant",
        date(2025, 8, 1),
        "Neural network chiplet package",
        "issuer_amd",
        "security_amd",
        "theme_ai_accelerators",
        0.82,
        "https://api.uspto.gov/api/v1/patent/applications/search",
    ]
    assert json.loads(params[10]) == ["assignee_alias:0.92", "classification:G06N"]
    assert json.loads(params[11])["assignee_alias"] == "Advanced Micro Devices"
    assert json.loads(params[12]) == {"requires_review": False}
    assert written == [_signal()]


@pytest.mark.asyncio
async def test_list_signals_filters_by_theme_and_issuer(mock_database: AsyncMock) -> None:
    mock_database.fetch.return_value = [_row()]
    repository = PatentSignalRepository(mock_database)

    rows = await repository.list_signals(
        theme_id="theme_ai_accelerators",
        issuer_concept_id="issuer_amd",
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        limit=25,
    )

    sql, *params = mock_database.fetch.call_args.args
    assert "theme_id = $1" in sql
    assert "issuer_concept_id = $2" in sql
    assert "event_date >= $3" in sql
    assert "event_date <= $4" in sql
    assert "LIMIT $5" in sql
    assert params == [
        "theme_ai_accelerators",
        "issuer_amd",
        date(2025, 1, 1),
        date(2025, 12, 31),
        25,
    ]
    assert rows == [_signal()]
