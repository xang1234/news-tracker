"""Tests for persisted research innovation signals."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock

import pytest

from src.innovation.research import ResearchSignal
from src.innovation.research_repository import ResearchSignalRepository


@pytest.fixture
def mock_database() -> AsyncMock:
    db = AsyncMock()
    db.fetchrow = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    return db


def _signal() -> ResearchSignal:
    return ResearchSignal(
        source="openalex",
        record_id="https://openalex.org/W1",
        published_date=date(2026, 5, 20),
        title="AI accelerator interconnect research",
        issuer_concept_id="issuer_nvda",
        security_concept_id="security_nvda",
        theme_id="theme_ai_accelerators",
        confidence=0.78,
        confidence_reasons=["institution_alias:0.88", "topic:ai accelerators"],
        source_lineage={"institution_alias": "NVIDIA Research"},
        metadata={"alias_review_status": "accepted"},
        url="https://example.org/research",
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )


def _row() -> dict:
    return {
        "source": "openalex",
        "record_id": "https://openalex.org/W1",
        "published_date": date(2026, 5, 20),
        "title": "AI accelerator interconnect research",
        "issuer_concept_id": "issuer_nvda",
        "security_concept_id": "security_nvda",
        "theme_id": "theme_ai_accelerators",
        "confidence": 0.78,
        "confidence_reasons": ["institution_alias:0.88", "topic:ai accelerators"],
        "source_lineage": {"institution_alias": "NVIDIA Research"},
        "metadata": {"alias_review_status": "accepted"},
        "url": "https://example.org/research",
        "fetched_at": datetime(2026, 6, 1, tzinfo=UTC),
        "created_at": datetime(2026, 6, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 1, tzinfo=UTC),
    }


@pytest.mark.asyncio
async def test_upsert_signals_persists_lineage_and_confidence_json(
    mock_database: AsyncMock,
) -> None:
    mock_database.fetchrow.return_value = _row()
    repository = ResearchSignalRepository(mock_database)

    written = await repository.upsert_signals([_signal()])

    sql, *params = mock_database.fetchrow.call_args.args
    assert "INSERT INTO innovation_research_signals" in sql
    assert "ON CONFLICT" in sql
    assert params[:10] == [
        "openalex",
        "https://openalex.org/W1",
        date(2026, 5, 20),
        "AI accelerator interconnect research",
        "issuer_nvda",
        "security_nvda",
        "theme_ai_accelerators",
        0.78,
        "https://example.org/research",
        datetime(2026, 6, 1, tzinfo=UTC),
    ]
    assert json.loads(params[10]) == ["institution_alias:0.88", "topic:ai accelerators"]
    assert json.loads(params[11]) == {"institution_alias": "NVIDIA Research"}
    assert json.loads(params[12]) == {"alias_review_status": "accepted"}
    assert written == [_signal()]


@pytest.mark.asyncio
async def test_list_signals_filters_by_theme_issuer_and_source(mock_database: AsyncMock) -> None:
    mock_database.fetch.return_value = [_row()]
    repository = ResearchSignalRepository(mock_database)

    rows = await repository.list_signals(
        source="openalex",
        theme_id="theme_ai_accelerators",
        issuer_concept_id="issuer_nvda",
        limit=50,
    )

    sql, *params = mock_database.fetch.call_args.args
    assert "source = $1" in sql
    assert "theme_id = $2" in sql
    assert "issuer_concept_id = $3" in sql
    assert "LIMIT $4" in sql
    assert params == ["openalex", "theme_ai_accelerators", "issuer_nvda", 50]
    assert rows == [_signal()]
