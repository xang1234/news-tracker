"""Tests for concept repository alias resolution."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.security_master.concept_repository import ConceptRepository
from src.security_master.concept_schemas import ConceptAlias

NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _concept_alias_row(
    *,
    concept_id: str,
    canonical_name: str,
    alias: str,
    confidence: float,
    review_status: str,
) -> dict:
    return {
        "concept_id": concept_id,
        "concept_type": "issuer",
        "canonical_name": canonical_name,
        "description": "",
        "concept_metadata": {"sector": "semiconductors"},
        "is_active": True,
        "concept_created_at": NOW,
        "updated_at": NOW,
        "alias": alias,
        "alias_type": "lab",
        "is_primary": False,
        "confidence": confidence,
        "source_attribution": "curated_innovation_aliases",
        "review_status": review_status,
        "review_note": "ambiguous lab name",
        "alias_metadata": {"source_contexts": ["patents", "research"]},
        "created_at": NOW,
    }


@pytest.fixture
def mock_database() -> AsyncMock:
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="UPDATE 0")
    return db


@pytest.mark.asyncio
async def test_add_alias_persists_confidence_and_review_metadata(
    mock_database: AsyncMock,
) -> None:
    mock_database.fetchrow.return_value = {
        "alias": "NVIDIA Research",
        "concept_id": "concept_issuer_nvda",
        "alias_type": "lab",
        "is_primary": False,
        "confidence": 0.72,
        "source_attribution": "curated_innovation_aliases",
        "review_status": "accepted",
        "review_note": "",
        "metadata": {"source_contexts": ["patents", "research"]},
        "created_at": NOW,
    }
    repo = ConceptRepository(mock_database)

    result = await repo.add_alias(
        ConceptAlias(
            alias="NVIDIA Research",
            concept_id="concept_issuer_nvda",
            alias_type="lab",
            confidence=0.72,
            source_attribution="curated_innovation_aliases",
            metadata={"source_contexts": ["patents", "research"]},
        )
    )

    args = mock_database.fetchrow.call_args[0]
    sql = args[0]
    assert "confidence" in sql
    assert "review_status" in sql
    assert args[5] == 0.72
    assert args[6] == "curated_innovation_aliases"
    assert args[7] == "accepted"
    assert result.confidence == 0.72
    assert result.metadata["source_contexts"] == ["patents", "research"]


@pytest.mark.asyncio
async def test_resolve_alias_candidates_retains_ambiguous_matches(
    mock_database: AsyncMock,
) -> None:
    mock_database.fetch.return_value = [
        _concept_alias_row(
            concept_id="concept_issuer_nvda",
            canonical_name="NVIDIA Corporation",
            alias="NVIDIA Research",
            confidence=0.72,
            review_status="accepted",
        ),
        _concept_alias_row(
            concept_id="concept_issuer_uni",
            canonical_name="University Lab With Similar Name",
            alias="NVIDIA Research",
            confidence=0.41,
            review_status="needs_review",
        ),
    ]
    repo = ConceptRepository(mock_database)

    candidates = await repo.resolve_alias_candidates("NVIDIA Research", limit=5)

    args = mock_database.fetch.call_args[0]
    sql = args[0]
    assert "JOIN concept_aliases" in sql
    assert "ca.review_status <> 'rejected'" in sql
    assert "ca.confidence DESC" in sql
    assert args[1] == "NVIDIA Research"
    assert args[2] == 5
    assert len(candidates) == 2
    assert candidates[0].concept.canonical_name == "NVIDIA Corporation"
    assert candidates[0].concept.created_at == NOW
    assert candidates[0].alias.confidence == 0.72
    assert candidates[1].requires_review is True


@pytest.mark.asyncio
async def test_resolve_alias_uses_highest_ranked_candidate(
    mock_database: AsyncMock,
) -> None:
    mock_database.fetch.return_value = [
        _concept_alias_row(
            concept_id="concept_issuer_nvda",
            canonical_name="NVIDIA Corporation",
            alias="NVIDIA Research",
            confidence=0.72,
            review_status="accepted",
        )
    ]
    repo = ConceptRepository(mock_database)

    concept = await repo.resolve_alias("NVIDIA Research")

    assert concept is not None
    assert concept.concept_id == "concept_issuer_nvda"
