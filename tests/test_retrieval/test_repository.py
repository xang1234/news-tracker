"""Tests for ClaimRetrievalRepository.

AsyncMock Database (project convention for repository tests) — verifies the
generated SQL, bind params, and Python-side row mapping without a live
Postgres. The vector is passed as a pgvector string literal, mirroring the
documents embedding-store pattern.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from src.retrieval.repository import ClaimRetrievalRepository
from src.retrieval.schemas import ClaimRetrievalFilter


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
        "predicate": "supplies_to",
        "object_text": "NVIDIA",
        "object_concept_id": None,
        "confidence": 0.8,
        "extraction_method": "rule",
        "metric": None,
        "numeric_value": None,
        "unit": None,
        "period": None,
        "modality": None,
        "claim_valid_from": None,
        "claim_valid_to": None,
        "source_published_at": None,
        "contract_version": "v1",
        "status": "active",
        "metadata": {},
        "created_at": None,
        "updated_at": None,
        "similarity": 0.91,
    }
    row.update(overrides)
    return row


class TestStoreEmbedding:
    async def test_writes_vector_literal_and_returns_true_on_hit(self) -> None:
        db = AsyncMock()
        db.fetchval = AsyncMock(return_value="claim_x")
        repo = ClaimRetrievalRepository(db)

        ok = await repo.store_embedding("claim_x", [0.1, 0.2, 0.3])

        assert ok is True
        sql = db.fetchval.call_args.args[0]
        assert "UPDATE news_intel.evidence_claims" in sql
        assert "embedding = $2" in sql
        assert "embedded_at = NOW()" in sql
        # Vector passed as pgvector string literal.
        passed = db.fetchval.call_args.args
        assert "claim_x" in passed
        assert "[0.1,0.2,0.3]" in passed

    async def test_returns_false_when_claim_missing(self) -> None:
        db = AsyncMock()
        db.fetchval = AsyncMock(return_value=None)
        repo = ClaimRetrievalRepository(db)

        assert await repo.store_embedding("nope", [0.1]) is False


class TestListUnembedded:
    async def test_filters_null_embedding_active_and_maps(self) -> None:
        db = AsyncMock()
        db.fetch = AsyncMock(return_value=[_full_row(), _full_row(claim_id="claim_y")])
        repo = ClaimRetrievalRepository(db)

        claims = await repo.list_unembedded(limit=50)

        assert [c.claim_id for c in claims] == ["claim_x", "claim_y"]
        sql = db.fetch.call_args.args[0]
        assert "embedding IS NULL" in sql
        assert "status = 'active'" in sql
        assert 50 in db.fetch.call_args.args


class TestSearch:
    async def test_maps_rows_to_scored_claims(self) -> None:
        db = AsyncMock()
        db.fetch = AsyncMock(
            return_value=[
                _full_row(similarity=0.91),
                _full_row(claim_id="claim_y", similarity=0.74),
            ]
        )
        repo = ClaimRetrievalRepository(db)

        results = await repo.search([0.1, 0.2], limit=10, threshold=0.3, filters=None)

        assert [(c.claim_id, s) for c, s in results] == [("claim_x", 0.91), ("claim_y", 0.74)]

    async def test_sql_uses_cosine_distance_and_threshold(self) -> None:
        db = AsyncMock()
        db.fetch = AsyncMock(return_value=[])
        repo = ClaimRetrievalRepository(db)

        await repo.search([0.1, 0.2], limit=7, threshold=0.4, filters=None)

        sql = db.fetch.call_args.args[0]
        assert "embedding <=> $1" in sql  # cosine distance ranking
        assert "1 - (embedding <=> $1)" in sql  # similarity projection + threshold
        assert "ORDER BY embedding <=> $1" in sql
        assert "embedding IS NOT NULL" in sql
        passed = db.fetch.call_args.args
        assert "[0.1,0.2]" in passed
        assert 0.4 in passed
        assert 7 in passed

    async def test_applies_structured_filters(self) -> None:
        db = AsyncMock()
        db.fetch = AsyncMock(return_value=[])
        repo = ClaimRetrievalRepository(db)

        await repo.search(
            [0.1],
            limit=10,
            threshold=0.3,
            filters=ClaimRetrievalFilter(
                lanes=["narrative", "filing"],
                status="active",
                min_confidence=0.5,
                subject_concept_id="concept_tsmc",
                exclude_claim_ids=["claim_z"],
            ),
        )

        sql = db.fetch.call_args.args[0]
        assert "lane = ANY(" in sql
        assert "status =" in sql
        assert "confidence >=" in sql
        assert "subject_concept_id =" in sql
        assert "claim_id != ALL(" in sql
        passed = db.fetch.call_args.args
        assert ["narrative", "filing"] in passed
        assert "concept_tsmc" in passed
        assert 0.5 in passed

    async def test_theme_filter_uses_document_membership(self) -> None:
        db = AsyncMock()
        db.fetch = AsyncMock(return_value=[])
        repo = ClaimRetrievalRepository(db)

        await repo.search(
            [0.1], limit=10, threshold=0.3, filters=ClaimRetrievalFilter(theme_id="theme_ai")
        )

        sql = db.fetch.call_args.args[0]
        # Theme membership is resolved through the source document's theme_ids.
        assert "EXISTS" in sql
        assert "documents" in sql
        assert "theme_ids" in sql
        assert "theme_ai" in db.fetch.call_args.args

    async def test_none_status_filter_searches_all_states(self) -> None:
        db = AsyncMock()
        db.fetch = AsyncMock(return_value=[])
        repo = ClaimRetrievalRepository(db)

        await repo.search([0.1], limit=10, threshold=0.3, filters=ClaimRetrievalFilter(status=None))

        sql = db.fetch.call_args.args[0]
        assert "status =" not in sql
