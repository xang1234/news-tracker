"""Repository for the claim retrieval substrate.

Owns the embedding column on ``news_intel.evidence_claims`` (store + pending
lookup) and the pgvector cosine-similarity search. Mirrors the documents
embedding pattern: vectors are passed as pgvector string literals and ranked
with the ``<=>`` distance operator over an HNSW index (migration 045).
"""

from __future__ import annotations

import logging
from typing import Any

from src.claims.repository import claim_from_row
from src.claims.schemas import EvidenceClaim
from src.retrieval.schemas import ClaimRetrievalFilter
from src.storage.database import Database
from src.storage.pgvector import to_pgvector_literal

logger = logging.getLogger(__name__)


class ClaimRetrievalRepository:
    """Embedding persistence + semantic search over evidence claims."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def store_embedding(self, claim_id: str, embedding: list[float]) -> bool:
        """Persist a claim's retrieval embedding. Returns False if not found."""
        result = await self._db.fetchval(
            """
            UPDATE news_intel.evidence_claims
            SET embedding = $2, embedded_at = NOW()
            WHERE claim_id = $1
            RETURNING claim_id
            """,
            claim_id,
            to_pgvector_literal(embedding),
        )
        return result is not None

    async def list_unembedded(self, limit: int) -> list[EvidenceClaim]:
        """List active claims that still lack a retrieval embedding."""
        rows = await self._db.fetch(
            """
            SELECT * FROM news_intel.evidence_claims
            WHERE embedding IS NULL AND status = 'active'
            ORDER BY created_at ASC
            LIMIT $1
            """,
            limit,
        )
        return [claim_from_row(row) for row in rows]

    async def search(
        self,
        query_embedding: list[float],
        *,
        limit: int,
        threshold: float,
        filters: ClaimRetrievalFilter | None,
    ) -> list[tuple[EvidenceClaim, float]]:
        """Return ``(claim, similarity)`` pairs ranked by cosine similarity.

        Structured constraints in ``filters`` are applied as SQL predicates
        alongside the similarity threshold; ``theme_id`` is resolved through
        the source document's ``theme_ids`` via a correlated ``EXISTS``. The
        service wraps these pairs into :class:`RetrievedClaim`.
        """
        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = [to_pgvector_literal(query_embedding)]

        if filters is not None:
            if filters.lanes:
                params.append(filters.lanes)
                conditions.append(f"lane = ANY(${len(params)})")
            if filters.status is not None:
                params.append(filters.status)
                conditions.append(f"status = ${len(params)}")
            if filters.min_confidence is not None:
                params.append(filters.min_confidence)
                conditions.append(f"confidence >= ${len(params)}")
            if filters.subject_concept_id is not None:
                params.append(filters.subject_concept_id)
                conditions.append(f"subject_concept_id = ${len(params)}")
            if filters.exclude_claim_ids:
                params.append(filters.exclude_claim_ids)
                conditions.append(f"claim_id != ALL(${len(params)})")
            if filters.theme_id is not None:
                params.append(filters.theme_id)
                conditions.append(
                    "source_type = 'document' AND EXISTS ("
                    "SELECT 1 FROM documents d "
                    "WHERE d.id = source_id "
                    f"AND d.theme_ids && ARRAY[${len(params)}]::text[])"
                )

        params.append(threshold)
        threshold_idx = len(params)
        params.append(limit)
        limit_idx = len(params)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT *, 1 - (embedding <=> $1) AS similarity
            FROM news_intel.evidence_claims
            WHERE {where_clause}
              AND 1 - (embedding <=> $1) >= ${threshold_idx}
            ORDER BY embedding <=> $1
            LIMIT ${limit_idx}
        """
        rows = await self._db.fetch(sql, *params)
        return [(claim_from_row(row), float(row["similarity"])) for row in rows]
