"""Repository for evidence claims.

CRUD operations against news_intel.evidence_claims with idempotent
upserts keyed by claim_key.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.claims.schemas import VALID_CLAIM_STATUSES, EvidenceClaim
from src.storage.database import Database

logger = logging.getLogger(__name__)


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value)


def _row_to_claim(row: Any) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=row["claim_id"],
        claim_key=row["claim_key"],
        lane=row["lane"],
        run_id=row["run_id"],
        source_id=row["source_id"],
        source_type=row["source_type"],
        source_span_start=row["source_span_start"],
        source_span_end=row["source_span_end"],
        source_text=row["source_text"],
        subject_text=row["subject_text"],
        subject_concept_id=row["subject_concept_id"],
        predicate=row["predicate"],
        object_text=row["object_text"],
        object_concept_id=row["object_concept_id"],
        confidence=float(row["confidence"]),
        extraction_method=row["extraction_method"],
        claim_valid_from=row["claim_valid_from"],
        claim_valid_to=row["claim_valid_to"],
        source_published_at=row["source_published_at"],
        contract_version=row["contract_version"],
        status=row["status"],
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class ClaimRepository:
    """CRUD operations for evidence claims."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def upsert_claim(self, claim: EvidenceClaim) -> EvidenceClaim:
        """Insert or update a claim (idempotent on claim_key).

        On conflict (same claim_key), updates confidence, status,
        metadata, and concept resolutions — preserving the original
        source lineage fields.
        """
        row = await self._db.fetchrow(
            """
            INSERT INTO news_intel.evidence_claims (
                claim_id, claim_key, lane, run_id,
                source_id, source_type,
                source_span_start, source_span_end, source_text,
                subject_text, subject_concept_id,
                predicate, object_text, object_concept_id,
                confidence, extraction_method,
                claim_valid_from, claim_valid_to, source_published_at,
                contract_version, status, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19,
                $20, $21, $22
            )
            ON CONFLICT (claim_key) DO UPDATE SET
                run_id = COALESCE($4, news_intel.evidence_claims.run_id),
                confidence = $15,
                subject_concept_id = COALESCE($11, news_intel.evidence_claims.subject_concept_id),
                object_concept_id = COALESCE($14, news_intel.evidence_claims.object_concept_id),
                status = $21,
                metadata = $22
            RETURNING *
            """,
            claim.claim_id,
            claim.claim_key,
            claim.lane,
            claim.run_id,
            claim.source_id,
            claim.source_type,
            claim.source_span_start,
            claim.source_span_end,
            claim.source_text,
            claim.subject_text,
            claim.subject_concept_id,
            claim.predicate,
            claim.object_text,
            claim.object_concept_id,
            claim.confidence,
            claim.extraction_method,
            claim.claim_valid_from,
            claim.claim_valid_to,
            claim.source_published_at,
            claim.contract_version,
            claim.status,
            json.dumps(claim.metadata),
        )
        return _row_to_claim(row)

    async def get_claim(self, claim_id: str) -> EvidenceClaim | None:
        """Fetch a claim by ID."""
        row = await self._db.fetchrow(
            "SELECT * FROM news_intel.evidence_claims WHERE claim_id = $1",
            claim_id,
        )
        return _row_to_claim(row) if row else None

    async def get_by_key(self, claim_key: str) -> EvidenceClaim | None:
        """Fetch a claim by its deterministic key."""
        row = await self._db.fetchrow(
            "SELECT * FROM news_intel.evidence_claims WHERE claim_key = $1",
            claim_key,
        )
        return _row_to_claim(row) if row else None

    async def list_claims(
        self,
        *,
        lane: str | None = None,
        source_id: str | None = None,
        subject_concept_id: str | None = None,
        predicate: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[EvidenceClaim]:
        """List claims with optional filters."""
        conditions = []
        params: list[Any] = []
        if lane is not None:
            params.append(lane)
            conditions.append(f"lane = ${len(params)}")
        if source_id is not None:
            params.append(source_id)
            conditions.append(f"source_id = ${len(params)}")
        if subject_concept_id is not None:
            params.append(subject_concept_id)
            conditions.append(f"subject_concept_id = ${len(params)}")
        if predicate is not None:
            params.append(predicate)
            conditions.append(f"predicate = ${len(params)}")
        if status is not None:
            params.append(status)
            conditions.append(f"status = ${len(params)}")
        params.append(limit)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT * FROM news_intel.evidence_claims
            {where}
            ORDER BY created_at DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_claim(row) for row in rows]

    async def update_status(self, claim_id: str, new_status: str) -> EvidenceClaim | None:
        """Update a claim's status."""
        if new_status not in VALID_CLAIM_STATUSES:
            raise ValueError(
                f"Invalid claim status {new_status!r}. "
                f"Must be one of {sorted(VALID_CLAIM_STATUSES)}"
            )
        row = await self._db.fetchrow(
            """
            UPDATE news_intel.evidence_claims
            SET status = $2
            WHERE claim_id = $1
            RETURNING *
            """,
            claim_id,
            new_status,
        )
        return _row_to_claim(row) if row else None

    async def resolve_concepts(
        self,
        claim_id: str,
        *,
        subject_concept_id: str | None = None,
        object_concept_id: str | None = None,
    ) -> EvidenceClaim | None:
        """Update concept resolutions on a claim."""
        row = await self._db.fetchrow(
            """
            UPDATE news_intel.evidence_claims
            SET subject_concept_id = COALESCE($2, subject_concept_id),
                object_concept_id = COALESCE($3, object_concept_id)
            WHERE claim_id = $1
            RETURNING *
            """,
            claim_id,
            subject_concept_id,
            object_concept_id,
        )
        return _row_to_claim(row) if row else None
