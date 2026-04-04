"""Repository for resolved assertions and claim links.

CRUD operations against news_intel.resolved_assertions and
news_intel.assertion_claim_links. Assertions use idempotent
upserts keyed by assertion_id. Claim links use composite PK
(assertion_id, claim_id) for dedup.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.assertions.schemas import (
    VALID_ASSERTION_STATUSES,
    AssertionClaimLink,
    ResolvedAssertion,
)
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


def _row_to_assertion(row: Any) -> ResolvedAssertion:
    return ResolvedAssertion(
        assertion_id=row["assertion_id"],
        subject_concept_id=row["subject_concept_id"],
        predicate=row["predicate"],
        object_concept_id=row["object_concept_id"],
        confidence=float(row["confidence"]),
        status=row["status"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        support_count=row["support_count"],
        contradiction_count=row["contradiction_count"],
        first_seen_at=row["first_seen_at"],
        last_evidence_at=row["last_evidence_at"],
        source_diversity=row["source_diversity"],
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_link(row: Any) -> AssertionClaimLink:
    return AssertionClaimLink(
        assertion_id=row["assertion_id"],
        claim_id=row["claim_id"],
        link_type=row["link_type"],
        contribution_weight=float(row["contribution_weight"]),
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
    )


class AssertionRepository:
    """CRUD operations for resolved assertions and claim links."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # -- Assertion CRUD ----------------------------------------------------

    async def upsert_assertion(
        self, assertion: ResolvedAssertion
    ) -> ResolvedAssertion:
        """Insert or update an assertion (idempotent on assertion_id).

        On conflict, updates aggregate fields (confidence, counts,
        timestamps, status) while preserving the original triple.
        """
        row = await self._db.fetchrow(
            """
            INSERT INTO news_intel.resolved_assertions (
                assertion_id, subject_concept_id, predicate,
                object_concept_id, confidence, status,
                valid_from, valid_to,
                support_count, contradiction_count,
                first_seen_at, last_evidence_at,
                source_diversity, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14
            )
            ON CONFLICT (assertion_id) DO UPDATE SET
                confidence = $5,
                status = $6,
                valid_from = COALESCE($7, news_intel.resolved_assertions.valid_from),
                valid_to = $8,
                support_count = $9,
                contradiction_count = $10,
                first_seen_at = LEAST(
                    $11, news_intel.resolved_assertions.first_seen_at
                ),
                last_evidence_at = GREATEST(
                    $12, news_intel.resolved_assertions.last_evidence_at
                ),
                source_diversity = $13,
                metadata = $14
            RETURNING *
            """,
            assertion.assertion_id,
            assertion.subject_concept_id,
            assertion.predicate,
            assertion.object_concept_id,
            assertion.confidence,
            assertion.status,
            assertion.valid_from,
            assertion.valid_to,
            assertion.support_count,
            assertion.contradiction_count,
            assertion.first_seen_at,
            assertion.last_evidence_at,
            assertion.source_diversity,
            json.dumps(assertion.metadata),
        )
        return _row_to_assertion(row)

    async def get_assertion(
        self, assertion_id: str
    ) -> ResolvedAssertion | None:
        """Fetch an assertion by ID."""
        row = await self._db.fetchrow(
            "SELECT * FROM news_intel.resolved_assertions "
            "WHERE assertion_id = $1",
            assertion_id,
        )
        return _row_to_assertion(row) if row else None

    async def get_by_triple(
        self,
        subject_concept_id: str,
        predicate: str,
        object_concept_id: str | None = None,
    ) -> ResolvedAssertion | None:
        """Fetch an assertion by its subject-predicate-object triple."""
        from src.assertions.schemas import make_assertion_id

        aid = make_assertion_id(subject_concept_id, predicate, object_concept_id)
        return await self.get_assertion(aid)

    async def list_assertions(
        self,
        *,
        subject_concept_id: str | None = None,
        predicate: str | None = None,
        object_concept_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ResolvedAssertion]:
        """List assertions with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if subject_concept_id is not None:
            params.append(subject_concept_id)
            conditions.append(f"subject_concept_id = ${len(params)}")
        if predicate is not None:
            params.append(predicate)
            conditions.append(f"predicate = ${len(params)}")
        if object_concept_id is not None:
            params.append(object_concept_id)
            conditions.append(f"object_concept_id = ${len(params)}")
        if status is not None:
            params.append(status)
            conditions.append(f"status = ${len(params)}")

        params.append(limit)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT * FROM news_intel.resolved_assertions
            {where}
            ORDER BY confidence DESC, last_evidence_at DESC NULLS LAST
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [_row_to_assertion(row) for row in rows]

    async def list_for_concept(
        self, concept_id: str, *, limit: int = 50
    ) -> list[ResolvedAssertion]:
        """List all assertions involving a concept (as subject or object)."""
        rows = await self._db.fetch(
            """
            SELECT * FROM news_intel.resolved_assertions
            WHERE subject_concept_id = $1 OR object_concept_id = $1
            ORDER BY confidence DESC, last_evidence_at DESC NULLS LAST
            LIMIT $2
            """,
            concept_id,
            limit,
        )
        return [_row_to_assertion(row) for row in rows]

    async def update_status(
        self, assertion_id: str, new_status: str
    ) -> ResolvedAssertion | None:
        """Update an assertion's status."""
        if new_status not in VALID_ASSERTION_STATUSES:
            raise ValueError(
                f"Invalid assertion status {new_status!r}. "
                f"Must be one of {sorted(VALID_ASSERTION_STATUSES)}"
            )
        row = await self._db.fetchrow(
            """
            UPDATE news_intel.resolved_assertions
            SET status = $2
            WHERE assertion_id = $1
            RETURNING *
            """,
            assertion_id,
            new_status,
        )
        return _row_to_assertion(row) if row else None

    # -- Claim link CRUD ---------------------------------------------------

    async def upsert_link(
        self, link: AssertionClaimLink
    ) -> AssertionClaimLink:
        """Insert or update a claim link (idempotent on assertion_id + claim_id).

        On conflict, updates link_type and weight (a claim might
        transition from support to contradiction on re-evaluation).
        """
        row = await self._db.fetchrow(
            """
            INSERT INTO news_intel.assertion_claim_links (
                assertion_id, claim_id, link_type,
                contribution_weight, metadata
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (assertion_id, claim_id) DO UPDATE SET
                link_type = $3,
                contribution_weight = $4,
                metadata = $5
            RETURNING *
            """,
            link.assertion_id,
            link.claim_id,
            link.link_type,
            link.contribution_weight,
            json.dumps(link.metadata),
        )
        return _row_to_link(row)

    async def get_links_for_assertion(
        self, assertion_id: str
    ) -> list[AssertionClaimLink]:
        """Get all claim links for an assertion."""
        rows = await self._db.fetch(
            """
            SELECT * FROM news_intel.assertion_claim_links
            WHERE assertion_id = $1
            ORDER BY contribution_weight DESC, created_at ASC
            """,
            assertion_id,
        )
        return [_row_to_link(row) for row in rows]

    async def get_links_for_claim(
        self, claim_id: str
    ) -> list[AssertionClaimLink]:
        """Get all assertion links for a specific claim."""
        rows = await self._db.fetch(
            """
            SELECT * FROM news_intel.assertion_claim_links
            WHERE claim_id = $1
            ORDER BY created_at ASC
            """,
            claim_id,
        )
        return [_row_to_link(row) for row in rows]

    async def count_links(
        self,
        assertion_id: str,
        link_type: str | None = None,
    ) -> int:
        """Count claim links for an assertion, optionally by type."""
        if link_type is not None:
            row = await self._db.fetchrow(
                """
                SELECT COUNT(*) AS cnt
                FROM news_intel.assertion_claim_links
                WHERE assertion_id = $1 AND link_type = $2
                """,
                assertion_id,
                link_type,
            )
        else:
            row = await self._db.fetchrow(
                """
                SELECT COUNT(*) AS cnt
                FROM news_intel.assertion_claim_links
                WHERE assertion_id = $1
                """,
                assertion_id,
            )
        return row["cnt"] if row else 0
