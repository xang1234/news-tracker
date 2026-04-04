"""Repository for the concept registry.

CRUD and resolution operations against the concepts, concept_aliases,
and issuer_security_map tables.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.security_master.concept_schemas import (
    Concept,
    ConceptAlias,
    IssuerSecurityLink,
)
from src.storage.database import Database

logger = logging.getLogger(__name__)


# -- Row converters --------------------------------------------------------


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value)


def _row_to_concept(row: Any) -> Concept:
    return Concept(
        concept_id=row["concept_id"],
        concept_type=row["concept_type"],
        canonical_name=row["canonical_name"],
        description=row["description"],
        metadata=_parse_json(row["metadata"]),
        is_active=row["is_active"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_alias(row: Any) -> ConceptAlias:
    return ConceptAlias(
        alias=row["alias"],
        concept_id=row["concept_id"],
        alias_type=row["alias_type"],
        is_primary=row["is_primary"],
        created_at=row["created_at"],
    )


def _row_to_link(row: Any) -> IssuerSecurityLink:
    return IssuerSecurityLink(
        issuer_concept_id=row["issuer_concept_id"],
        security_concept_id=row["security_concept_id"],
        relationship_type=row["relationship_type"],
        metadata=_parse_json(row["metadata"]),
        created_at=row["created_at"],
    )


class ConceptRepository:
    """CRUD operations for the concept registry."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # -- Concepts ----------------------------------------------------------

    async def upsert_concept(self, concept: Concept) -> Concept:
        """Insert or update a concept."""
        row = await self._db.fetchrow(
            """
            INSERT INTO concepts (
                concept_id, concept_type, canonical_name,
                description, metadata, is_active
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (concept_id) DO UPDATE
            SET canonical_name = $3,
                description = $4,
                metadata = $5,
                is_active = $6
            RETURNING *
            """,
            concept.concept_id,
            concept.concept_type,
            concept.canonical_name,
            concept.description,
            json.dumps(concept.metadata),
            concept.is_active,
        )
        return _row_to_concept(row)

    async def get_concept(self, concept_id: str) -> Concept | None:
        """Fetch a concept by ID."""
        row = await self._db.fetchrow(
            "SELECT * FROM concepts WHERE concept_id = $1",
            concept_id,
        )
        return _row_to_concept(row) if row else None

    async def list_concepts(
        self,
        concept_type: str | None = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Concept]:
        """List concepts with optional filters."""
        conditions = []
        params: list[Any] = []
        if concept_type is not None:
            params.append(concept_type)
            conditions.append(f"concept_type = ${len(params)}")
        if active_only:
            conditions.append("is_active = TRUE")
        params.append(limit)
        limit_idx = len(params)
        params.append(offset)
        offset_idx = len(params)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.fetch(
            f"""
            SELECT * FROM concepts
            {where}
            ORDER BY canonical_name
            LIMIT ${limit_idx} OFFSET ${offset_idx}
            """,
            *params,
        )
        return [_row_to_concept(row) for row in rows]

    async def search_concepts(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Concept]:
        """Fuzzy search concepts by canonical name using pg_trgm."""
        rows = await self._db.fetch(
            """
            SELECT *, similarity(canonical_name, $1) AS sim
            FROM concepts
            WHERE is_active = TRUE
              AND similarity(canonical_name, $1) > 0.2
            ORDER BY sim DESC
            LIMIT $2
            """,
            query,
            limit,
        )
        return [_row_to_concept(row) for row in rows]

    async def deactivate_concept(self, concept_id: str) -> bool:
        """Soft-delete a concept."""
        result = await self._db.execute(
            """
            UPDATE concepts SET is_active = FALSE
            WHERE concept_id = $1 AND is_active = TRUE
            """,
            concept_id,
        )
        return "UPDATE 1" in (result or "")

    # -- Aliases -----------------------------------------------------------

    async def add_alias(self, alias: ConceptAlias) -> ConceptAlias:
        """Add an alias for a concept (upsert)."""
        row = await self._db.fetchrow(
            """
            INSERT INTO concept_aliases (
                alias, concept_id, alias_type, is_primary
            ) VALUES ($1, $2, $3, $4)
            ON CONFLICT (alias, concept_id) DO UPDATE
            SET alias_type = $3, is_primary = $4
            RETURNING *
            """,
            alias.alias,
            alias.concept_id,
            alias.alias_type,
            alias.is_primary,
        )
        return _row_to_alias(row)

    async def resolve_alias(self, alias: str) -> Concept | None:
        """Resolve an alias to its concept (case-insensitive)."""
        row = await self._db.fetchrow(
            """
            SELECT c.* FROM concepts c
            JOIN concept_aliases ca ON c.concept_id = ca.concept_id
            WHERE lower(ca.alias) = lower($1)
              AND c.is_active = TRUE
            LIMIT 1
            """,
            alias,
        )
        return _row_to_concept(row) if row else None

    async def get_aliases(self, concept_id: str) -> list[ConceptAlias]:
        """Get all aliases for a concept."""
        rows = await self._db.fetch(
            """
            SELECT * FROM concept_aliases
            WHERE concept_id = $1
            ORDER BY is_primary DESC, alias
            """,
            concept_id,
        )
        return [_row_to_alias(row) for row in rows]

    async def remove_alias(self, alias: str, concept_id: str) -> bool:
        """Remove a specific alias."""
        result = await self._db.execute(
            "DELETE FROM concept_aliases WHERE alias = $1 AND concept_id = $2",
            alias,
            concept_id,
        )
        return "DELETE 1" in (result or "")

    # -- Issuer/security crosswalk -----------------------------------------

    async def link_issuer_security(
        self, link: IssuerSecurityLink
    ) -> IssuerSecurityLink:
        """Link an issuer concept to a security concept (upsert)."""
        row = await self._db.fetchrow(
            """
            INSERT INTO issuer_security_map (
                issuer_concept_id, security_concept_id,
                relationship_type, metadata
            ) VALUES ($1, $2, $3, $4)
            ON CONFLICT (issuer_concept_id, security_concept_id) DO UPDATE
            SET relationship_type = $3, metadata = $4
            RETURNING *
            """,
            link.issuer_concept_id,
            link.security_concept_id,
            link.relationship_type,
            json.dumps(link.metadata),
        )
        return _row_to_link(row)

    async def get_securities_for_issuer(
        self, issuer_concept_id: str
    ) -> list[IssuerSecurityLink]:
        """Get all securities linked to an issuer."""
        rows = await self._db.fetch(
            """
            SELECT * FROM issuer_security_map
            WHERE issuer_concept_id = $1
            ORDER BY relationship_type, security_concept_id
            """,
            issuer_concept_id,
        )
        return [_row_to_link(row) for row in rows]

    async def get_issuer_for_security(
        self, security_concept_id: str
    ) -> IssuerSecurityLink | None:
        """Get the issuer linked to a security (primary first)."""
        row = await self._db.fetchrow(
            """
            SELECT * FROM issuer_security_map
            WHERE security_concept_id = $1
            ORDER BY CASE WHEN relationship_type = 'primary' THEN 0 ELSE 1 END
            LIMIT 1
            """,
            security_concept_id,
        )
        return _row_to_link(row) if row else None

    # -- Security concept_id linkage ---------------------------------------

    async def link_security_to_concept(
        self, ticker: str, exchange: str, concept_id: str
    ) -> bool:
        """Set the concept_id on an existing securities row."""
        result = await self._db.execute(
            """
            UPDATE securities SET concept_id = $3
            WHERE ticker = $1 AND exchange = $2
            """,
            ticker,
            exchange,
            concept_id,
        )
        return "UPDATE 1" in (result or "")

    async def get_concept_for_security(
        self, ticker: str, exchange: str = "US"
    ) -> Concept | None:
        """Resolve a ticker+exchange to its concept."""
        row = await self._db.fetchrow(
            """
            SELECT c.* FROM concepts c
            JOIN securities s ON s.concept_id = c.concept_id
            WHERE s.ticker = $1 AND s.exchange = $2
              AND c.is_active = TRUE
            """,
            ticker,
            exchange,
        )
        return _row_to_concept(row) if row else None
