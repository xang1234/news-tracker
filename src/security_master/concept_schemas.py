"""Schema definitions for the concept registry.

Extends the security master with canonical concept IDs, aliases,
and issuer/security crosswalks. These dataclasses map 1:1 to the
tables created in migration 019.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

VALID_CONCEPT_TYPES = frozenset(
    {
        "issuer",
        "security",
        "technology",
        "product",
        "theme",
        "narrative_frame",
        "facility",
        "index",
    }
)

VALID_ALIAS_TYPES = frozenset(
    {"name", "ticker", "abbreviation", "former_name", "local_name"}
)

VALID_ISSUER_SECURITY_RELATIONSHIPS = frozenset(
    {"primary", "adr", "subsidiary_listing", "preferred", "warrant"}
)


def make_concept_id(concept_type: str, canonical_name: str) -> str:
    """Generate a deterministic concept ID.

    Same type + name always produces the same ID, enabling replay
    safety and idempotent upserts.
    """
    key = f"{concept_type}:{canonical_name.lower().strip()}"
    return f"concept_{concept_type}_{hashlib.sha256(key.encode()).hexdigest()[:12]}"


@dataclass
class Concept:
    """A canonical concept in the identity registry.

    Attributes:
        concept_id: Deterministic ID (concept_{type}_{hash}).
        concept_type: Kind of entity (issuer, security, technology, etc.).
        canonical_name: The authoritative name for this concept.
        description: Human-readable description.
        metadata: Extensible metadata (sector, country, etc.).
        is_active: Soft-delete flag.
    """

    concept_id: str
    concept_type: str
    canonical_name: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.concept_type not in VALID_CONCEPT_TYPES:
            raise ValueError(
                f"Invalid concept_type {self.concept_type!r}. "
                f"Must be one of {sorted(VALID_CONCEPT_TYPES)}"
            )


@dataclass
class ConceptAlias:
    """An alias that resolves to a canonical concept.

    Attributes:
        alias: The alternative name (e.g., "TSMC", "2330.TW").
        concept_id: The concept this alias resolves to.
        alias_type: Kind of alias (name, ticker, abbreviation, etc.).
        is_primary: Whether this is the primary/preferred alias.
    """

    alias: str
    concept_id: str
    alias_type: str = "name"
    is_primary: bool = False
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.alias_type not in VALID_ALIAS_TYPES:
            raise ValueError(
                f"Invalid alias_type {self.alias_type!r}. "
                f"Must be one of {sorted(VALID_ALIAS_TYPES)}"
            )


@dataclass
class IssuerSecurityLink:
    """Maps an issuer concept to a security concept.

    Attributes:
        issuer_concept_id: The issuer (company) concept.
        security_concept_id: The security (tradeable instrument) concept.
        relationship_type: How they relate (primary, adr, subsidiary, etc.).
        metadata: Extensible metadata.
    """

    issuer_concept_id: str
    security_concept_id: str
    relationship_type: str = "primary"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.relationship_type not in VALID_ISSUER_SECURITY_RELATIONSHIPS:
            raise ValueError(
                f"Invalid relationship_type {self.relationship_type!r}. "
                f"Must be one of {sorted(VALID_ISSUER_SECURITY_RELATIONSHIPS)}"
            )
