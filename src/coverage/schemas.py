"""Schema definitions for coverage profiles and domain packs.

Maps 1:1 to the tables in migration 021.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

VALID_COVERAGE_TIERS = frozenset({"full", "partial", "stub", "unsupported"})

VALID_PACK_ROLES = frozenset({"anchor", "member", "peripheral"})


@dataclass
class CoverageProfile:
    """Current coverage state for a concept.

    Attributes:
        concept_id: The concept this profile describes.
        coverage_tier: Maturity level (full, partial, stub, unsupported).
        coverage_notes: Human-readable notes about coverage gaps.
        structural_completeness: 0-1 score of structural data completeness.
        filing_coverage: Whether SEC filing data is available.
        narrative_coverage: Whether narrative/news data is tracked.
        graph_coverage: Whether graph relationships are populated.
        last_assessed_at: When the coverage was last evaluated.
        metadata: Extensible metadata.
    """

    concept_id: str
    coverage_tier: str = "stub"
    coverage_notes: str = ""
    structural_completeness: float = 0.0
    filing_coverage: bool = False
    narrative_coverage: bool = False
    graph_coverage: bool = False
    last_assessed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.coverage_tier not in VALID_COVERAGE_TIERS:
            raise ValueError(
                f"Invalid coverage_tier {self.coverage_tier!r}. "
                f"Must be one of {sorted(VALID_COVERAGE_TIERS)}"
            )


@dataclass
class CoverageTierChange:
    """A historical tier change record (append-only).

    Attributes:
        id: Auto-generated row ID.
        concept_id: The concept whose tier changed.
        coverage_tier: The tier at this point in time.
        valid_from: When this tier became effective.
        valid_to: When this tier was superseded (None = current).
        changed_by: Who/what triggered the change.
        change_reason: Why the tier changed.
        metadata: Extensible metadata.
    """

    concept_id: str
    coverage_tier: str
    valid_from: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    valid_to: datetime | None = None
    changed_by: str | None = None
    change_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.coverage_tier not in VALID_COVERAGE_TIERS:
            raise ValueError(
                f"Invalid coverage_tier {self.coverage_tier!r}. "
                f"Must be one of {sorted(VALID_COVERAGE_TIERS)}"
            )


@dataclass
class DomainPack:
    """A named coverage domain grouping related concepts.

    Attributes:
        pack_id: Unique identifier (e.g., "semiconductors_pack_1").
        name: Human-readable name.
        description: What this pack covers.
        version: Pack version string.
        is_active: Soft-delete flag.
        metadata: Extensible metadata.
    """

    pack_id: str
    name: str
    description: str = ""
    version: str = "1.0"
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class DomainPackMember:
    """Membership of a concept in a domain pack.

    Attributes:
        pack_id: Which pack the concept belongs to.
        concept_id: The member concept.
        role: Role within the pack (anchor, member, peripheral).
        added_at: When the concept was added to the pack.
        metadata: Extensible metadata.
    """

    pack_id: str
    concept_id: str
    role: str = "member"
    added_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in VALID_PACK_ROLES:
            raise ValueError(
                f"Invalid role {self.role!r}. "
                f"Must be one of {sorted(VALID_PACK_ROLES)}"
            )
