"""Seed loader for domain packs.

Loads a pack JSON file and populates the concept registry, coverage
profiles, and pack membership tables. Designed for idempotent seeding
— safe to call multiple times.

Usage:
    seeder = PackSeeder(concept_repo, coverage_repo)
    stats = await seeder.seed_pack("src/coverage/data/semiconductors_pack_1.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.coverage.schemas import (
    CoverageProfile,
    CoverageTierChange,
    DomainPack,
    DomainPackMember,
)
from src.security_master.concept_schemas import (
    Concept,
    ConceptAlias,
    ConceptRelationship,
    IssuerSecurityLink,
    make_concept_id,
)

logger = logging.getLogger(__name__)

# Default pack data path
_DEFAULT_PACK_PATH = Path(__file__).parent / "data" / "semiconductors_pack_1.json"


@dataclass
class SeedStats:
    """Statistics from a seeding operation."""

    concepts: int = 0
    aliases: int = 0
    relationships: int = 0
    coverage_profiles: int = 0
    pack_members: int = 0
    errors: list[str] = field(default_factory=list)


def load_pack_data(path: Path | None = None) -> dict[str, Any]:
    """Load pack JSON data from disk."""
    path = path or _DEFAULT_PACK_PATH
    with open(path) as f:
        return json.load(f)


def build_seed_objects(
    data: dict[str, Any],
) -> tuple[
    DomainPack,
    list[Concept],
    list[ConceptAlias],
    list[IssuerSecurityLink],
    list[ConceptRelationship],
    list[CoverageProfile],
    list[DomainPackMember],
]:
    """Build all seed objects from pack JSON data.

    This is a pure function — no I/O. Returns objects ready to be
    persisted by a repository. Can be used in tests without a database.

    Returns:
        Tuple of (pack, concepts, aliases, issuer_security_links,
        relationships, coverage_profiles, pack_members).
    """
    pack_data = data["pack"]
    pack = DomainPack(
        pack_id=pack_data["pack_id"],
        name=pack_data["name"],
        description=pack_data.get("description", ""),
        version=pack_data.get("version", "1.0"),
    )

    concepts: list[Concept] = []
    aliases: list[ConceptAlias] = []
    issuer_links: list[IssuerSecurityLink] = []
    relationships: list[ConceptRelationship] = []
    coverage_profiles: list[CoverageProfile] = []
    pack_members: list[DomainPackMember] = []

    # Name → concept_id lookup for building relationships
    name_to_id: dict[str, str] = {}

    # -- Issuers and their securities --
    for issuer_data in data.get("issuers", []):
        name = issuer_data["canonical_name"]
        issuer_id = make_concept_id("issuer", name)
        name_to_id[name] = issuer_id

        concepts.append(
            Concept(
                concept_id=issuer_id,
                concept_type="issuer",
                canonical_name=name,
                metadata={"sector": issuer_data.get("sector", "")},
            )
        )

        # Primary alias is the canonical name
        aliases.append(
            ConceptAlias(
                alias=name,
                concept_id=issuer_id,
                alias_type="name",
                is_primary=True,
            )
        )
        # Additional aliases
        for alias_name in issuer_data.get("aliases", []):
            aliases.append(
                ConceptAlias(
                    alias=alias_name,
                    concept_id=issuer_id,
                    alias_type="name",
                )
            )
        # Ticker alias
        ticker = issuer_data.get("ticker", "")
        if ticker:
            aliases.append(
                ConceptAlias(
                    alias=ticker,
                    concept_id=issuer_id,
                    alias_type="ticker",
                )
            )

            # Security concept + issuer-security link
            exchange = issuer_data.get("exchange", "US")
            sec_id = make_concept_id("security", f"{ticker}:{exchange}")
            concepts.append(
                Concept(
                    concept_id=sec_id,
                    concept_type="security",
                    canonical_name=f"{ticker} ({exchange})",
                    metadata={
                        "ticker": ticker,
                        "exchange": exchange,
                        "sector": issuer_data.get("sector", ""),
                    },
                )
            )
            issuer_links.append(
                IssuerSecurityLink(
                    issuer_concept_id=issuer_id,
                    security_concept_id=sec_id,
                )
            )

        # Coverage profile
        coverage_profiles.append(
            CoverageProfile(
                concept_id=issuer_id,
                coverage_tier=issuer_data.get("coverage_tier", "stub"),
                narrative_coverage=True,
            )
        )

        # Pack membership
        pack_members.append(
            DomainPackMember(
                pack_id=pack.pack_id,
                concept_id=issuer_id,
                role=issuer_data.get("role", "member"),
            )
        )

    # -- Technologies --
    for tech_data in data.get("technologies", []):
        name = tech_data["canonical_name"]
        tech_id = make_concept_id("technology", name)
        name_to_id[name] = tech_id

        concepts.append(
            Concept(
                concept_id=tech_id,
                concept_type="technology",
                canonical_name=name,
                description=tech_data.get("description", ""),
            )
        )
        aliases.append(
            ConceptAlias(
                alias=name,
                concept_id=tech_id,
                alias_type="name",
                is_primary=True,
            )
        )
        for alias_name in tech_data.get("aliases", []):
            aliases.append(
                ConceptAlias(
                    alias=alias_name,
                    concept_id=tech_id,
                    alias_type="abbreviation",
                )
            )

        coverage_profiles.append(
            CoverageProfile(
                concept_id=tech_id,
                coverage_tier="partial",
            )
        )
        pack_members.append(
            DomainPackMember(
                pack_id=pack.pack_id,
                concept_id=tech_id,
                role="peripheral",
            )
        )

    # -- Themes --
    for theme_data in data.get("themes", []):
        name = theme_data["canonical_name"]
        theme_id = make_concept_id("theme", name)
        name_to_id[name] = theme_id

        concepts.append(
            Concept(
                concept_id=theme_id,
                concept_type="theme",
                canonical_name=name,
                description=theme_data.get("description", ""),
            )
        )
        aliases.append(
            ConceptAlias(
                alias=name,
                concept_id=theme_id,
                alias_type="name",
                is_primary=True,
            )
        )

        coverage_profiles.append(
            CoverageProfile(
                concept_id=theme_id,
                coverage_tier="partial",
                narrative_coverage=True,
            )
        )
        pack_members.append(
            DomainPackMember(
                pack_id=pack.pack_id,
                concept_id=theme_id,
                role="peripheral",
            )
        )

    # -- Supply chain relationships --
    for rel_data in data.get("supply_chain", []):
        source_name = rel_data["source"]
        target_name = rel_data["target"]
        source_id = name_to_id.get(source_name)
        target_id = name_to_id.get(target_name)
        if source_id and target_id:
            relationships.append(
                ConceptRelationship(
                    source_concept_id=source_id,
                    target_concept_id=target_id,
                    relationship_type=rel_data["type"],
                    source_attribution="semiconductors_pack_1",
                )
            )

    return (
        pack,
        concepts,
        aliases,
        issuer_links,
        relationships,
        coverage_profiles,
        pack_members,
    )
