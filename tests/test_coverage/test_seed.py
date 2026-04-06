"""Tests for semiconductors Pack 1 seed data and build_seed_objects.

Validates that:
1. The JSON seed file loads and parses correctly.
2. build_seed_objects produces valid objects for all entity families.
3. The seed data covers the expected semiconductor issuers and relationships.
4. All generated objects pass dataclass validation.
5. The seed is usable as a template for future pack expansion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.coverage.schemas import (
    DomainPack,
)
from src.coverage.seed import (
    build_seed_objects,
    load_pack_data,
)
from src.security_master.concept_schemas import (
    VALID_CONCEPT_TYPES,
    VALID_RELATIONSHIP_TYPES,
)

PACK_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "coverage" / "data" / "semiconductors_pack_1.json"
)


@pytest.fixture()
def pack_data() -> dict:
    return load_pack_data(PACK_PATH)


@pytest.fixture()
def seed_objects(pack_data):
    return build_seed_objects(pack_data)


class TestLoadPackData:
    """JSON loading."""

    def test_file_exists(self) -> None:
        assert PACK_PATH.exists()

    def test_loads_valid_json(self) -> None:
        data = load_pack_data(PACK_PATH)
        assert "pack" in data
        assert "issuers" in data
        assert "technologies" in data
        assert "themes" in data
        assert "supply_chain" in data

    def test_pack_metadata(self, pack_data: dict) -> None:
        assert pack_data["pack"]["pack_id"] == "semiconductors_pack_1"
        assert "Semiconductor" in pack_data["pack"]["name"]


class TestBuildSeedObjects:
    """build_seed_objects pure function."""

    def test_returns_all_object_types(self, seed_objects) -> None:
        pack, concepts, aliases, links, rels, profiles, members = seed_objects
        assert isinstance(pack, DomainPack)
        assert len(concepts) > 0
        assert len(aliases) > 0
        assert len(links) > 0
        assert len(rels) > 0
        assert len(profiles) > 0
        assert len(members) > 0

    def test_pack_identity(self, seed_objects) -> None:
        pack = seed_objects[0]
        assert pack.pack_id == "semiconductors_pack_1"
        assert pack.version == "1.0"

    def test_concepts_have_valid_types(self, seed_objects) -> None:
        concepts = seed_objects[1]
        for c in concepts:
            assert c.concept_type in VALID_CONCEPT_TYPES

    def test_concept_types_represented(self, seed_objects) -> None:
        concepts = seed_objects[1]
        types = {c.concept_type for c in concepts}
        assert "issuer" in types
        assert "security" in types
        assert "technology" in types
        assert "theme" in types

    def test_concept_ids_deterministic(self, seed_objects) -> None:
        concepts = seed_objects[1]
        for c in concepts:
            assert c.concept_id.startswith("concept_")

    def test_aliases_reference_valid_concepts(self, seed_objects) -> None:
        concepts = seed_objects[1]
        concept_ids = {c.concept_id for c in concepts}
        aliases = seed_objects[2]
        for a in aliases:
            assert a.concept_id in concept_ids

    def test_issuer_security_links_valid(self, seed_objects) -> None:
        concepts = seed_objects[1]
        concept_ids = {c.concept_id for c in concepts}
        links = seed_objects[3]
        for link in links:
            assert link.issuer_concept_id in concept_ids
            assert link.security_concept_id in concept_ids

    def test_relationships_have_valid_types(self, seed_objects) -> None:
        rels = seed_objects[4]
        for r in rels:
            assert r.relationship_type in VALID_RELATIONSHIP_TYPES

    def test_relationships_reference_valid_concepts(self, seed_objects) -> None:
        concepts = seed_objects[1]
        concept_ids = {c.concept_id for c in concepts}
        rels = seed_objects[4]
        for r in rels:
            assert r.source_concept_id in concept_ids, f"Missing source: {r.source_concept_id}"
            assert r.target_concept_id in concept_ids, f"Missing target: {r.target_concept_id}"

    def test_coverage_profiles_match_concepts(self, seed_objects) -> None:
        profiles = seed_objects[5]
        for p in profiles:
            assert p.concept_id.startswith("concept_")

    def test_pack_members_match_concepts(self, seed_objects) -> None:
        members = seed_objects[6]
        for m in members:
            assert m.pack_id == "semiconductors_pack_1"


class TestSemiconductorPackContent:
    """Validate specific semiconductor domain content."""

    def test_anchor_issuers_present(self, pack_data: dict) -> None:
        issuer_names = {i["canonical_name"] for i in pack_data["issuers"]}
        for expected in [
            "NVIDIA Corporation",
            "Taiwan Semiconductor Manufacturing",
            "Advanced Micro Devices",
            "Intel Corporation",
            "Micron Technology",
            "ASML Holding",
        ]:
            assert expected in issuer_names, f"Missing anchor: {expected}"

    def test_key_technologies_present(self, pack_data: dict) -> None:
        tech_names = {t["canonical_name"] for t in pack_data["technologies"]}
        assert "EUV Lithography" in tech_names
        assert "High Bandwidth Memory" in tech_names
        assert "CoWoS Packaging" in tech_names

    def test_key_themes_present(self, pack_data: dict) -> None:
        theme_names = {t["canonical_name"] for t in pack_data["themes"]}
        assert "AI Chip Demand" in theme_names
        assert "HBM Shortage" in theme_names

    def test_tsmc_supplies_nvidia(self, pack_data: dict) -> None:
        tsmc_nvidia = [
            r
            for r in pack_data["supply_chain"]
            if r["source"] == "Taiwan Semiconductor Manufacturing"
            and r["target"] == "NVIDIA Corporation"
        ]
        assert len(tsmc_nvidia) == 1
        assert tsmc_nvidia[0]["type"] == "supplies_to"

    def test_competition_edges_exist(self, pack_data: dict) -> None:
        compete = [r for r in pack_data["supply_chain"] if r["type"] == "competes_with"]
        assert len(compete) >= 3

    def test_korean_stocks_have_exchange(self, pack_data: dict) -> None:
        for issuer in pack_data["issuers"]:
            if issuer["ticker"].endswith(".KS"):
                assert issuer.get("exchange") == "KRX"

    def test_anchor_issuers_have_full_or_partial_coverage(self, pack_data: dict) -> None:
        for issuer in pack_data["issuers"]:
            if issuer.get("role") == "anchor":
                assert issuer["coverage_tier"] in ("full", "partial")


class TestPackAsTemplate:
    """Verify the pack works as a template for future expansion."""

    def test_all_objects_pass_dataclass_validation(self, seed_objects) -> None:
        """Every object was constructed without validation errors."""
        _, concepts, aliases, links, rels, profiles, members = seed_objects
        # If we got here without exceptions, all __post_init__ passed
        assert len(concepts) > 20  # issuers + securities + technologies + themes
        assert len(aliases) > 50  # multiple aliases per concept
        assert len(rels) > 10  # supply chain edges
        assert len(profiles) > 20  # one per non-security concept

    def test_build_is_pure_function(self, pack_data: dict) -> None:
        """Calling twice produces identical results."""
        objs1 = build_seed_objects(pack_data)
        objs2 = build_seed_objects(pack_data)
        # Same concept IDs
        ids1 = {c.concept_id for c in objs1[1]}
        ids2 = {c.concept_id for c in objs2[1]}
        assert ids1 == ids2
