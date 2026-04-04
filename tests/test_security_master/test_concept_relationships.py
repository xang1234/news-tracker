"""Tests for concept relationships and theme/narrative links."""

import pathlib

import pytest

from src.security_master.concept_schemas import (
    VALID_RELATIONSHIP_TYPES,
    VALID_THEME_LINK_TYPES,
    ConceptRelationship,
    ConceptThemeLink,
)

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "migrations"
    / "020_concept_relationships.sql"
)


class TestConceptRelationship:
    """ConceptRelationship dataclass validation."""

    def test_minimal_valid(self) -> None:
        rel = ConceptRelationship(
            source_concept_id="concept_issuer_abc",
            target_concept_id="concept_issuer_def",
            relationship_type="supplies_to",
        )
        assert rel.confidence == 1.0
        assert rel.is_active is True
        assert rel.metadata == {}

    def test_all_relationship_types_accepted(self) -> None:
        for rt in VALID_RELATIONSHIP_TYPES:
            rel = ConceptRelationship(
                source_concept_id="s",
                target_concept_id="t",
                relationship_type=rt,
            )
            assert rel.relationship_type == rt

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid relationship_type"):
            ConceptRelationship(
                source_concept_id="s",
                target_concept_id="t",
                relationship_type="bad",
            )

    def test_with_confidence_and_attribution(self) -> None:
        rel = ConceptRelationship(
            source_concept_id="s",
            target_concept_id="t",
            relationship_type="subsidiary_of",
            confidence=0.85,
            source_attribution="10-K filing 2025",
        )
        assert rel.confidence == 0.85
        assert rel.source_attribution == "10-K filing 2025"


class TestConceptThemeLink:
    """ConceptThemeLink dataclass validation."""

    def test_minimal_valid(self) -> None:
        link = ConceptThemeLink(
            theme_concept_id="concept_theme_abc",
            linked_concept_id="concept_issuer_def",
        )
        assert link.link_type == "covers"
        assert link.relevance_score == 1.0

    def test_all_link_types_accepted(self) -> None:
        for lt in VALID_THEME_LINK_TYPES:
            link = ConceptThemeLink(
                theme_concept_id="t",
                linked_concept_id="l",
                link_type=lt,
            )
            assert link.link_type == lt

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid link_type"):
            ConceptThemeLink(
                theme_concept_id="t",
                linked_concept_id="l",
                link_type="bad",
            )

    def test_with_relevance_score(self) -> None:
        link = ConceptThemeLink(
            theme_concept_id="t",
            linked_concept_id="l",
            link_type="driven_by",
            relevance_score=0.7,
        )
        assert link.relevance_score == 0.7


class TestRelationshipTypeSemantics:
    """Verify relationship types cover required entity families."""

    def test_supply_chain_types(self) -> None:
        assert "supplies_to" in VALID_RELATIONSHIP_TYPES
        assert "customer_of" in VALID_RELATIONSHIP_TYPES

    def test_corporate_structure_types(self) -> None:
        assert "subsidiary_of" in VALID_RELATIONSHIP_TYPES
        assert "parent_of" in VALID_RELATIONSHIP_TYPES

    def test_technology_types(self) -> None:
        assert "uses_technology" in VALID_RELATIONSHIP_TYPES
        assert "develops_technology" in VALID_RELATIONSHIP_TYPES

    def test_product_types(self) -> None:
        assert "produces" in VALID_RELATIONSHIP_TYPES
        assert "consumes" in VALID_RELATIONSHIP_TYPES
        assert "component_of" in VALID_RELATIONSHIP_TYPES

    def test_facility_types(self) -> None:
        assert "operates_facility" in VALID_RELATIONSHIP_TYPES
        assert "located_at" in VALID_RELATIONSHIP_TYPES

    def test_competition_type(self) -> None:
        assert "competes_with" in VALID_RELATIONSHIP_TYPES

    def test_theme_link_types_cover_requirements(self) -> None:
        assert "covers" in VALID_THEME_LINK_TYPES
        assert "driven_by" in VALID_THEME_LINK_TYPES
        assert "impacts" in VALID_THEME_LINK_TYPES
        assert "monitors" in VALID_THEME_LINK_TYPES


class TestMigration020:
    """Migration 020 structural checks."""

    @pytest.fixture()
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_concept_relationships(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS concept_relationships" in sql

    def test_creates_concept_theme_links(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS concept_theme_links" in sql

    def test_relationship_type_check_constraint(self, sql: str) -> None:
        assert "'subsidiary_of'" in sql
        assert "'supplies_to'" in sql
        assert "'competes_with'" in sql
        assert "'uses_technology'" in sql

    def test_theme_link_type_check_constraint(self, sql: str) -> None:
        assert "'covers'" in sql
        assert "'driven_by'" in sql
        assert "'impacts'" in sql

    def test_composite_primary_keys(self, sql: str) -> None:
        assert "PRIMARY KEY (source_concept_id, target_concept_id, relationship_type)" in sql
        assert "PRIMARY KEY (theme_concept_id, linked_concept_id, link_type)" in sql

    def test_foreign_keys_reference_concepts(self, sql: str) -> None:
        assert "REFERENCES concepts(concept_id)" in sql

    def test_updated_at_trigger(self, sql: str) -> None:
        assert "update_concept_relationships_updated_at" in sql

    def test_uses_if_not_exists(self, sql: str) -> None:
        for line in sql.splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("CREATE TABLE ") and "IF NOT EXISTS" not in stripped:
                pytest.fail(f"CREATE TABLE without IF NOT EXISTS: {line.strip()}")
