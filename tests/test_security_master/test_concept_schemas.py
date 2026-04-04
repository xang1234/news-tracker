"""Tests for concept registry schemas."""

import pytest

from src.security_master.concept_schemas import (
    VALID_ALIAS_TYPES,
    VALID_CONCEPT_TYPES,
    VALID_ISSUER_SECURITY_RELATIONSHIPS,
    Concept,
    ConceptAlias,
    IssuerSecurityLink,
    make_concept_id,
)


class TestMakeConceptId:
    """Deterministic concept ID generation."""

    def test_deterministic(self) -> None:
        id1 = make_concept_id("issuer", "Taiwan Semiconductor")
        id2 = make_concept_id("issuer", "Taiwan Semiconductor")
        assert id1 == id2

    def test_case_insensitive(self) -> None:
        id1 = make_concept_id("issuer", "TSMC")
        id2 = make_concept_id("issuer", "tsmc")
        assert id1 == id2

    def test_strips_whitespace(self) -> None:
        id1 = make_concept_id("issuer", "TSMC")
        id2 = make_concept_id("issuer", "  TSMC  ")
        assert id1 == id2

    def test_different_types_different_ids(self) -> None:
        issuer_id = make_concept_id("issuer", "TSMC")
        security_id = make_concept_id("security", "TSMC")
        assert issuer_id != security_id

    def test_format(self) -> None:
        cid = make_concept_id("issuer", "TSMC")
        assert cid.startswith("concept_issuer_")
        assert len(cid) == len("concept_issuer_") + 12

    def test_different_names_different_ids(self) -> None:
        id1 = make_concept_id("issuer", "Samsung")
        id2 = make_concept_id("issuer", "TSMC")
        assert id1 != id2


class TestConcept:
    """Concept dataclass validation."""

    def test_minimal_valid(self) -> None:
        c = Concept(
            concept_id="concept_issuer_abc123",
            concept_type="issuer",
            canonical_name="Taiwan Semiconductor",
        )
        assert c.is_active is True
        assert c.metadata == {}

    def test_all_concept_types_accepted(self) -> None:
        for ct in VALID_CONCEPT_TYPES:
            c = Concept(
                concept_id=f"concept_{ct}_test",
                concept_type=ct,
                canonical_name="Test",
            )
            assert c.concept_type == ct

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid concept_type"):
            Concept(
                concept_id="test",
                concept_type="bad_type",
                canonical_name="Test",
            )


class TestConceptAlias:
    """ConceptAlias dataclass validation."""

    def test_minimal_valid(self) -> None:
        a = ConceptAlias(alias="TSMC", concept_id="concept_issuer_abc")
        assert a.alias_type == "name"
        assert a.is_primary is False

    def test_all_alias_types_accepted(self) -> None:
        for at in VALID_ALIAS_TYPES:
            a = ConceptAlias(
                alias="test", concept_id="c1", alias_type=at
            )
            assert a.alias_type == at

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid alias_type"):
            ConceptAlias(alias="test", concept_id="c1", alias_type="bad")


class TestIssuerSecurityLink:
    """IssuerSecurityLink dataclass validation."""

    def test_minimal_valid(self) -> None:
        link = IssuerSecurityLink(
            issuer_concept_id="issuer_1",
            security_concept_id="security_1",
        )
        assert link.relationship_type == "primary"

    def test_all_relationship_types_accepted(self) -> None:
        for rt in VALID_ISSUER_SECURITY_RELATIONSHIPS:
            link = IssuerSecurityLink(
                issuer_concept_id="i1",
                security_concept_id="s1",
                relationship_type=rt,
            )
            assert link.relationship_type == rt

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid relationship_type"):
            IssuerSecurityLink(
                issuer_concept_id="i1",
                security_concept_id="s1",
                relationship_type="bad",
            )


class TestMigrationConsistency:
    """Verify schema constants match migration CHECK constraints."""

    def test_concept_types_include_core_types(self) -> None:
        assert "issuer" in VALID_CONCEPT_TYPES
        assert "security" in VALID_CONCEPT_TYPES
        assert "technology" in VALID_CONCEPT_TYPES
        assert "theme" in VALID_CONCEPT_TYPES
        assert "narrative_frame" in VALID_CONCEPT_TYPES

    def test_alias_types_include_ticker(self) -> None:
        assert "ticker" in VALID_ALIAS_TYPES
        assert "name" in VALID_ALIAS_TYPES

    def test_relationship_types_include_primary(self) -> None:
        assert "primary" in VALID_ISSUER_SECURITY_RELATIONSHIPS
        assert "adr" in VALID_ISSUER_SECURITY_RELATIONSHIPS
