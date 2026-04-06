"""Tests for coverage profile and domain pack schemas."""

import pathlib

import pytest

from src.coverage.schemas import (
    VALID_COVERAGE_TIERS,
    VALID_PACK_ROLES,
    CoverageProfile,
    CoverageTierChange,
    DomainPack,
    DomainPackMember,
)

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "migrations" / "021_coverage_profiles.sql"
)


class TestCoverageProfile:
    """CoverageProfile dataclass validation."""

    def test_minimal_valid(self) -> None:
        p = CoverageProfile(concept_id="concept_issuer_abc")
        assert p.coverage_tier == "stub"
        assert p.structural_completeness == 0.0
        assert p.filing_coverage is False
        assert p.narrative_coverage is False
        assert p.graph_coverage is False

    def test_all_tiers_accepted(self) -> None:
        for tier in VALID_COVERAGE_TIERS:
            p = CoverageProfile(concept_id="c1", coverage_tier=tier)
            assert p.coverage_tier == tier

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid coverage_tier"):
            CoverageProfile(concept_id="c1", coverage_tier="bad")

    def test_full_profile(self) -> None:
        p = CoverageProfile(
            concept_id="c1",
            coverage_tier="full",
            structural_completeness=0.95,
            filing_coverage=True,
            narrative_coverage=True,
            graph_coverage=True,
            coverage_notes="Fully covered with SEC filings",
        )
        assert p.structural_completeness == 0.95
        assert p.filing_coverage is True


class TestCoverageTierChange:
    """CoverageTierChange dataclass validation."""

    def test_minimal_valid(self) -> None:
        c = CoverageTierChange(concept_id="c1", coverage_tier="partial")
        assert c.valid_to is None
        assert c.change_reason == ""

    def test_all_tiers_accepted(self) -> None:
        for tier in VALID_COVERAGE_TIERS:
            c = CoverageTierChange(concept_id="c1", coverage_tier=tier)
            assert c.coverage_tier == tier

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid coverage_tier"):
            CoverageTierChange(concept_id="c1", coverage_tier="bad")

    def test_with_attribution(self) -> None:
        c = CoverageTierChange(
            concept_id="c1",
            coverage_tier="full",
            changed_by="filing_lane",
            change_reason="10-K filing data now available",
        )
        assert c.changed_by == "filing_lane"


class TestDomainPack:
    """DomainPack dataclass."""

    def test_minimal_valid(self) -> None:
        p = DomainPack(pack_id="semi_pack_1", name="Semiconductors Pack 1")
        assert p.version == "1.0"
        assert p.is_active is True

    def test_with_description(self) -> None:
        p = DomainPack(
            pack_id="semi_pack_1",
            name="Semiconductors Pack 1",
            description="Top-tier semiconductor issuers with full structural coverage",
            version="2.0",
        )
        assert p.version == "2.0"


class TestDomainPackMember:
    """DomainPackMember dataclass validation."""

    def test_minimal_valid(self) -> None:
        m = DomainPackMember(pack_id="p1", concept_id="c1")
        assert m.role == "member"

    def test_all_roles_accepted(self) -> None:
        for role in VALID_PACK_ROLES:
            m = DomainPackMember(pack_id="p1", concept_id="c1", role=role)
            assert m.role == role

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid role"):
            DomainPackMember(pack_id="p1", concept_id="c1", role="bad")


class TestCoverageTierSemantics:
    """Verify tier semantics match requirements."""

    def test_tiers_ordered_by_maturity(self) -> None:
        assert "full" in VALID_COVERAGE_TIERS
        assert "partial" in VALID_COVERAGE_TIERS
        assert "stub" in VALID_COVERAGE_TIERS
        assert "unsupported" in VALID_COVERAGE_TIERS

    def test_pack_roles_include_hierarchy(self) -> None:
        assert "anchor" in VALID_PACK_ROLES
        assert "member" in VALID_PACK_ROLES
        assert "peripheral" in VALID_PACK_ROLES


class TestMigration021:
    """Migration 021 structural checks."""

    @pytest.fixture()
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_coverage_profiles(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS coverage_profiles" in sql

    def test_creates_coverage_tier_history(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS coverage_tier_history" in sql

    def test_creates_domain_packs(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS domain_packs" in sql

    def test_creates_domain_pack_members(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS domain_pack_members" in sql

    def test_tier_check_constraint(self, sql: str) -> None:
        assert "'full'" in sql
        assert "'partial'" in sql
        assert "'stub'" in sql
        assert "'unsupported'" in sql

    def test_role_check_constraint(self, sql: str) -> None:
        assert "'anchor'" in sql
        assert "'member'" in sql
        assert "'peripheral'" in sql

    def test_foreign_keys(self, sql: str) -> None:
        assert "REFERENCES concepts(concept_id)" in sql
        assert "REFERENCES domain_packs(pack_id)" in sql

    def test_history_has_validity_window(self, sql: str) -> None:
        assert "valid_from" in sql
        assert "valid_to" in sql

    def test_updated_at_triggers(self, sql: str) -> None:
        assert "update_coverage_profiles_updated_at" in sql
        assert "update_domain_packs_updated_at" in sql

    def test_uses_if_not_exists(self, sql: str) -> None:
        for line in sql.splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("CREATE TABLE ") and "IF NOT EXISTS" not in stripped:
                pytest.fail(f"CREATE TABLE without IF NOT EXISTS: {line.strip()}")
