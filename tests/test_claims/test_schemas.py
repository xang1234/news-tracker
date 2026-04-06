"""Tests for evidence claims schemas and key generation."""

import pathlib

import pytest

from src.claims.schemas import (
    VALID_CLAIM_STATUSES,
    VALID_EXTRACTION_METHODS,
    VALID_SOURCE_TYPES,
    EvidenceClaim,
    make_claim_id,
    make_claim_key,
)

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "migrations" / "023_evidence_claims.sql"
)


class TestMakeClaimKey:
    """Deterministic claim key generation."""

    def test_deterministic(self) -> None:
        k1 = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        k2 = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        assert k1 == k2

    def test_case_insensitive(self) -> None:
        k1 = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        k2 = make_claim_key("narrative", "doc_1", "tsmc", "supplies_to", "nvidia")
        assert k1 == k2

    def test_whitespace_trimmed(self) -> None:
        k1 = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        k2 = make_claim_key("narrative", "doc_1", " TSMC ", "supplies_to", " NVIDIA ")
        assert k1 == k2

    def test_different_sources_different_keys(self) -> None:
        k1 = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        k2 = make_claim_key("narrative", "doc_2", "TSMC", "supplies_to", "NVIDIA")
        assert k1 != k2

    def test_different_predicates_different_keys(self) -> None:
        k1 = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        k2 = make_claim_key("narrative", "doc_1", "TSMC", "competes_with", "NVIDIA")
        assert k1 != k2

    def test_different_lanes_different_keys(self) -> None:
        k1 = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        k2 = make_claim_key("filing", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        assert k1 != k2

    def test_none_object_handled(self) -> None:
        k = make_claim_key("narrative", "doc_1", "TSMC", "revenue_growth")
        assert k.startswith("clk_")

    def test_format(self) -> None:
        k = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        assert k.startswith("clk_")
        assert len(k) == len("clk_") + 16


class TestMakeClaimId:
    """Claim ID from claim key."""

    def test_deterministic(self) -> None:
        key = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        id1 = make_claim_id(key)
        id2 = make_claim_id(key)
        assert id1 == id2

    def test_format(self) -> None:
        cid = make_claim_id("clk_abc")
        assert cid.startswith("claim_")


class TestEvidenceClaim:
    """EvidenceClaim dataclass validation."""

    def _make_claim(self, **overrides) -> EvidenceClaim:
        key = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        defaults = {
            "claim_id": make_claim_id(key),
            "claim_key": key,
            "lane": "narrative",
            "source_id": "doc_1",
            "subject_text": "TSMC",
            "predicate": "supplies_to",
            "object_text": "NVIDIA",
            "contract_version": "0.1.0",
        }
        defaults.update(overrides)
        return EvidenceClaim(**defaults)

    def test_minimal_valid(self) -> None:
        claim = self._make_claim()
        assert claim.status == "active"
        assert claim.confidence == 0.5
        assert claim.extraction_method == "rule"

    def test_all_statuses_accepted(self) -> None:
        for s in VALID_CLAIM_STATUSES:
            claim = self._make_claim(status=s)
            assert claim.status == s

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid claim status"):
            self._make_claim(status="bad")

    def test_all_source_types_accepted(self) -> None:
        for st in VALID_SOURCE_TYPES:
            claim = self._make_claim(source_type=st)
            assert claim.source_type == st

    def test_invalid_source_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid source_type"):
            self._make_claim(source_type="bad")

    def test_all_extraction_methods_accepted(self) -> None:
        for em in VALID_EXTRACTION_METHODS:
            claim = self._make_claim(extraction_method=em)
            assert claim.extraction_method == em

    def test_invalid_extraction_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid extraction_method"):
            self._make_claim(extraction_method="bad")

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid lane"):
            self._make_claim(lane="bad")

    def test_with_source_span(self) -> None:
        claim = self._make_claim(
            source_span_start=100,
            source_span_end=200,
            source_text="TSMC supplies NVIDIA with chips",
        )
        assert claim.source_span_start == 100
        assert claim.source_span_end == 200

    def test_with_concept_ids(self) -> None:
        claim = self._make_claim(
            subject_concept_id="concept_issuer_abc",
            object_concept_id="concept_issuer_def",
        )
        assert claim.subject_concept_id == "concept_issuer_abc"

    def test_idempotent_key_generation(self) -> None:
        """Same inputs produce same claim_key → same claim_id."""
        k1 = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        k2 = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
        assert k1 == k2
        assert make_claim_id(k1) == make_claim_id(k2)


class TestMigration023:
    """Migration 023 structural checks."""

    @pytest.fixture()
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_evidence_claims(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS news_intel.evidence_claims" in sql

    def test_claim_key_unique(self, sql: str) -> None:
        assert "claim_key           TEXT NOT NULL UNIQUE" in sql

    def test_lane_check_constraint(self, sql: str) -> None:
        assert "'narrative'" in sql
        assert "'filing'" in sql

    def test_source_type_check(self, sql: str) -> None:
        assert "'document'" in sql
        assert "'filing_section'" in sql

    def test_extraction_method_check(self, sql: str) -> None:
        assert "'rule'" in sql
        assert "'llm'" in sql

    def test_status_check(self, sql: str) -> None:
        assert "'active'" in sql
        assert "'superseded'" in sql
        assert "'retracted'" in sql

    def test_concept_id_fks(self, sql: str) -> None:
        assert "REFERENCES concepts(concept_id)" in sql

    def test_bitemporal_fields(self, sql: str) -> None:
        assert "claim_valid_from" in sql
        assert "claim_valid_to" in sql
        assert "source_published_at" in sql

    def test_updated_at_trigger(self, sql: str) -> None:
        assert "update_evidence_claims_updated_at" in sql

    def test_indexes(self, sql: str) -> None:
        assert "idx_claims_lane_status" in sql
        assert "idx_claims_source" in sql
        assert "idx_claims_subject" in sql
        assert "idx_claims_predicate" in sql
