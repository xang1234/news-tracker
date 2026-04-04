"""Tests for passage-to-concept mapping.

Verifies that source passages are mapped to distinct narrative_frame
and theme_concept identities, preserving traceability from raw text
through both concept layers.
"""

from __future__ import annotations

from pathlib import Path

from src.narrative.passage_mapping import (
    PassageMapper,
    PassageMapping,
    make_mapping_id,
)
from src.security_master.concept_schemas import make_concept_id

MIGRATION_PATH = Path("migrations/027_passage_mappings.sql")


# -- PassageMapping tests ---------------------------------------------------


class TestPassageMapping:
    """PassageMapping dataclass."""

    def test_valid_construction(self) -> None:
        m = PassageMapping(
            mapping_id="pmap_test",
            source_id="doc_1",
            narrative_frame_id="concept_narrative_frame_abc",
            theme_concept_id="concept_theme_xyz",
        )
        assert m.confidence == 1.0
        assert m.narrative_run_id is None

    def test_with_all_fields(self) -> None:
        m = PassageMapping(
            mapping_id="pmap_test",
            source_id="doc_1",
            narrative_frame_id="concept_nf_1",
            theme_concept_id="concept_th_1",
            narrative_frame_name="TSMC HBM Bottleneck",
            theme_concept_name="High Bandwidth Memory",
            source_span_start=100,
            source_span_end=250,
            passage_text="TSMC faces HBM3E expansion challenges...",
            narrative_run_id="nr_001",
            confidence=0.9,
            metadata={"extraction_method": "rule"},
        )
        assert m.passage_text is not None
        assert m.narrative_run_id == "nr_001"


# -- make_mapping_id tests -------------------------------------------------


class TestMakeMappingId:
    """Deterministic mapping ID generation."""

    def test_deterministic(self) -> None:
        id1 = make_mapping_id("doc_1", "frame_a", "theme_b")
        id2 = make_mapping_id("doc_1", "frame_a", "theme_b")
        assert id1 == id2

    def test_prefix(self) -> None:
        mid = make_mapping_id("doc_1", "frame_a", "theme_b")
        assert mid.startswith("pmap_")

    def test_different_inputs(self) -> None:
        id1 = make_mapping_id("doc_1", "frame_a", "theme_b")
        id2 = make_mapping_id("doc_1", "frame_a", "theme_c")
        assert id1 != id2

    def test_different_sources(self) -> None:
        id1 = make_mapping_id("doc_1", "frame_a", "theme_b")
        id2 = make_mapping_id("doc_2", "frame_a", "theme_b")
        assert id1 != id2


# -- PassageMapper tests ---------------------------------------------------


class TestPassageMapper:
    """Mapping passages to frame and theme concepts."""

    def test_map_passage_basic(self) -> None:
        mapper = PassageMapper()
        mapping = mapper.map_passage(
            source_id="doc_1",
            frame_name="TSMC HBM Expansion Bottleneck",
            theme_name="High Bandwidth Memory",
        )
        assert mapping.source_id == "doc_1"
        assert mapping.narrative_frame_name == "TSMC HBM Expansion Bottleneck"
        assert mapping.theme_concept_name == "High Bandwidth Memory"
        assert mapping.narrative_frame_id.startswith("concept_narrative_frame_")
        assert mapping.theme_concept_id.startswith("concept_theme_")

    def test_frame_and_theme_are_distinct(self) -> None:
        """Frame and theme concept IDs must be different."""
        mapper = PassageMapper()
        mapping = mapper.map_passage(
            source_id="doc_1",
            frame_name="TSMC HBM Expansion Bottleneck",
            theme_name="High Bandwidth Memory",
        )
        assert mapping.narrative_frame_id != mapping.theme_concept_id

    def test_deterministic_concept_ids(self) -> None:
        mapper = PassageMapper()
        m1 = mapper.map_passage(
            source_id="doc_1",
            frame_name="Supply Chain Risk",
            theme_name="Semiconductors",
        )
        m2 = mapper.map_passage(
            source_id="doc_2",
            frame_name="Supply Chain Risk",
            theme_name="Semiconductors",
        )
        # Same frame/theme names → same concept IDs
        assert m1.narrative_frame_id == m2.narrative_frame_id
        assert m1.theme_concept_id == m2.theme_concept_id
        # Different source → different mapping IDs
        assert m1.mapping_id != m2.mapping_id

    def test_map_passage_with_span(self) -> None:
        mapper = PassageMapper()
        mapping = mapper.map_passage(
            source_id="doc_1",
            frame_name="NVIDIA Revenue Beat",
            theme_name="AI Infrastructure",
            passage_text="NVIDIA reported revenue of $26B...",
            source_span_start=0,
            source_span_end=34,
        )
        assert mapping.passage_text is not None
        assert mapping.source_span_start == 0
        assert mapping.source_span_end == 34

    def test_map_passage_with_run_id(self) -> None:
        mapper = PassageMapper()
        mapping = mapper.map_passage(
            source_id="doc_1",
            frame_name="Frame",
            theme_name="Theme",
            narrative_run_id="nr_001",
        )
        assert mapping.narrative_run_id == "nr_001"

    def test_map_passage_confidence(self) -> None:
        mapper = PassageMapper()
        mapping = mapper.map_passage(
            source_id="doc_1",
            frame_name="F",
            theme_name="T",
            confidence=0.7,
        )
        assert mapping.confidence == 0.7

    def test_map_from_narrative_run(self) -> None:
        mapper = PassageMapper()
        mapping = mapper.map_from_narrative_run(
            source_id="doc_1",
            frame_name="HBM Supply Surge",
            theme_name="High Bandwidth Memory",
            narrative_run_id="nr_001",
            passage_text="Samsung HBM3E production...",
        )
        assert mapping.narrative_run_id == "nr_001"
        assert mapping.passage_text is not None
        assert mapping.narrative_frame_id.startswith("concept_narrative_frame_")

    def test_resolve_frame_concept_id(self) -> None:
        cid = PassageMapper.resolve_frame_concept_id("TSMC HBM Bottleneck")
        expected = make_concept_id("narrative_frame", "TSMC HBM Bottleneck")
        assert cid == expected

    def test_resolve_theme_concept_id(self) -> None:
        cid = PassageMapper.resolve_theme_concept_id("High Bandwidth Memory")
        expected = make_concept_id("theme", "High Bandwidth Memory")
        assert cid == expected

    def test_same_passage_same_mapping_id(self) -> None:
        """Idempotent: same inputs → same mapping ID."""
        mapper = PassageMapper()
        m1 = mapper.map_passage(
            source_id="doc_1",
            frame_name="F1",
            theme_name="T1",
        )
        m2 = mapper.map_passage(
            source_id="doc_1",
            frame_name="F1",
            theme_name="T1",
        )
        assert m1.mapping_id == m2.mapping_id


# -- Migration structural tests -------------------------------------------


class TestMigration027:
    """Structural validation of migration 027."""

    @staticmethod
    def _load_sql() -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_passage_mappings_table(self) -> None:
        sql = self._load_sql()
        assert "CREATE TABLE IF NOT EXISTS news_intel.passage_mappings" in sql

    def test_concept_fks(self) -> None:
        sql = self._load_sql()
        assert "REFERENCES concepts(concept_id)" in sql

    def test_source_index(self) -> None:
        sql = self._load_sql()
        assert "idx_passage_map_source" in sql

    def test_frame_index(self) -> None:
        sql = self._load_sql()
        assert "idx_passage_map_frame" in sql

    def test_theme_index(self) -> None:
        sql = self._load_sql()
        assert "idx_passage_map_theme" in sql

    def test_run_index(self) -> None:
        sql = self._load_sql()
        assert "idx_passage_map_run" in sql

    def test_confidence_column(self) -> None:
        sql = self._load_sql()
        assert "confidence" in sql
