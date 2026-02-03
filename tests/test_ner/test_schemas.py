"""Tests for NER schemas."""

import pytest

from src.ner.schemas import FinancialEntity


class TestFinancialEntity:
    """Tests for FinancialEntity dataclass."""

    def test_create_entity(self):
        """Should create entity with required fields."""
        entity = FinancialEntity(
            text="Nvidia",
            type="COMPANY",
            normalized="NVIDIA",
            start=0,
            end=6,
        )

        assert entity.text == "Nvidia"
        assert entity.type == "COMPANY"
        assert entity.normalized == "NVIDIA"
        assert entity.start == 0
        assert entity.end == 6
        assert entity.confidence == 1.0  # Default
        assert entity.metadata == {}  # Default

    def test_create_entity_with_metadata(self):
        """Should create entity with metadata."""
        entity = FinancialEntity(
            text="$NVDA",
            type="TICKER",
            normalized="NVDA",
            start=0,
            end=5,
            confidence=0.99,
            metadata={"source": "cashtag"},
        )

        assert entity.confidence == 0.99
        assert entity.metadata["source"] == "cashtag"

    def test_to_dict(self, sample_entity):
        """Should serialize to dictionary."""
        d = sample_entity.to_dict()

        assert d["text"] == "Nvidia"
        assert d["type"] == "COMPANY"
        assert d["normalized"] == "NVIDIA"
        assert d["start"] == 0
        assert d["end"] == 6
        assert d["confidence"] == 0.95
        assert d["metadata"]["ticker"] == "NVDA"

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "text": "H100",
            "type": "PRODUCT",
            "normalized": "H100",
            "start": 10,
            "end": 14,
            "confidence": 0.9,
            "metadata": {"category": "GPU"},
        }

        entity = FinancialEntity.from_dict(d)

        assert entity.text == "H100"
        assert entity.type == "PRODUCT"
        assert entity.normalized == "H100"
        assert entity.start == 10
        assert entity.end == 14
        assert entity.confidence == 0.9
        assert entity.metadata["category"] == "GPU"

    def test_from_dict_defaults(self):
        """Should use defaults for optional fields."""
        d = {
            "text": "HBM3",
            "type": "TECHNOLOGY",
            "normalized": "HBM3",
            "start": 0,
            "end": 4,
        }

        entity = FinancialEntity.from_dict(d)

        assert entity.confidence == 1.0
        assert entity.metadata == {}

    def test_from_dict_missing_required(self):
        """Should raise on missing required fields."""
        d = {"text": "Nvidia", "type": "COMPANY"}

        with pytest.raises(KeyError):
            FinancialEntity.from_dict(d)

    def test_equality(self):
        """Entities with same normalized and type should be equal."""
        e1 = FinancialEntity(
            text="Nvidia",
            type="COMPANY",
            normalized="NVIDIA",
            start=0,
            end=6,
        )
        e2 = FinancialEntity(
            text="nvidia",
            type="COMPANY",
            normalized="NVIDIA",
            start=10,
            end=16,
        )

        assert e1 == e2

    def test_inequality(self):
        """Entities with different normalized or type should not be equal."""
        e1 = FinancialEntity(
            text="Nvidia",
            type="COMPANY",
            normalized="NVIDIA",
            start=0,
            end=6,
        )
        e2 = FinancialEntity(
            text="NVDA",
            type="TICKER",
            normalized="NVDA",
            start=0,
            end=4,
        )

        assert e1 != e2

    def test_hash_for_deduplication(self):
        """Should be hashable for set deduplication."""
        e1 = FinancialEntity(
            text="Nvidia",
            type="COMPANY",
            normalized="NVIDIA",
            start=0,
            end=6,
        )
        e2 = FinancialEntity(
            text="nvidia corp",
            type="COMPANY",
            normalized="NVIDIA",
            start=10,
            end=21,
        )

        entities = {e1, e2}
        assert len(entities) == 1  # Deduplicated

    def test_overlaps_true(self):
        """Should detect overlapping spans."""
        e1 = FinancialEntity(
            text="Taiwan Semiconductor",
            type="COMPANY",
            normalized="TSM",
            start=0,
            end=20,
        )
        e2 = FinancialEntity(
            text="Semiconductor",
            type="COMPANY",
            normalized="SEMI",
            start=7,
            end=20,
        )

        assert e1.overlaps(e2)
        assert e2.overlaps(e1)

    def test_overlaps_false(self):
        """Should not detect non-overlapping spans."""
        e1 = FinancialEntity(
            text="Nvidia",
            type="COMPANY",
            normalized="NVDA",
            start=0,
            end=6,
        )
        e2 = FinancialEntity(
            text="AMD",
            type="COMPANY",
            normalized="AMD",
            start=20,
            end=23,
        )

        assert not e1.overlaps(e2)
        assert not e2.overlaps(e1)

    def test_overlaps_adjacent(self):
        """Adjacent spans should not overlap."""
        e1 = FinancialEntity(
            text="Taiwan",
            type="COMPANY",
            normalized="TAIWAN",
            start=0,
            end=6,
        )
        e2 = FinancialEntity(
            text="Semiconductor",
            type="COMPANY",
            normalized="SEMI",
            start=7,
            end=20,
        )

        assert not e1.overlaps(e2)
