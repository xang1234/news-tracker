"""Tests for ThemeCluster dataclass."""

import numpy as np
import pytest

from src.clustering.schemas import ThemeCluster


class TestThemeClusterToDict:
    """Tests for ThemeCluster serialization."""

    def test_to_dict_contains_all_fields(self):
        """Should include all fields in the serialized dict."""
        theme = ThemeCluster(
            theme_id="theme_abc123",
            name="gpu_nvidia_architecture",
            topic_words=[("gpu", 0.15), ("nvidia", 0.12)],
            centroid=np.array([1.0, 2.0, 3.0]),
            document_count=10,
            document_ids=["doc_001", "doc_002"],
            metadata={"bertopic_topic_id": 0},
        )
        result = theme.to_dict()

        assert result["theme_id"] == "theme_abc123"
        assert result["name"] == "gpu_nvidia_architecture"
        assert result["topic_words"] == [("gpu", 0.15), ("nvidia", 0.12)]
        assert result["centroid"] == [1.0, 2.0, 3.0]
        assert result["document_count"] == 10
        assert result["document_ids"] == ["doc_001", "doc_002"]
        assert result["metadata"] == {"bertopic_topic_id": 0}
        assert "created_at" in result

    def test_to_dict_centroid_is_list(self):
        """Centroid ndarray should be converted to a plain list."""
        theme = ThemeCluster(
            theme_id="theme_x",
            name="test",
            topic_words=[("a", 0.1)],
            centroid=np.zeros(768),
            document_count=1,
        )
        result = theme.to_dict()

        assert isinstance(result["centroid"], list)
        assert len(result["centroid"]) == 768


class TestThemeClusterFromDict:
    """Tests for ThemeCluster deserialization."""

    def test_from_dict_roundtrip(self):
        """to_dict → from_dict should preserve all fields."""
        original = ThemeCluster(
            theme_id="theme_abc123",
            name="gpu_nvidia",
            topic_words=[("gpu", 0.15), ("nvidia", 0.12)],
            centroid=np.array([1.0, 2.0, 3.0]),
            document_count=5,
            document_ids=["doc_1"],
            metadata={"bertopic_topic_id": 0},
        )
        data = original.to_dict()
        restored = ThemeCluster.from_dict(data)

        assert restored.theme_id == original.theme_id
        assert restored.name == original.name
        assert restored.topic_words == original.topic_words
        np.testing.assert_array_almost_equal(restored.centroid, original.centroid)
        assert restored.document_count == original.document_count
        assert restored.document_ids == original.document_ids
        assert restored.metadata == original.metadata

    def test_from_dict_missing_optional_fields(self):
        """Should use defaults for missing optional fields."""
        data = {
            "theme_id": "theme_x",
            "name": "test",
            "topic_words": [("word", 0.1)],
            "centroid": [0.0, 0.0],
            "document_count": 1,
        }
        theme = ThemeCluster.from_dict(data)

        assert theme.document_ids == []
        assert theme.metadata == {}

    def test_from_dict_parses_iso_datetime(self):
        """Should parse ISO format datetime strings."""
        data = {
            "theme_id": "theme_x",
            "name": "test",
            "topic_words": [("word", 0.1)],
            "centroid": [0.0],
            "document_count": 1,
            "created_at": "2025-01-15T10:30:00+00:00",
        }
        theme = ThemeCluster.from_dict(data)

        assert theme.created_at.year == 2025
        assert theme.created_at.month == 1


class TestThemeClusterEquality:
    """Tests for equality and hashing based on theme_id."""

    def test_equal_by_theme_id(self):
        """Two ThemeClusters with same theme_id should be equal."""
        t1 = ThemeCluster(
            theme_id="theme_abc",
            name="one",
            topic_words=[("a", 0.1)],
            centroid=np.zeros(3),
            document_count=5,
        )
        t2 = ThemeCluster(
            theme_id="theme_abc",
            name="two",
            topic_words=[("b", 0.2)],
            centroid=np.ones(3),
            document_count=10,
        )
        assert t1 == t2

    def test_not_equal_different_id(self):
        """ThemeClusters with different theme_ids should not be equal."""
        t1 = ThemeCluster(
            theme_id="theme_abc",
            name="same",
            topic_words=[("a", 0.1)],
            centroid=np.zeros(3),
            document_count=5,
        )
        t2 = ThemeCluster(
            theme_id="theme_xyz",
            name="same",
            topic_words=[("a", 0.1)],
            centroid=np.zeros(3),
            document_count=5,
        )
        assert t1 != t2

    def test_hash_same_id(self):
        """Same theme_id should produce same hash."""
        t1 = ThemeCluster(
            theme_id="theme_abc",
            name="one",
            topic_words=[],
            centroid=np.zeros(1),
            document_count=1,
        )
        t2 = ThemeCluster(
            theme_id="theme_abc",
            name="two",
            topic_words=[],
            centroid=np.ones(1),
            document_count=2,
        )
        assert hash(t1) == hash(t2)

    def test_usable_in_set(self):
        """ThemeClusters with same ID should deduplicate in sets."""
        t1 = ThemeCluster(
            theme_id="theme_abc",
            name="one",
            topic_words=[],
            centroid=np.zeros(1),
            document_count=1,
        )
        t2 = ThemeCluster(
            theme_id="theme_abc",
            name="two",
            topic_words=[],
            centroid=np.ones(1),
            document_count=2,
        )
        assert len({t1, t2}) == 1

    def test_not_equal_to_non_theme(self):
        """Comparison with non-ThemeCluster should return NotImplemented."""
        theme = ThemeCluster(
            theme_id="theme_abc",
            name="test",
            topic_words=[],
            centroid=np.zeros(1),
            document_count=1,
        )
        assert theme != "theme_abc"
        assert theme != 42


class TestGenerateThemeId:
    """Tests for deterministic theme ID generation."""

    def test_deterministic(self):
        """Same words should always produce the same ID."""
        words = [("gpu", 0.15), ("nvidia", 0.12), ("architecture", 0.10)]
        id1 = ThemeCluster.generate_theme_id(words)
        id2 = ThemeCluster.generate_theme_id(words)
        assert id1 == id2

    def test_order_independent(self):
        """Different word order should produce the same ID (sorted internally)."""
        words_a = [("gpu", 0.15), ("nvidia", 0.12)]
        words_b = [("nvidia", 0.12), ("gpu", 0.15)]
        assert ThemeCluster.generate_theme_id(words_a) == ThemeCluster.generate_theme_id(words_b)

    def test_score_independent(self):
        """Different scores for same words should produce the same ID."""
        words_a = [("gpu", 0.15), ("nvidia", 0.12)]
        words_b = [("gpu", 0.99), ("nvidia", 0.01)]
        assert ThemeCluster.generate_theme_id(words_a) == ThemeCluster.generate_theme_id(words_b)

    def test_different_words_different_id(self):
        """Different word sets should produce different IDs."""
        words_a = [("gpu", 0.15), ("nvidia", 0.12)]
        words_b = [("memory", 0.18), ("hbm3e", 0.14)]
        assert ThemeCluster.generate_theme_id(words_a) != ThemeCluster.generate_theme_id(words_b)

    def test_format(self):
        """ID should match theme_{12_hex_chars} format."""
        words = [("test", 0.1)]
        theme_id = ThemeCluster.generate_theme_id(words)
        assert theme_id.startswith("theme_")
        assert len(theme_id) == len("theme_") + 12


class TestGenerateName:
    """Tests for human-readable name generation."""

    def test_default_top_3(self):
        """Should join top 3 words with underscores by default."""
        words = [("gpu", 0.15), ("nvidia", 0.12), ("architecture", 0.10), ("ai", 0.08)]
        name = ThemeCluster.generate_name(words)
        assert name == "gpu_nvidia_architecture"

    def test_custom_top_n(self):
        """Should respect custom top_n parameter."""
        words = [("gpu", 0.15), ("nvidia", 0.12), ("arch", 0.10)]
        assert ThemeCluster.generate_name(words, top_n=2) == "gpu_nvidia"
        assert ThemeCluster.generate_name(words, top_n=1) == "gpu"

    def test_fewer_words_than_top_n(self):
        """Should handle fewer words than top_n without error."""
        words = [("gpu", 0.15)]
        assert ThemeCluster.generate_name(words, top_n=3) == "gpu"

    def test_empty_words(self):
        """Should return empty string for empty word list."""
        assert ThemeCluster.generate_name([]) == ""
