"""Tests for NER configuration."""

import os
from pathlib import Path

import pytest

from src.ner.config import NERConfig


class TestNERConfig:
    """Tests for NERConfig settings."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = NERConfig()

        assert config.spacy_model == "en_core_web_trf"
        assert config.fallback_model == "en_core_web_sm"
        assert config.fuzzy_threshold == 85
        assert config.enable_coreference is True
        assert config.batch_size == 32
        assert config.max_text_length == 10000
        assert config.confidence_threshold == 0.5

    def test_patterns_dir_exists(self):
        """Patterns directory should be within the NER package."""
        config = NERConfig()

        assert isinstance(config.patterns_dir, Path)
        assert "ner" in str(config.patterns_dir)
        assert "patterns" in str(config.patterns_dir)

    def test_extract_types_default(self):
        """Should include all entity types by default."""
        config = NERConfig()

        assert "TICKER" in config.extract_types
        assert "COMPANY" in config.extract_types
        assert "PRODUCT" in config.extract_types
        assert "TECHNOLOGY" in config.extract_types
        assert "METRIC" in config.extract_types

    def test_env_override(self, monkeypatch):
        """Should allow environment variable overrides."""
        monkeypatch.setenv("NER_SPACY_MODEL", "en_core_web_sm")
        monkeypatch.setenv("NER_FUZZY_THRESHOLD", "90")
        monkeypatch.setenv("NER_ENABLE_COREFERENCE", "false")

        config = NERConfig()

        assert config.spacy_model == "en_core_web_sm"
        assert config.fuzzy_threshold == 90
        assert config.enable_coreference is False

    def test_fuzzy_threshold_bounds(self):
        """Fuzzy threshold should be between 0 and 100."""
        # Valid values
        config = NERConfig(fuzzy_threshold=0)
        assert config.fuzzy_threshold == 0

        config = NERConfig(fuzzy_threshold=100)
        assert config.fuzzy_threshold == 100

        # Invalid values should raise
        with pytest.raises(ValueError):
            NERConfig(fuzzy_threshold=-1)

        with pytest.raises(ValueError):
            NERConfig(fuzzy_threshold=101)

    def test_confidence_threshold_bounds(self):
        """Confidence threshold should be between 0.0 and 1.0."""
        config = NERConfig(confidence_threshold=0.0)
        assert config.confidence_threshold == 0.0

        config = NERConfig(confidence_threshold=1.0)
        assert config.confidence_threshold == 1.0

        with pytest.raises(ValueError):
            NERConfig(confidence_threshold=-0.1)

        with pytest.raises(ValueError):
            NERConfig(confidence_threshold=1.1)

    def test_batch_size_bounds(self):
        """Batch size should be between 1 and 256."""
        config = NERConfig(batch_size=1)
        assert config.batch_size == 1

        config = NERConfig(batch_size=256)
        assert config.batch_size == 256

        with pytest.raises(ValueError):
            NERConfig(batch_size=0)

        with pytest.raises(ValueError):
            NERConfig(batch_size=257)

    def test_semantic_linking_defaults(self):
        """Semantic linking should be disabled by default with sensible thresholds."""
        config = NERConfig()

        assert config.enable_semantic_linking is False
        assert config.semantic_similarity_threshold == 0.5
        assert config.semantic_base_score == 0.6

    def test_semantic_similarity_threshold_bounds(self):
        """Semantic similarity threshold should be between 0.0 and 1.0."""
        config = NERConfig(semantic_similarity_threshold=0.0)
        assert config.semantic_similarity_threshold == 0.0

        config = NERConfig(semantic_similarity_threshold=1.0)
        assert config.semantic_similarity_threshold == 1.0

        with pytest.raises(ValueError):
            NERConfig(semantic_similarity_threshold=-0.1)

        with pytest.raises(ValueError):
            NERConfig(semantic_similarity_threshold=1.1)

    def test_semantic_base_score_bounds(self):
        """Semantic base score should be between 0.0 and 1.0."""
        config = NERConfig(semantic_base_score=0.0)
        assert config.semantic_base_score == 0.0

        config = NERConfig(semantic_base_score=1.0)
        assert config.semantic_base_score == 1.0

        with pytest.raises(ValueError):
            NERConfig(semantic_base_score=-0.1)

        with pytest.raises(ValueError):
            NERConfig(semantic_base_score=1.1)

    def test_semantic_linking_env_override(self, monkeypatch):
        """Semantic linking settings should be overridable via env vars."""
        monkeypatch.setenv("NER_ENABLE_SEMANTIC_LINKING", "true")
        monkeypatch.setenv("NER_SEMANTIC_SIMILARITY_THRESHOLD", "0.7")
        monkeypatch.setenv("NER_SEMANTIC_BASE_SCORE", "0.8")

        config = NERConfig()

        assert config.enable_semantic_linking is True
        assert config.semantic_similarity_threshold == 0.7
        assert config.semantic_base_score == 0.8
