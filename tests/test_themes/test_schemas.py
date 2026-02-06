"""Tests for Theme and ThemeMetrics dataclasses."""

from datetime import date, datetime, timezone

import numpy as np
import pytest

from src.themes.schemas import VALID_LIFECYCLE_STAGES, Theme, ThemeMetrics


class TestThemeInit:
    """Test Theme construction and validation."""

    def test_valid_lifecycle_stages(self) -> None:
        """All four valid stages should be accepted."""
        for stage in ("emerging", "accelerating", "mature", "fading"):
            theme = Theme(
                theme_id="theme_test",
                name="test",
                centroid=np.zeros(768),
                lifecycle_stage=stage,
            )
            assert theme.lifecycle_stage == stage

    def test_invalid_lifecycle_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid lifecycle_stage"):
            Theme(
                theme_id="theme_test",
                name="test",
                centroid=np.zeros(768),
                lifecycle_stage="invalid",
            )

    def test_defaults(self) -> None:
        theme = Theme(
            theme_id="theme_test",
            name="test",
            centroid=np.zeros(768),
        )
        assert theme.lifecycle_stage == "emerging"
        assert theme.document_count == 0
        assert theme.top_keywords == []
        assert theme.top_tickers == []
        assert theme.top_entities == []
        assert theme.metadata == {}
        assert theme.description is None
        assert isinstance(theme.created_at, datetime)
        assert isinstance(theme.updated_at, datetime)

    def test_valid_lifecycle_stages_constant(self) -> None:
        assert VALID_LIFECYCLE_STAGES == {
            "emerging", "accelerating", "mature", "fading",
        }


class TestThemeEquality:
    """Test __eq__ and __hash__."""

    def test_equal_by_theme_id(self) -> None:
        a = Theme(theme_id="theme_abc", name="a", centroid=np.zeros(768))
        b = Theme(theme_id="theme_abc", name="b", centroid=np.ones(768))
        assert a == b

    def test_not_equal_different_id(self) -> None:
        a = Theme(theme_id="theme_abc", name="x", centroid=np.zeros(768))
        b = Theme(theme_id="theme_xyz", name="x", centroid=np.zeros(768))
        assert a != b

    def test_hash_consistent(self) -> None:
        a = Theme(theme_id="theme_abc", name="a", centroid=np.zeros(768))
        b = Theme(theme_id="theme_abc", name="b", centroid=np.ones(768))
        assert hash(a) == hash(b)

    def test_set_deduplication(self) -> None:
        a = Theme(theme_id="theme_abc", name="a", centroid=np.zeros(768))
        b = Theme(theme_id="theme_abc", name="b", centroid=np.ones(768))
        c = Theme(theme_id="theme_xyz", name="c", centroid=np.zeros(768))
        assert len({a, b, c}) == 2

    def test_non_theme_comparison(self) -> None:
        theme = Theme(theme_id="theme_abc", name="a", centroid=np.zeros(768))
        assert theme.__eq__("not a theme") is NotImplemented


class TestThemeSerialization:
    """Test to_dict / from_dict roundtrip."""

    def test_to_dict_centroid_is_list(self, sample_theme: Theme) -> None:
        d = sample_theme.to_dict()
        assert isinstance(d["centroid"], list)
        assert len(d["centroid"]) == 768

    def test_to_dict_datetime_is_iso(self, sample_theme: Theme) -> None:
        d = sample_theme.to_dict()
        # Should be parseable ISO strings
        datetime.fromisoformat(d["created_at"])
        datetime.fromisoformat(d["updated_at"])

    def test_to_dict_none_description(self) -> None:
        theme = Theme(
            theme_id="theme_test",
            name="test",
            centroid=np.zeros(768),
            description=None,
        )
        d = theme.to_dict()
        assert d["description"] is None

    def test_roundtrip(self, sample_theme: Theme) -> None:
        d = sample_theme.to_dict()
        restored = Theme.from_dict(d)
        assert restored.theme_id == sample_theme.theme_id
        assert restored.name == sample_theme.name
        assert restored.lifecycle_stage == sample_theme.lifecycle_stage
        assert restored.document_count == sample_theme.document_count
        assert restored.top_keywords == sample_theme.top_keywords
        assert restored.top_tickers == sample_theme.top_tickers
        assert restored.description == sample_theme.description
        assert restored.top_entities == sample_theme.top_entities
        assert restored.metadata == sample_theme.metadata
        np.testing.assert_allclose(
            restored.centroid, sample_theme.centroid, atol=1e-6,
        )

    def test_from_dict_parses_json_strings(self) -> None:
        """from_dict handles JSONB strings for top_entities/metadata."""
        d = {
            "theme_id": "theme_test",
            "name": "test",
            "centroid": [0.0] * 768,
            "top_entities": '[{"type": "COMPANY"}]',
            "metadata": '{"key": "value"}',
        }
        theme = Theme.from_dict(d)
        assert theme.top_entities == [{"type": "COMPANY"}]
        assert theme.metadata == {"key": "value"}

    def test_from_dict_defaults(self) -> None:
        """from_dict fills in missing optional fields."""
        d = {
            "theme_id": "theme_test",
            "name": "test",
            "centroid": [0.0] * 768,
        }
        theme = Theme.from_dict(d)
        assert theme.top_keywords == []
        assert theme.top_tickers == []
        assert theme.lifecycle_stage == "emerging"
        assert theme.document_count == 0
        assert theme.description is None
        assert theme.top_entities == []
        assert theme.metadata == {}


class TestThemeMetrics:
    """Test ThemeMetrics dataclass."""

    def test_construction(self) -> None:
        metrics = ThemeMetrics(
            theme_id="theme_abc",
            date=date(2025, 6, 15),
            document_count=42,
            sentiment_score=0.35,
            volume_zscore=1.8,
            velocity=0.12,
            acceleration=0.03,
            avg_authority=0.65,
            bullish_ratio=0.72,
        )
        assert metrics.theme_id == "theme_abc"
        assert metrics.date == date(2025, 6, 15)
        assert metrics.document_count == 42
        assert metrics.sentiment_score == pytest.approx(0.35)

    def test_defaults(self) -> None:
        metrics = ThemeMetrics(
            theme_id="theme_abc",
            date=date(2025, 6, 15),
        )
        assert metrics.document_count == 0
        assert metrics.sentiment_score is None
        assert metrics.volume_zscore is None
        assert metrics.velocity is None
        assert metrics.acceleration is None
        assert metrics.avg_authority is None
        assert metrics.bullish_ratio is None
