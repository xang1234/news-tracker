"""Tests for feedback schema validation."""

import pytest

from src.feedback.schemas import (
    VALID_ENTITY_TYPES,
    VALID_QUALITY_LABELS,
    Feedback,
)


class TestFeedbackCreation:
    """Tests for valid Feedback construction."""

    def test_valid_theme_feedback(self, sample_feedback):
        assert sample_feedback.entity_type == "theme"
        assert sample_feedback.entity_id == "theme_xyz789"
        assert sample_feedback.rating == 4
        assert sample_feedback.quality_label == "useful"
        assert sample_feedback.comment == "Very actionable insight"

    def test_minimal_feedback(self, sample_feedback_minimal):
        assert sample_feedback_minimal.entity_type == "alert"
        assert sample_feedback_minimal.rating == 2
        assert sample_feedback_minimal.quality_label is None
        assert sample_feedback_minimal.comment is None
        assert sample_feedback_minimal.user_id is None

    def test_feedback_id_format(self):
        f = Feedback(entity_type="document", entity_id="doc_1", rating=3)
        assert f.feedback_id.startswith("feedback_")
        assert len(f.feedback_id) == len("feedback_") + 12

    def test_all_entity_types(self):
        for et in VALID_ENTITY_TYPES:
            f = Feedback(entity_type=et, entity_id="x", rating=3)
            assert f.entity_type == et

    def test_all_quality_labels(self):
        for ql in VALID_QUALITY_LABELS:
            f = Feedback(entity_type="theme", entity_id="x", rating=3, quality_label=ql)
            assert f.quality_label == ql

    def test_none_quality_label_valid(self):
        f = Feedback(entity_type="theme", entity_id="x", rating=5)
        assert f.quality_label is None

    def test_rating_boundaries(self):
        for r in (1, 2, 3, 4, 5):
            f = Feedback(entity_type="theme", entity_id="x", rating=r)
            assert f.rating == r


class TestFeedbackValidation:
    """Tests for invalid Feedback construction."""

    def test_invalid_entity_type(self):
        with pytest.raises(ValueError, match="Invalid entity_type"):
            Feedback(entity_type="user", entity_id="x", rating=3)

    def test_invalid_entity_type_empty(self):
        with pytest.raises(ValueError, match="Invalid entity_type"):
            Feedback(entity_type="", entity_id="x", rating=3)

    def test_rating_too_low(self):
        with pytest.raises(ValueError, match="Invalid rating"):
            Feedback(entity_type="theme", entity_id="x", rating=0)

    def test_rating_too_high(self):
        with pytest.raises(ValueError, match="Invalid rating"):
            Feedback(entity_type="theme", entity_id="x", rating=6)

    def test_rating_negative(self):
        with pytest.raises(ValueError, match="Invalid rating"):
            Feedback(entity_type="theme", entity_id="x", rating=-1)

    def test_invalid_quality_label(self):
        with pytest.raises(ValueError, match="Invalid quality_label"):
            Feedback(
                entity_type="theme",
                entity_id="x",
                rating=3,
                quality_label="bad",
            )

    def test_invalid_quality_label_empty(self):
        with pytest.raises(ValueError, match="Invalid quality_label"):
            Feedback(
                entity_type="theme",
                entity_id="x",
                rating=3,
                quality_label="",
            )
