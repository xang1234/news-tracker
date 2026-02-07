"""Tests for alert schema validation and serialization."""

import pytest
from datetime import datetime, timezone

from src.alerts.schemas import (
    Alert,
    VALID_TRIGGER_TYPES,
    VALID_SEVERITIES,
)


class TestAlertValidation:
    """Test Alert __post_init__ validation."""

    def test_valid_alert(self):
        alert = Alert(
            theme_id="theme_abc123",
            trigger_type="volume_surge",
            severity="warning",
            title="Test",
            message="Test message",
        )
        assert alert.trigger_type == "volume_surge"
        assert alert.severity == "warning"
        assert alert.acknowledged is False
        assert alert.alert_id  # UUID generated

    def test_invalid_trigger_type_raises(self):
        with pytest.raises(ValueError, match="Invalid trigger_type"):
            Alert(
                theme_id="t1",
                trigger_type="invalid_type",
                severity="warning",
                title="Bad",
                message="msg",
            )

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="Invalid severity"):
            Alert(
                theme_id="t1",
                trigger_type="volume_surge",
                severity="urgent",
                title="Bad",
                message="msg",
            )

    def test_all_trigger_types_valid(self):
        for tt in VALID_TRIGGER_TYPES:
            alert = Alert(
                theme_id="t1",
                trigger_type=tt,
                severity="info",
                title="Test",
                message="msg",
            )
            assert alert.trigger_type == tt

    def test_all_severities_valid(self):
        for sev in VALID_SEVERITIES:
            alert = Alert(
                theme_id="t1",
                trigger_type="new_theme",
                severity=sev,
                title="Test",
                message="msg",
            )
            assert alert.severity == sev

    def test_default_values(self):
        alert = Alert(
            theme_id="t1",
            trigger_type="new_theme",
            severity="info",
            title="T",
            message="M",
        )
        assert alert.trigger_data == {}
        assert alert.acknowledged is False
        assert isinstance(alert.created_at, datetime)
        assert alert.created_at.tzinfo is not None


class TestAlertSerialization:
    """Test to_dict/from_dict round-trip."""

    def test_to_dict(self):
        alert = Alert(
            theme_id="theme_abc",
            trigger_type="volume_surge",
            severity="critical",
            title="Volume surge",
            message="Z-score 4.5",
            trigger_data={"volume_zscore": 4.5},
        )
        d = alert.to_dict()

        assert d["theme_id"] == "theme_abc"
        assert d["trigger_type"] == "volume_surge"
        assert d["severity"] == "critical"
        assert d["trigger_data"] == {"volume_zscore": 4.5}
        assert d["acknowledged"] is False
        assert isinstance(d["created_at"], str)

    def test_from_dict(self):
        data = {
            "alert_id": "test-id-123",
            "theme_id": "theme_xyz",
            "trigger_type": "new_theme",
            "severity": "info",
            "title": "New theme",
            "message": "New theme detected",
            "trigger_data": {"theme_id": "theme_xyz"},
            "acknowledged": True,
            "created_at": "2026-02-07T10:00:00+00:00",
        }
        alert = Alert.from_dict(data)

        assert alert.alert_id == "test-id-123"
        assert alert.theme_id == "theme_xyz"
        assert alert.trigger_type == "new_theme"
        assert alert.acknowledged is True
        assert alert.created_at.year == 2026

    def test_round_trip(self):
        original = Alert(
            theme_id="theme_rt",
            trigger_type="lifecycle_change",
            severity="warning",
            title="Lifecycle",
            message="Theme transitioned",
            trigger_data={"from_stage": "emerging", "to_stage": "accelerating"},
        )
        restored = Alert.from_dict(original.to_dict())

        assert restored.alert_id == original.alert_id
        assert restored.theme_id == original.theme_id
        assert restored.trigger_type == original.trigger_type
        assert restored.severity == original.severity
        assert restored.trigger_data == original.trigger_data

    def test_from_dict_missing_optional_fields(self):
        data = {
            "theme_id": "t1",
            "trigger_type": "new_theme",
            "severity": "info",
            "title": "Title",
            "message": "Msg",
        }
        alert = Alert.from_dict(data)
        assert alert.trigger_data == {}
        assert alert.acknowledged is False
        assert isinstance(alert.alert_id, str)

    def test_from_dict_trigger_data_as_string(self):
        data = {
            "theme_id": "t1",
            "trigger_type": "new_theme",
            "severity": "info",
            "title": "T",
            "message": "M",
            "trigger_data": '{"key": "value"}',
        }
        alert = Alert.from_dict(data)
        assert alert.trigger_data == {"key": "value"}
