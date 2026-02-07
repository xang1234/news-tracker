"""Schema definitions for alert records.

Maps 1:1 to the ``alerts`` database table. Each alert represents an
actionable condition detected during daily clustering analysis —
sentiment velocity shifts, extreme sentiment, volume surges,
lifecycle transitions, or new theme emergence.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

AlertTriggerType = Literal[
    "sentiment_velocity",
    "extreme_sentiment",
    "volume_surge",
    "lifecycle_change",
    "new_theme",
    "propagated_impact",
]

VALID_TRIGGER_TYPES: frozenset[str] = frozenset({
    "sentiment_velocity",
    "extreme_sentiment",
    "volume_surge",
    "lifecycle_change",
    "new_theme",
    "propagated_impact",
})

AlertSeverity = Literal["critical", "warning", "info"]

VALID_SEVERITIES: frozenset[str] = frozenset({
    "critical",
    "warning",
    "info",
})


@dataclass
class Alert:
    """A persisted alert record from the alerts table.

    Attributes:
        alert_id: UUID4 identifier.
        theme_id: Theme that triggered the alert.
        trigger_type: What condition was detected.
        severity: Urgency level (critical, warning, info).
        title: Short human-readable summary.
        message: Detailed description of the condition.
        trigger_data: JSONB payload with trigger-specific context.
        acknowledged: Whether a user has reviewed the alert.
        created_at: When the alert was generated.
    """

    theme_id: str
    trigger_type: str
    severity: str
    title: str
    message: str
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_data: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if self.trigger_type not in VALID_TRIGGER_TYPES:
            raise ValueError(
                f"Invalid trigger_type {self.trigger_type!r}. "
                f"Must be one of: {sorted(VALID_TRIGGER_TYPES)}"
            )
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity {self.severity!r}. "
                f"Must be one of: {sorted(VALID_SEVERITIES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "alert_id": self.alert_id,
            "theme_id": self.theme_id,
            "trigger_type": self.trigger_type,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "trigger_data": self.trigger_data,
            "acknowledged": self.acknowledged,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Alert":
        """Create an Alert from a dictionary.

        Args:
            data: Dictionary with alert fields.

        Returns:
            Alert instance.
        """
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        trigger_data = data.get("trigger_data", {})
        if isinstance(trigger_data, str):
            trigger_data = json.loads(trigger_data)

        return cls(
            alert_id=data.get("alert_id", str(uuid.uuid4())),
            theme_id=data["theme_id"],
            trigger_type=data["trigger_type"],
            severity=data["severity"],
            title=data["title"],
            message=data["message"],
            trigger_data=trigger_data,
            acknowledged=data.get("acknowledged", False),
            created_at=created_at,
        )
