"""Lifecycle transition records for theme stage changes.

A LifecycleTransition captures the moment a theme moves between lifecycle
stages (e.g., emerging -> accelerating). These records support downstream
alerting and audit logging.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Transitions worth alerting on, with human-readable descriptions.
ALERTABLE_TRANSITIONS: dict[tuple[str, str], str] = {
    ("emerging", "accelerating"): "Theme gaining momentum",
    ("accelerating", "mature"): "Theme may be peaking",
    ("emerging", "fading"): "Theme losing momentum",
    ("accelerating", "fading"): "Theme losing momentum",
    ("mature", "fading"): "Theme losing momentum",
}


@dataclass
class LifecycleTransition:
    """A detected lifecycle stage transition for a theme.

    Attributes:
        theme_id: The theme that transitioned.
        from_stage: Previous lifecycle stage.
        to_stage: New lifecycle stage.
        detected_at: When the transition was detected.
        confidence: Classifier confidence in the new stage (0-1).
        metadata: Additional context (e.g., trigger metrics).
    """

    theme_id: str
    from_stage: str
    to_stage: str
    detected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_alertable(self) -> bool:
        """Whether this transition warrants an alert."""
        return (self.from_stage, self.to_stage) in ALERTABLE_TRANSITIONS

    @property
    def alert_message(self) -> str | None:
        """Human-readable alert message, or None if not alertable."""
        return ALERTABLE_TRANSITIONS.get((self.from_stage, self.to_stage))

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "theme_id": self.theme_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "detected_at": self.detected_at.isoformat(),
            "confidence": self.confidence,
            "is_alertable": self.is_alertable,
            "alert_message": self.alert_message,
            "metadata": self.metadata,
        }
