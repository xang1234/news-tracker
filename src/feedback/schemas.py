"""Schema definitions for feedback records.

Maps 1:1 to the ``feedback`` database table. Each feedback record
represents a user's quality rating on a theme, alert, or document.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

VALID_ENTITY_TYPES: frozenset[str] = frozenset({
    "theme",
    "alert",
    "document",
})

VALID_QUALITY_LABELS: frozenset[str] = frozenset({
    "useful",
    "noise",
    "too_late",
    "wrong_direction",
})


@dataclass
class Feedback:
    """A persisted feedback record from the feedback table.

    Attributes:
        feedback_id: Deterministic identifier (feedback_{uuid_hex[:12]}).
        entity_type: What kind of entity is being rated (theme, alert, document).
        entity_id: Identifier of the rated entity.
        rating: Quality score from 1 (poor) to 5 (excellent).
        quality_label: Optional categorical label for the feedback.
        comment: Optional free-text comment.
        user_id: Identifier of the user who submitted (typically the API key).
        created_at: When the feedback was submitted.
    """

    entity_type: str
    entity_id: str
    rating: int
    feedback_id: str = field(
        default_factory=lambda: f"feedback_{uuid.uuid4().hex[:12]}"
    )
    quality_label: str | None = None
    comment: str | None = None
    user_id: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if self.entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type {self.entity_type!r}. "
                f"Must be one of: {sorted(VALID_ENTITY_TYPES)}"
            )
        if not (1 <= self.rating <= 5):
            raise ValueError(
                f"Invalid rating {self.rating}. Must be between 1 and 5."
            )
        if self.quality_label is not None and self.quality_label not in VALID_QUALITY_LABELS:
            raise ValueError(
                f"Invalid quality_label {self.quality_label!r}. "
                f"Must be one of: {sorted(VALID_QUALITY_LABELS)}"
            )
