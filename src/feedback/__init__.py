"""Feedback service for user quality ratings on themes, alerts, and documents.

Components:
- Feedback: Dataclass mapping to the feedback table
- FeedbackConfig: Pydantic settings for feedback constraints
- FeedbackRepository: CRUD operations for feedback persistence
- VALID_ENTITY_TYPES / VALID_QUALITY_LABELS: Frozensets for runtime validation
"""

from src.feedback.config import FeedbackConfig
from src.feedback.repository import FeedbackRepository
from src.feedback.schemas import (
    VALID_ENTITY_TYPES,
    VALID_QUALITY_LABELS,
    Feedback,
)

__all__ = [
    "Feedback",
    "FeedbackConfig",
    "FeedbackRepository",
    "VALID_ENTITY_TYPES",
    "VALID_QUALITY_LABELS",
]
