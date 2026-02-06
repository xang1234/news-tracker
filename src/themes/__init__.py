"""
Theme persistence layer for clustering results.

Maps BERTopic clustering output to the themes database table,
providing CRUD operations for theme records with pgvector centroid
storage and lifecycle stage tracking.

Components:
- Theme: Dataclass mapping to the themes table
- ThemeRepository: CRUD operations for theme persistence
- VALID_LIFECYCLE_STAGES: Allowed lifecycle stage values
"""

from src.themes.repository import ThemeRepository
from src.themes.schemas import VALID_LIFECYCLE_STAGES, Theme

__all__ = [
    "Theme",
    "ThemeRepository",
    "VALID_LIFECYCLE_STAGES",
]
