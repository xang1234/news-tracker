"""
Theme persistence layer for clustering results.

Maps BERTopic clustering output to the themes database table,
providing CRUD operations for theme records with pgvector centroid
storage and lifecycle stage tracking.

Components:
- Theme: Dataclass mapping to the themes table
- ThemeRepository: CRUD operations for theme persistence
- VALID_LIFECYCLE_STAGES: Allowed lifecycle stage values
- LifecycleClassifier: Rule-based lifecycle stage classification
- LifecycleTransition: Stage change records for alerting
"""

from src.themes.lifecycle import LifecycleClassifier
from src.themes.metrics import VolumeMetricsConfig, VolumeMetricsService
from src.themes.ranking import RankedTheme, RankingConfig, ThemeRankingService
from src.themes.repository import ThemeRepository
from src.themes.schemas import VALID_LIFECYCLE_STAGES, Theme, ThemeMetrics
from src.themes.transitions import ALERTABLE_TRANSITIONS, LifecycleTransition

__all__ = [
    "ALERTABLE_TRANSITIONS",
    "LifecycleClassifier",
    "LifecycleTransition",
    "RankedTheme",
    "RankingConfig",
    "Theme",
    "ThemeMetrics",
    "ThemeRankingService",
    "ThemeRepository",
    "VALID_LIFECYCLE_STAGES",
    "VolumeMetricsConfig",
    "VolumeMetricsService",
]
