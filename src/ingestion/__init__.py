"""Data ingestion module - adapters, schemas, and preprocessing."""

from src.ingestion.schemas import (
    EngagementMetrics,
    NormalizedDocument,
    Platform,
)

__all__ = [
    "Platform",
    "EngagementMetrics",
    "NormalizedDocument",
]
