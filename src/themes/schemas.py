"""Schema definitions for persisted theme records.

Maps 1:1 to the `themes` database table. Distinct from ThemeCluster
(the in-memory clustering artifact) — Theme has top_keywords, top_tickers,
top_entities, lifecycle_stage, and description, but NOT topic_words or
document_ids which are clustering-time constructs.

Also defines ThemeMetrics for the theme_metrics daily time-series table.
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

VALID_LIFECYCLE_STAGES = frozenset(
    {"emerging", "accelerating", "mature", "fading"}
)


@dataclass
class Theme:
    """
    A persisted theme record from the themes table.

    Attributes:
        theme_id: Deterministic ID (theme_{sha256[:12]}).
        name: Human-readable name (e.g., "gpu_nvidia_architecture").
        centroid: FinBERT 768-dim mean embedding vector.
        top_keywords: Ranked topic keywords as flat strings.
        top_tickers: Most-mentioned ticker symbols.
        lifecycle_stage: One of emerging, accelerating, mature, fading.
        document_count: Number of documents assigned to this theme.
        created_at: When the theme was first persisted.
        updated_at: When the theme was last modified (DB trigger).
        description: Optional human-readable summary.
        top_entities: Entity objects with scores (JSONB).
        metadata: Flexible storage (bertopic_topic_id, merged_from, etc.).
    """

    theme_id: str
    name: str
    centroid: np.ndarray
    top_keywords: list[str] = field(default_factory=list)
    top_tickers: list[str] = field(default_factory=list)
    lifecycle_stage: str = "emerging"
    document_count: int = 0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    description: str | None = None
    top_entities: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.lifecycle_stage not in VALID_LIFECYCLE_STAGES:
            raise ValueError(
                f"Invalid lifecycle_stage {self.lifecycle_stage!r}. "
                f"Must be one of: {sorted(VALID_LIFECYCLE_STAGES)}"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Theme):
            return NotImplemented
        return self.theme_id == other.theme_id

    def __hash__(self) -> int:
        return hash(self.theme_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "theme_id": self.theme_id,
            "name": self.name,
            "centroid": self.centroid.tolist(),
            "top_keywords": self.top_keywords,
            "top_tickers": self.top_tickers,
            "lifecycle_stage": self.lifecycle_stage,
            "document_count": self.document_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "description": self.description,
            "top_entities": self.top_entities,
            "metadata": self.metadata,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Theme":
        """Create a Theme from a dictionary.

        Args:
            data: Dictionary with theme fields.

        Returns:
            Theme instance.
        """
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now(timezone.utc)

        # top_entities may be JSON string or already a list
        top_entities = data.get("top_entities", [])
        if isinstance(top_entities, str):
            top_entities = json.loads(top_entities)

        metadata = data.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        deleted_at = data.get("deleted_at")
        if isinstance(deleted_at, str):
            deleted_at = datetime.fromisoformat(deleted_at)

        return cls(
            theme_id=data["theme_id"],
            name=data["name"],
            centroid=np.array(data["centroid"], dtype=np.float32),
            top_keywords=data.get("top_keywords", []),
            top_tickers=data.get("top_tickers", []),
            lifecycle_stage=data.get("lifecycle_stage", "emerging"),
            document_count=data.get("document_count", 0),
            created_at=created_at,
            updated_at=updated_at,
            description=data.get("description"),
            top_entities=top_entities,
            metadata=metadata,
            deleted_at=deleted_at,
        )


@dataclass
class ThemeMetrics:
    """
    Daily time-series metrics for a theme.

    Maps 1:1 to the theme_metrics table. Primary key is (theme_id, date).

    Attributes:
        theme_id: Parent theme identifier.
        date: Calendar date for this metrics snapshot.
        document_count: Number of documents assigned on this date.
        weighted_volume: Platform-weighted volume with recency decay.
        sentiment_score: Aggregate sentiment (-1.0 to 1.0).
        volume_zscore: Standard deviations from mean volume.
        velocity: Rate of volume change.
        acceleration: Rate of velocity change.
        avg_authority: Mean authority_score of documents.
        bullish_ratio: Fraction of positive sentiment documents.
    """

    theme_id: str
    date: date
    document_count: int = 0
    weighted_volume: float | None = None
    sentiment_score: float | None = None
    volume_zscore: float | None = None
    velocity: float | None = None
    acceleration: float | None = None
    avg_authority: float | None = None
    bullish_ratio: float | None = None
