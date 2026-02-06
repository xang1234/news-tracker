"""Schema definitions for document clustering themes.

Provides a dataclass representing a discovered theme cluster from BERTopic,
with deterministic ID generation and serialization methods for storage
and API responses.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class ThemeCluster:
    """
    A discovered theme cluster from BERTopic topic modeling.

    Represents a group of semantically related documents sharing a common
    topic, with representative keywords, a centroid embedding for similarity
    matching, and the list of document IDs assigned to this theme.

    Attributes:
        theme_id: Deterministic ID derived from topic words (theme_{hash}).
        name: Human-readable name from top topic words (e.g., "gpu_architecture_nvidia").
        topic_words: Ranked list of (word, score) tuples from c-TF-IDF.
        centroid: Mean embedding vector of all documents in this cluster.
        document_count: Number of documents assigned to this theme.
        document_ids: List of document IDs belonging to this cluster.
        created_at: Timestamp when the theme was discovered.
        metadata: Additional information (e.g., bertopic_topic_id).

    Example:
        >>> theme = ThemeCluster(
        ...     theme_id="theme_a1b2c3d4e5f6",
        ...     name="gpu_architecture_nvidia",
        ...     topic_words=[("gpu", 0.15), ("architecture", 0.12)],
        ...     centroid=np.zeros(768),
        ...     document_count=25,
        ...     document_ids=["doc_001", "doc_002"],
        ... )
        >>> theme.to_dict()["theme_id"]
        'theme_a1b2c3d4e5f6'
    """

    theme_id: str
    name: str
    topic_words: list[tuple[str, float]]
    centroid: np.ndarray
    document_count: int
    document_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_theme_id(topic_words: list[tuple[str, float]]) -> str:
        """
        Generate a deterministic theme ID from topic words.

        Uses SHA256 hash of sorted word strings (ignoring scores) to produce
        a stable ID that remains consistent across re-fits with the same
        topic word set.

        Args:
            topic_words: List of (word, score) tuples from BERTopic.

        Returns:
            Deterministic ID string in format "theme_{hash[:12]}".
        """
        sorted_words = sorted(word for word, _ in topic_words)
        words_str = ",".join(sorted_words)
        hash_digest = hashlib.sha256(words_str.encode()).hexdigest()
        return f"theme_{hash_digest[:12]}"

    @staticmethod
    def generate_name(topic_words: list[tuple[str, float]], top_n: int = 3) -> str:
        """
        Generate a human-readable name from top topic words.

        Args:
            topic_words: List of (word, score) tuples, ordered by importance.
            top_n: Number of top words to include in the name.

        Returns:
            Underscore-joined name of the top words (e.g., "gpu_architecture_nvidia").
        """
        words = [word for word, _ in topic_words[:top_n]]
        return "_".join(words)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert theme cluster to dictionary for JSON serialization.

        The centroid ndarray is converted to a plain list for JSON compatibility.

        Returns:
            Dictionary representation suitable for storage or API response.
        """
        return {
            "theme_id": self.theme_id,
            "name": self.name,
            "topic_words": self.topic_words,
            "centroid": self.centroid.tolist(),
            "document_count": self.document_count,
            "document_ids": self.document_ids,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThemeCluster":
        """
        Create a ThemeCluster from a dictionary.

        Args:
            data: Dictionary with theme cluster fields.

        Returns:
            ThemeCluster instance.

        Raises:
            KeyError: If required fields are missing.
        """
        # Convert topic_words from list-of-lists back to list-of-tuples
        topic_words = [
            (word, score) if isinstance((word, score), tuple) else (word, score)
            for word, score in data["topic_words"]
        ]

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            theme_id=data["theme_id"],
            name=data["name"],
            topic_words=topic_words,
            centroid=np.array(data["centroid"]),
            document_count=data["document_count"],
            document_ids=data.get("document_ids", []),
            created_at=created_at,
            metadata=data.get("metadata", {}),
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on theme_id."""
        if not isinstance(other, ThemeCluster):
            return NotImplemented
        return self.theme_id == other.theme_id

    def __hash__(self) -> int:
        """Hash based on theme_id for use in sets and dicts."""
        return hash(self.theme_id)
