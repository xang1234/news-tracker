"""Schema definitions for extracted keywords.

Provides a lightweight dataclass for representing extracted keywords,
with serialization methods for storage and API responses.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedKeyword:
    """
    A single extracted keyword from TextRank.

    Represents a keyword phrase found in text with its importance score,
    rank position, and associated metadata.

    Attributes:
        text: Original keyword phrase as found in the document.
        score: TextRank importance score (higher = more important).
        rank: 1-based ranking position (1 = most important).
        lemma: Lemmatized/normalized form of the keyword.
        count: Frequency of the keyword in the document.
        metadata: Additional extraction metadata (e.g., algorithm version).

    Example:
        >>> keyword = ExtractedKeyword(
        ...     text="semiconductor chip",
        ...     score=0.125,
        ...     rank=1,
        ...     lemma="semiconductor chip",
        ...     count=3,
        ...     metadata={"algorithm": "textrank"}
        ... )
        >>> keyword.to_dict()
        {'text': 'semiconductor chip', 'score': 0.125, 'rank': 1, ...}
    """

    text: str
    score: float
    rank: int
    lemma: str
    count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert keyword to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for storage or API response.
        """
        return {
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "lemma": self.lemma,
            "count": self.count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedKeyword":
        """
        Create keyword from dictionary.

        Args:
            data: Dictionary with keyword fields.

        Returns:
            ExtractedKeyword instance.

        Raises:
            KeyError: If required fields are missing.
        """
        return cls(
            text=data["text"],
            score=data["score"],
            rank=data["rank"],
            lemma=data["lemma"],
            count=data.get("count", 1),
            metadata=data.get("metadata", {}),
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on lemmatized form."""
        if not isinstance(other, ExtractedKeyword):
            return NotImplemented
        return self.lemma == other.lemma

    def __hash__(self) -> int:
        """Hash based on lemmatized form for deduplication."""
        return hash(self.lemma)
