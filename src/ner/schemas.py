"""Schema definitions for NER entities.

Provides lightweight dataclasses for representing extracted entities,
with serialization methods for storage and API responses.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

# Supported entity types for financial NER
EntityType = Literal["TICKER", "COMPANY", "PRODUCT", "TECHNOLOGY", "METRIC"]


@dataclass
class FinancialEntity:
    """
    A single extracted financial entity.

    Represents an entity found in text with its type, position,
    normalized form, and confidence score.

    Attributes:
        text: Original text span as found in the document.
        type: Entity type (TICKER, COMPANY, PRODUCT, TECHNOLOGY, METRIC).
        normalized: Normalized form of the entity (e.g., "NVDA" for "Nvidia").
        start: Character offset where entity starts in the text.
        end: Character offset where entity ends in the text.
        confidence: Confidence score from 0.0 to 1.0.
        metadata: Additional entity-specific metadata (e.g., ticker symbol for companies).

    Example:
        >>> entity = FinancialEntity(
        ...     text="Nvidia",
        ...     type="COMPANY",
        ...     normalized="NVIDIA",
        ...     start=0,
        ...     end=6,
        ...     confidence=0.95,
        ...     metadata={"ticker": "NVDA"}
        ... )
        >>> entity.to_dict()
        {'text': 'Nvidia', 'type': 'COMPANY', 'normalized': 'NVIDIA', ...}
    """

    text: str
    type: EntityType
    normalized: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert entity to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for storage or API response.
        """
        return {
            "text": self.text,
            "type": self.type,
            "normalized": self.normalized,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinancialEntity":
        """
        Create entity from dictionary.

        Args:
            data: Dictionary with entity fields.

        Returns:
            FinancialEntity instance.

        Raises:
            KeyError: If required fields are missing.
        """
        return cls(
            text=data["text"],
            type=data["type"],
            normalized=data["normalized"],
            start=data["start"],
            end=data["end"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on normalized form and type."""
        if not isinstance(other, FinancialEntity):
            return NotImplemented
        return self.normalized == other.normalized and self.type == other.type

    def __hash__(self) -> int:
        """Hash based on normalized form and type for deduplication."""
        return hash((self.normalized, self.type))

    def overlaps(self, other: "FinancialEntity") -> bool:
        """
        Check if this entity overlaps with another in the text.

        Args:
            other: Another entity to check for overlap.

        Returns:
            True if the entities' character spans overlap.
        """
        return not (self.end <= other.start or other.end <= self.start)
