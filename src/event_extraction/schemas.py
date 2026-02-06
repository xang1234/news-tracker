"""Schema definitions for extracted events.

Provides EventType literals and EventRecord dataclass for representing
structured events extracted from financial text.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

EventType = Literal[
    "capacity_expansion",
    "capacity_constraint",
    "product_launch",
    "product_delay",
    "price_change",
    "guidance_change",
]

VALID_EVENT_TYPES: set[str] = {
    "capacity_expansion",
    "capacity_constraint",
    "product_launch",
    "product_delay",
    "price_change",
    "guidance_change",
}


@dataclass
class EventRecord:
    """
    A structured event extracted from financial text.

    Follows SVO (Subject-Verb-Object) structure:
    - actor: The entity performing the action (e.g., "TSMC", "Intel")
    - action: The verb/action phrase (e.g., "is expanding", "announced")
    - object: The target of the action (e.g., "fab capacity", "H200 GPU")

    Attributes:
        event_id: Unique event identifier (UUID4).
        doc_id: Foreign key to source document.
        event_type: Category of the event.
        actor: Entity performing the action (optional).
        action: The action phrase.
        object: Target of the action (optional).
        time_ref: Temporal reference from text (e.g., "Q3 2026").
        quantity: Numeric quantity mentioned (e.g., "$20 billion").
        tickers: Ticker symbols linked from surrounding context.
        confidence: Extraction confidence score (0.0-1.0).
        span_start: Character offset where the match starts.
        span_end: Character offset where the match ends.
        extractor_version: Version of the extractor that produced this.
        created_at: Timestamp when the event was extracted.
    """

    doc_id: str
    event_type: str
    action: str
    span_start: int
    span_end: int
    extractor_version: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    actor: str | None = None
    object: str | None = None
    time_ref: str | None = None
    quantity: str | None = None
    tickers: list[str] = field(default_factory=list)
    confidence: float = 0.7
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "doc_id": self.doc_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "action": self.action,
            "object": self.object,
            "time_ref": self.time_ref,
            "quantity": self.quantity,
            "tickers": self.tickers,
            "confidence": self.confidence,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "extractor_version": self.extractor_version,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventRecord":
        """
        Create EventRecord from dictionary.

        Args:
            data: Dictionary with event fields.

        Returns:
            EventRecord instance.
        """
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            event_id=data.get("event_id", str(uuid4())),
            doc_id=data["doc_id"],
            event_type=data["event_type"],
            actor=data.get("actor"),
            action=data["action"],
            object=data.get("object"),
            time_ref=data.get("time_ref"),
            quantity=data.get("quantity"),
            tickers=data.get("tickers", []),
            confidence=data.get("confidence", 0.7),
            span_start=data["span_start"],
            span_end=data["span_end"],
            extractor_version=data["extractor_version"],
            created_at=created_at,
        )
