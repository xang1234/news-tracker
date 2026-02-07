"""Schema definitions for the causal graph.

Defines domain objects for graph nodes and edges that map 1:1 to the
causal_nodes and causal_edges database tables.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

NodeType = Literal["ticker", "theme", "technology"]

RelationType = Literal[
    "depends_on", "supplies_to", "competes_with", "drives", "blocks"
]

VALID_NODE_TYPES = frozenset({"ticker", "theme", "technology"})

VALID_RELATION_TYPES = frozenset(
    {"depends_on", "supplies_to", "competes_with", "drives", "blocks"}
)


@dataclass
class CausalNode:
    """A node in the causal graph.

    Attributes:
        node_id: Unique identifier (e.g., "NVDA", "theme_abc123", "HBM3E").
        node_type: One of ticker, theme, technology.
        name: Human-readable display name.
        metadata: Flexible storage (sector, market_cap, etc.).
        created_at: When the node was first persisted.
        updated_at: When the node was last modified (DB trigger).
    """

    node_id: str
    node_type: NodeType
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if self.node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type {self.node_type!r}. "
                f"Must be one of: {sorted(VALID_NODE_TYPES)}"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CausalNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)


@dataclass
class CausalEdge:
    """A directed edge in the causal graph.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        relation: Relationship type.
        confidence: Edge confidence score (0.0-1.0).
        source_doc_ids: Document IDs supporting this edge.
        created_at: When the edge was first persisted.
        metadata: Flexible storage.
    """

    source: str
    target: str
    relation: RelationType
    confidence: float = 1.0
    source_doc_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.relation not in VALID_RELATION_TYPES:
            raise ValueError(
                f"Invalid relation {self.relation!r}. "
                f"Must be one of: {sorted(VALID_RELATION_TYPES)}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
