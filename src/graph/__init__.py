"""Causal graph for semiconductor supply chain relationship modeling.

Provides directed graph infrastructure for modeling inter-entity
relationships (supplier→customer, competitor, technology dependencies)
with recursive CTE traversal for upstream/downstream impact analysis.

Components:
- CausalGraph: High-level service with auto-node-creation and config defaults
- GraphRepository: Low-level asyncpg CRUD and recursive CTE traversals
- CausalNode: Node dataclass (ticker, theme, technology)
- CausalEdge: Edge dataclass with confidence and provenance
- GraphConfig: Pydantic settings with GRAPH_ prefix
"""

from src.graph.causal_graph import CausalGraph
from src.graph.config import GraphConfig
from src.graph.propagation import PropagationImpact, SentimentPropagation
from src.graph.schemas import (
    VALID_NODE_TYPES,
    VALID_RELATION_TYPES,
    CausalEdge,
    CausalNode,
)
from src.graph.seed_data import SEED_VERSION, seed_graph
from src.graph.storage import GraphRepository

__all__ = [
    "CausalEdge",
    "CausalGraph",
    "CausalNode",
    "GraphConfig",
    "GraphRepository",
    "PropagationImpact",
    "SEED_VERSION",
    "SentimentPropagation",
    "VALID_NODE_TYPES",
    "VALID_RELATION_TYPES",
    "seed_graph",
]
