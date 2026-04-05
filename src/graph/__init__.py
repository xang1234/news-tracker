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

from src.graph.baskets import (
    BasketMember,
    ThematicBasket,
    build_thematic_basket,
)
from src.graph.causal_graph import CausalGraph
from src.graph.config import GraphConfig
from src.graph.propagation import PropagationImpact, SentimentPropagation
from src.graph.schemas import (
    VALID_NODE_TYPES,
    VALID_RELATION_TYPES,
    CausalEdge,
    CausalNode,
)
from src.graph.publisher import (
    BasketPayload,
    PathExplanation,
    StructuralPublicationResult,
    build_basket_payload,
    build_path_explanation,
    prepare_structural_publication,
)
from src.graph.path_scoring import (
    ScoredEdge,
    ScoredPath,
    score_edge,
    score_paths_from,
)
from src.graph.seed_data import SEED_VERSION, seed_graph
from src.graph.storage import GraphRepository
from src.graph.structural import (
    StructuralRelation,
    StructuralSnapshot,
    build_structural_snapshot,
    get_predicate_sign,
    translate_derived_edge,
)

__all__ = [
    "BasketMember",
    "BasketPayload",
    "CausalEdge",
    "CausalGraph",
    "CausalNode",
    "GraphConfig",
    "GraphRepository",
    "PathExplanation",
    "PropagationImpact",
    "SEED_VERSION",
    "ScoredEdge",
    "ScoredPath",
    "SentimentPropagation",
    "StructuralPublicationResult",
    "StructuralRelation",
    "StructuralSnapshot",
    "ThematicBasket",
    "VALID_NODE_TYPES",
    "VALID_RELATION_TYPES",
    "build_basket_payload",
    "build_path_explanation",
    "build_structural_snapshot",
    "build_thematic_basket",
    "get_predicate_sign",
    "score_edge",
    "score_paths_from",
    "prepare_structural_publication",
    "seed_graph",
    "translate_derived_edge",
]
