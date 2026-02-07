"""Sentiment propagation through the causal graph.

Implements edge-type-aware BFS to propagate sentiment deltas from a source
node to downstream nodes. Each edge type has a configurable weight (with sign)
and edge confidence attenuates the signal. Level-by-level processing ensures
the shallowest path determines impact (first arrival wins).

Example:
    propagation = SentimentPropagation(causal_graph)
    impacts = await propagation.propagate("theme_hbm_demand", -0.2)
    # → {"NVDA": PropagationImpact(impact=-0.14, depth=1, ...), ...}
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

from .causal_graph import CausalGraph
from .config import GraphConfig

logger = logging.getLogger(__name__)

# Mapping from relation type to config field name suffix
_WEIGHT_FIELD_MAP = {
    "depends_on": "propagation_weight_depends_on",
    "supplies_to": "propagation_weight_supplies_to",
    "competes_with": "propagation_weight_competes_with",
    "drives": "propagation_weight_drives",
    "blocks": "propagation_weight_blocks",
}


@dataclass(frozen=True)
class PropagationImpact:
    """Impact of sentiment propagation on a single node.

    Attributes:
        node_id: The affected node.
        impact: Propagated sentiment delta (sign preserved through edge weights).
        depth: Number of hops from the source node.
        path_relation: Edge type of the first hop that reached this node.
        edge_confidence: Confidence of that first-arriving edge.
    """

    node_id: str
    impact: float
    depth: int
    path_relation: str
    edge_confidence: float


class SentimentPropagation:
    """Propagate sentiment changes through the causal graph.

    Uses level-by-level BFS with edge-type-aware weights:
    - depends_on (+0.8): downstream dependency inherits impact
    - supplies_to (+0.6): supplier impact flows to customer
    - competes_with (-0.3): competitor impact inverts
    - drives (+0.5): driver impact flows forward
    - blocks (-0.4): blocker impact inverts
    """

    def __init__(
        self,
        graph: CausalGraph,
        config: GraphConfig | None = None,
    ) -> None:
        self._graph = graph
        self._config = config or GraphConfig()

    def _get_edge_weight(self, relation: str) -> float:
        """Look up the propagation weight for an edge type."""
        field_name = _WEIGHT_FIELD_MAP.get(relation)
        if field_name is None:
            return 0.0
        return getattr(self._config, field_name)

    async def propagate(
        self,
        source_node: str,
        sentiment_delta: float,
        max_depth: int | None = None,
        min_impact: float | None = None,
    ) -> dict[str, PropagationImpact]:
        """Propagate a sentiment delta through downstream edges.

        Algorithm:
        1. Fetch all downstream edges via get_downstream_edges()
        2. Group edges by depth level
        3. Process level by level:
           - target_impact = source_impact * edge_weight * confidence * decay
           - First arrival wins (shallowest path determines impact)
           - Skip if abs(impact) < min_impact
        4. Return {node_id: PropagationImpact}

        Args:
            source_node: Node where sentiment changed.
            sentiment_delta: Magnitude of sentiment change (e.g., -0.2).
            max_depth: Override config propagation_max_depth.
            min_impact: Override config propagation_min_impact.

        Returns:
            Dict mapping node_id to PropagationImpact for all affected nodes.
        """
        depth = max_depth if max_depth is not None else self._config.propagation_max_depth
        threshold = min_impact if min_impact is not None else self._config.propagation_min_impact

        edges = await self._graph.get_downstream_edges(source_node, max_depth=depth)

        if not edges:
            return {}

        # Group edges by depth for level-by-level processing
        edges_by_depth: dict[int, list[tuple[str, str, str, float]]] = defaultdict(list)
        for source, target, relation, confidence, edge_depth in edges:
            edges_by_depth[edge_depth].append((source, target, relation, confidence))

        # Track computed impacts: node_id → PropagationImpact
        impacts: dict[str, PropagationImpact] = {}

        # Source node has the original delta
        source_impacts: dict[str, float] = {source_node: sentiment_delta}

        decay = self._config.propagation_default_decay

        for level in sorted(edges_by_depth.keys()):
            for source, target, relation, confidence in edges_by_depth[level]:
                # Source impact comes from either the original delta or a previously
                # computed propagation impact
                if source in source_impacts:
                    src_impact = source_impacts[source]
                elif source in impacts:
                    src_impact = impacts[source].impact
                else:
                    continue

                edge_weight = self._get_edge_weight(relation)
                target_impact = src_impact * edge_weight * confidence * decay

                # First arrival wins — skip if already reached via shallower path
                if target in impacts:
                    continue

                if abs(target_impact) < threshold:
                    continue

                impacts[target] = PropagationImpact(
                    node_id=target,
                    impact=round(target_impact, 6),
                    depth=level,
                    path_relation=relation,
                    edge_confidence=confidence,
                )

        logger.info(
            "Sentiment propagation complete",
            source_node=source_node,
            sentiment_delta=sentiment_delta,
            affected_nodes=len(impacts),
            max_depth=depth,
        )

        return impacts
