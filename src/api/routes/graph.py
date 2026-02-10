"""Graph endpoints for nodes, subgraphs, and sentiment propagation."""

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.auth import verify_api_key
from src.api.dependencies import get_graph_repository, get_propagation_service
from src.api.models import (
    ErrorResponse,
    GraphEdgeItem,
    GraphNodeItem,
    GraphNodesResponse,
    PropagateRequest,
    PropagateResponse,
    PropagationImpactItem,
    SubgraphResponse,
)
from src.graph.propagation import SentimentPropagation
from src.graph.storage import GraphRepository

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/graph/nodes",
    response_model=GraphNodesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="List graph nodes",
    description="List all causal graph nodes with optional type filter.",
)
async def list_graph_nodes(
    node_type: str | None = Query(default=None, description="Filter by node type (ticker, theme, technology)"),
    limit: int = Query(default=200, ge=1, le=1000, description="Maximum nodes to return"),
    api_key: str = Depends(verify_api_key),
    repo: GraphRepository = Depends(get_graph_repository),
) -> GraphNodesResponse:
    start_time = time.perf_counter()

    try:
        nodes = await repo.get_all_nodes(node_type=node_type, limit=limit)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return GraphNodesResponse(
            nodes=[
                GraphNodeItem(
                    node_id=n.node_id,
                    node_type=n.node_type,
                    name=n.name,
                    metadata=n.metadata,
                )
                for n in nodes
            ],
            total=len(nodes),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Failed to list graph nodes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list graph nodes: {str(e)}",
        )


@router.get(
    "/graph/nodes/{node_id}/subgraph",
    response_model=SubgraphResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Node not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get subgraph around a node",
    description="Extract a local subgraph (nodes + edges) centered on the given node.",
)
async def get_node_subgraph(
    node_id: str,
    depth: int = Query(default=2, ge=1, le=5, description="Traversal depth"),
    api_key: str = Depends(verify_api_key),
    repo: GraphRepository = Depends(get_graph_repository),
) -> SubgraphResponse:
    start_time = time.perf_counter()

    try:
        node = await repo.get_node(node_id)
        if node is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node '{node_id}' not found",
            )

        subgraph = await repo.get_subgraph(node_id, depth=depth)
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Subgraph extracted",
            center_node=node_id,
            depth=depth,
            node_count=len(subgraph["nodes"]),
            edge_count=len(subgraph["edges"]),
            latency_ms=round(latency_ms, 2),
        )

        return SubgraphResponse(
            nodes=[GraphNodeItem(**n) for n in subgraph["nodes"]],
            edges=[GraphEdgeItem(**e) for e in subgraph["edges"]],
            center_node=node_id,
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get subgraph: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get subgraph: {str(e)}",
        )


@router.post(
    "/graph/propagate",
    response_model=PropagateResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Propagate sentiment through causal graph",
    description=(
        "Given a source node and sentiment delta, propagate the impact "
        "through downstream edges in the causal graph. Returns all affected "
        "nodes with their computed impact, ordered by absolute impact magnitude."
    ),
)
async def propagate_sentiment(
    request: PropagateRequest,
    api_key: str = Depends(verify_api_key),
    propagation: SentimentPropagation = Depends(get_propagation_service),
) -> PropagateResponse:
    start_time = time.perf_counter()

    try:
        impacts = await propagation.propagate(
            source_node=request.source_node,
            sentiment_delta=request.sentiment_delta,
        )

        items = sorted(
            [
                PropagationImpactItem(
                    node_id=imp.node_id,
                    impact=imp.impact,
                    depth=imp.depth,
                    relation=imp.path_relation,
                    edge_confidence=imp.edge_confidence,
                )
                for imp in impacts.values()
            ],
            key=lambda x: abs(x.impact),
            reverse=True,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Sentiment propagation computed",
            source_node=request.source_node,
            sentiment_delta=request.sentiment_delta,
            total_affected=len(items),
            latency_ms=round(latency_ms, 2),
        )

        return PropagateResponse(
            source_node=request.source_node,
            sentiment_delta=request.sentiment_delta,
            impacts=items,
            total_affected=len(items),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Failed to propagate sentiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to propagate sentiment: {str(e)}",
        )
