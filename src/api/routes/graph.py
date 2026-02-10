"""Graph endpoints for nodes, subgraphs, and sentiment propagation."""

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.requests import Request

from src.api.auth import verify_api_key
from src.api.dependencies import get_graph_repository, get_propagation_service
from src.api.rate_limit import limiter
from src.config.settings import get_settings as _get_settings
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
@limiter.limit(lambda: _get_settings().rate_limit_graph)
async def list_graph_nodes(
    request: Request,
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
        logger.error("list_graph_nodes_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list graph nodes",
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
@limiter.limit(lambda: _get_settings().rate_limit_graph)
async def get_node_subgraph(
    request: Request,
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
        logger.error("get_subgraph_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get subgraph",
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
@limiter.limit(lambda: _get_settings().rate_limit_graph)
async def propagate_sentiment(
    request: Request,
    body: PropagateRequest,
    api_key: str = Depends(verify_api_key),
    propagation: SentimentPropagation = Depends(get_propagation_service),
) -> PropagateResponse:
    start_time = time.perf_counter()

    try:
        impacts = await propagation.propagate(
            source_node=body.source_node,
            sentiment_delta=body.sentiment_delta,
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
            source_node=body.source_node,
            sentiment_delta=body.sentiment_delta,
            total_affected=len(items),
            latency_ms=round(latency_ms, 2),
        )

        return PropagateResponse(
            source_node=body.source_node,
            sentiment_delta=body.sentiment_delta,
            impacts=items,
            total_affected=len(items),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error("propagate_sentiment_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to propagate sentiment",
        )
