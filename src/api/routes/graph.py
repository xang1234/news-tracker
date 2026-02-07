"""Graph propagation endpoints for sentiment impact analysis."""

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import verify_api_key
from src.api.dependencies import get_propagation_service
from src.api.models import ErrorResponse, PropagateRequest, PropagateResponse, PropagationImpactItem
from src.graph.propagation import SentimentPropagation

logger = structlog.get_logger(__name__)
router = APIRouter()


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
