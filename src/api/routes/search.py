"""
Semantic search endpoint for finding similar documents.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status
import structlog

from src.api.auth import verify_api_key
from src.api.dependencies import get_vector_store_manager
from src.api.models import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    ErrorResponse,
)
from src.vectorstore.base import VectorSearchFilter
from src.vectorstore.manager import VectorStoreManager

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post(
    "/search/similar",
    response_model=SearchResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Search error"},
    },
    summary="Search for similar documents",
    description="""
    Find documents semantically similar to a query text.

    Uses FinBERT embeddings (768-dim) and pgvector's HNSW index for
    efficient approximate nearest neighbor search.

    **Filters** (all optional, combined with AND):
    - `platforms`: Only return documents from these platforms
    - `tickers`: Only return documents mentioning these ticker symbols
    - `theme_ids`: Only return documents with these theme cluster IDs
    - `min_authority_score`: Only return documents above this authority threshold

    **Authority Score** (0.0-1.0) is computed from:
    - Author verification status
    - Follower count (log-scaled)
    - Engagement metrics (log-scaled)
    - Inverse spam score
    """,
)
async def search_similar(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key),
    manager: VectorStoreManager = Depends(get_vector_store_manager),
) -> SearchResponse:
    """
    Search for documents similar to query text.

    Args:
        request: Search request with query and filters
        api_key: Validated API key
        manager: Vector store manager

    Returns:
        Search results with similarity scores
    """
    start_time = time.perf_counter()

    try:
        # Build filter from request
        filters = None
        if any([
            request.platforms,
            request.tickers,
            request.theme_ids,
            request.min_authority_score is not None,
        ]):
            filters = VectorSearchFilter(
                platforms=request.platforms,
                tickers=request.tickers,
                theme_ids=request.theme_ids,
                min_authority_score=request.min_authority_score,
            )

        # Execute search
        results = await manager.query(
            text=request.query,
            limit=request.limit,
            threshold=request.threshold,
            filters=filters,
        )

        # Convert to response format
        items = [
            SearchResultItem(
                document_id=r.document_id,
                score=round(r.score, 4),
                platform=r.metadata.get("platform"),
                title=r.metadata.get("title"),
                content_preview=r.metadata.get("content_preview"),
                url=r.metadata.get("url"),
                author_name=r.metadata.get("author_name"),
                author_verified=r.metadata.get("author_verified", False),
                tickers=r.metadata.get("tickers", []),
                authority_score=r.metadata.get("authority_score"),
                timestamp=r.metadata.get("timestamp"),
            )
            for r in results
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Search completed",
            query_length=len(request.query),
            results_count=len(items),
            latency_ms=round(latency_ms, 2),
        )

        return SearchResponse(
            results=items,
            total=len(items),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )
