"""
Health check endpoint.
"""

import torch
from fastapi import APIRouter, Depends

from src.api.dependencies import get_embedding_service, get_redis_client
from src.api.models import HealthResponse
from src.embedding.service import EmbeddingService, ModelType

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Check the health of the embedding service and its dependencies.",
)
async def health_check(
    service: EmbeddingService = Depends(get_embedding_service),
) -> HealthResponse:
    """
    Check service health.

    Returns:
        Health status with model and cache availability
    """
    # Check which models are loaded
    models_loaded = {
        "finbert": service.is_model_initialized(ModelType.FINBERT),
        "minilm": service.is_model_initialized(ModelType.MINILM),
    }

    # Check cache availability using public method
    cache_available = await service.is_cache_available()

    # Get service stats
    service_stats = service.get_stats()

    # Determine overall status
    if cache_available:
        status = "healthy"
    else:
        status = "degraded"  # Service works but without caching

    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        cache_available=cache_available,
        gpu_available=torch.cuda.is_available(),
        service_stats=service_stats,
    )
