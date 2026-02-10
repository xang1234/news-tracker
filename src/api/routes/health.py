"""
Health check endpoint with comprehensive infrastructure checks.
"""

import time

import structlog
import torch
from fastapi import APIRouter, Depends

from src.api.dependencies import get_database, get_embedding_service, get_redis_client
from src.api.models import ComponentHealth, HealthResponse
from src.config.settings import get_settings
from src.embedding.service import EmbeddingService, ModelType
from src.storage.database import Database

router = APIRouter()
logger = structlog.get_logger(__name__)

# Stream names used by the pipeline
_QUEUE_STREAMS = ["embedding_queue", "sentiment_queue", "clustering_queue"]


async def _check_database(db: Database) -> ComponentHealth:
    """Check database connectivity and measure latency."""
    start = time.perf_counter()
    try:
        healthy = await db.health_check()
        latency_ms = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            status="healthy" if healthy else "unhealthy",
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            details={"error": str(e)},
        )


async def _check_redis(redis_client) -> ComponentHealth:
    """Check Redis connectivity and measure latency."""
    start = time.perf_counter()
    try:
        await redis_client.ping()
        latency_ms = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            status="healthy",
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            status="unhealthy",
            latency_ms=round(latency_ms, 2),
            details={"error": str(e)},
        )


async def _get_queue_depths(redis_client) -> dict[str, int]:
    """Get current depth of each processing queue."""
    depths: dict[str, int] = {}
    for stream in _QUEUE_STREAMS:
        try:
            length = await redis_client.xlen(stream)
            depths[stream] = length
        except Exception:
            depths[stream] = -1
    return depths


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Check the health of the service and its dependencies.",
)
async def health_check(
    service: EmbeddingService = Depends(get_embedding_service),
    db: Database = Depends(get_database),
) -> HealthResponse:
    """
    Check service health including database, Redis, and queue depths.

    Status logic:
    - unhealthy: database is down
    - degraded: Redis is down (service works but without caching/queues)
    - healthy: all components operational
    """
    settings = get_settings()

    # Check which models are loaded
    models_loaded = {
        "finbert": service.is_model_initialized(ModelType.FINBERT),
        "minilm": service.is_model_initialized(ModelType.MINILM),
    }

    # Check cache availability using public method
    cache_available = await service.is_cache_available()

    # Get service stats
    service_stats = service.get_stats()

    # Infrastructure checks
    components: dict[str, ComponentHealth] = {}

    # Database check
    db_health = await _check_database(db)
    components["database"] = db_health

    # Redis check + queue depths
    redis_health = ComponentHealth(status="unhealthy")
    queue_depths: dict[str, int] = {}

    try:
        # get_redis_client is an async generator, so we need to handle it manually
        import redis.asyncio as aioredis
        from src.api.dependencies import _redis_client

        redis_client = _redis_client
        if redis_client is None:
            redis_client = aioredis.from_url(
                str(settings.redis_url),
                encoding="utf-8",
                decode_responses=True,
            )

        redis_health = await _check_redis(redis_client)
        components["redis"] = redis_health

        if redis_health.status == "healthy":
            queue_depths = await _get_queue_depths(redis_client)
    except Exception as e:
        components["redis"] = ComponentHealth(
            status="unhealthy",
            details={"error": str(e)},
        )

    # Determine overall status
    if db_health.status == "unhealthy":
        status = "unhealthy"
    elif redis_health.status == "unhealthy":
        status = "degraded"
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        cache_available=cache_available,
        gpu_available=torch.cuda.is_available(),
        service_stats=service_stats,
        components=components,
        queue_depths=queue_depths,
        version="0.1.0",
    )
