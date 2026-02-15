"""
Health check endpoint with comprehensive infrastructure checks.
"""

import time

import structlog
import torch
from fastapi import APIRouter, Depends

from src.api.dependencies import get_database, get_embedding_service, get_redis_client
from src.api.models import ComponentHealth, HealthResponse, QueueMetrics
from src.config.settings import get_settings
from src.embedding.service import EmbeddingService, ModelType
from src.storage.database import Database

router = APIRouter()
logger = structlog.get_logger(__name__)


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


async def _get_queue_depths(redis_client) -> dict[str, QueueMetrics]:
    """Get pending backlog and processed count for each processing queue."""
    settings = get_settings()
    stream_groups = {
        settings.embedding_stream_name: settings.embedding_consumer_group,
        settings.sentiment_stream_name: settings.sentiment_consumer_group,
        settings.clustering_stream_name: settings.clustering_consumer_group,
    }

    depths: dict[str, QueueMetrics] = {}
    for stream, group_name in stream_groups.items():
        try:
            stream_len = await redis_client.xlen(stream)
            try:
                groups = await redis_client.xinfo_groups(stream)
            except Exception:
                # Stream exists (xlen succeeded) but has no groups,
                # or stream doesn't exist yet — entire stream is backlog
                depths[stream] = QueueMetrics(pending=stream_len, processed=0)
                continue

            group_info = None
            for g in groups:
                if g.get("name") == group_name:
                    group_info = g
                    break

            if group_info is None:
                # Group hasn't been created yet — entire stream is backlog
                depths[stream] = QueueMetrics(pending=stream_len, processed=0)
                continue

            in_flight = group_info.get("pel-count", 0) or 0
            lag = group_info.get("lag")
            entries_read = group_info.get("entries-read")

            if lag is not None:
                pending = lag + in_flight
            else:
                # Redis < 7.0 fallback: estimate from stream length
                pending = stream_len

            if entries_read is not None:
                processed = max(0, entries_read - in_flight)
            else:
                processed = 0

            depths[stream] = QueueMetrics(pending=pending, processed=processed)
        except Exception:
            depths[stream] = QueueMetrics(pending=-1, processed=-1)
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
    queue_depths: dict[str, QueueMetrics] = {}

    try:
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )

        redis_health = await _check_redis(redis_client)
        components["redis"] = redis_health

        if redis_health.status == "healthy":
            queue_depths = await _get_queue_depths(redis_client)

        await redis_client.aclose()
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
