"""
Embedding endpoint for batch text embedding.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.requests import Request

from src.api.auth import verify_api_key
from src.api.dependencies import get_embedding_service
from src.api.models import APIModelType, EmbedRequest, EmbedResponse, ErrorResponse
from src.api.rate_limit import limiter
from src.config.settings import get_settings
from src.embedding.service import EmbeddingService, ModelType

router = APIRouter()


def _api_to_model_type(api_model: APIModelType, avg_length: float) -> ModelType:
    """Convert API model type to internal ModelType with auto-selection."""
    if api_model == APIModelType.FINBERT:
        return ModelType.FINBERT
    elif api_model == APIModelType.MINILM:
        return ModelType.MINILM
    else:  # AUTO
        # Use MiniLM for short texts, FinBERT for longer ones
        if avg_length < 300:
            return ModelType.MINILM
        return ModelType.FINBERT


@router.post(
    "/embed",
    response_model=EmbedResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Embedding error"},
    },
    summary="Generate embeddings for texts",
    description="""
    Generate embeddings for a batch of texts.

    Model selection:
    - **auto**: Automatically select based on text length (<300 chars → MiniLM, else → FinBERT)
    - **finbert**: Use FinBERT (768-dim, financial domain, slower but higher quality)
    - **minilm**: Use MiniLM (384-dim, general purpose, faster)

    Caching is enabled by default to speed up repeated requests.
    """,
)
@limiter.limit(lambda: get_settings().rate_limit_embed)
async def embed_texts(
    request: Request,
    body: EmbedRequest,
    api_key: str = Depends(verify_api_key),
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbedResponse:
    """
    Generate embeddings for a batch of texts.

    Args:
        request: Starlette request (for rate limiting)
        body: Embedding request with texts and options
        api_key: Validated API key
        service: Embedding service

    Returns:
        Embeddings with metadata
    """
    start_time = time.perf_counter()

    # Calculate average text length for auto model selection
    avg_length = sum(len(t) for t in body.texts) / len(body.texts)
    model_type = _api_to_model_type(body.model, avg_length)

    try:
        # Generate embeddings
        embeddings = await service.embed_batch(
            body.texts,
            model_type=model_type,
            show_progress=False,
        )

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Determine dimensions
        dimensions = 768 if model_type == ModelType.FINBERT else 384

        return EmbedResponse(
            embeddings=embeddings,
            model_used=model_type.value,
            dimensions=dimensions,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        import structlog
        structlog.get_logger(__name__).error("embed_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed",
        )
