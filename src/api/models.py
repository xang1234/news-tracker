"""
Request and response models for the embedding API.
"""

from enum import Enum

from pydantic import BaseModel, Field


class APIModelType(str, Enum):
    """Model selection for API requests."""

    AUTO = "auto"
    FINBERT = "finbert"
    MINILM = "minilm"


class EmbedRequest(BaseModel):
    """Request model for batch embedding."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="List of texts to embed (1-64)",
    )
    model: APIModelType = Field(
        default=APIModelType.AUTO,
        description="Model to use: auto (select by content), finbert, or minilm",
    )
    cache: bool = Field(
        default=True,
        description="Whether to use caching",
    )


class EmbedResponse(BaseModel):
    """Response model for batch embedding."""

    embeddings: list[list[float]] = Field(
        ...,
        description="List of embedding vectors",
    )
    model_used: str = Field(
        ...,
        description="Model that was used for embedding",
    )
    dimensions: int = Field(
        ...,
        description="Embedding dimensions (768 for finbert, 384 for minilm)",
    )
    latency_ms: float = Field(
        ...,
        description="Processing latency in milliseconds",
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(
        ...,
        description="Overall service status: healthy, degraded, or unhealthy",
    )
    models_loaded: dict[str, bool] = Field(
        default_factory=dict,
        description="Which models are currently loaded",
    )
    cache_available: bool = Field(
        default=False,
        description="Whether Redis cache is available",
    )
    gpu_available: bool = Field(
        default=False,
        description="Whether GPU (CUDA) is available for inference",
    )
    service_stats: dict = Field(
        default_factory=dict,
        description="Embedding service statistics",
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(
        ...,
        description="Error message",
    )
    error_type: str = Field(
        default="error",
        description="Error type",
    )


# Search models


class SearchRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text query to find similar documents",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0)",
    )

    # Filters (all optional)
    platforms: list[str] | None = Field(
        default=None,
        description="Filter to documents from these platforms (twitter, reddit, news, substack)",
    )
    tickers: list[str] | None = Field(
        default=None,
        description="Filter to documents mentioning these ticker symbols",
    )
    theme_ids: list[str] | None = Field(
        default=None,
        description="Filter to documents with these theme cluster IDs",
    )
    min_authority_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum authority score threshold",
    )


class SearchResultItem(BaseModel):
    """Single search result item."""

    document_id: str = Field(
        ...,
        description="Unique document identifier",
    )
    score: float = Field(
        ...,
        description="Similarity score (0.0-1.0, higher is more similar)",
    )
    platform: str | None = Field(
        default=None,
        description="Source platform",
    )
    title: str | None = Field(
        default=None,
        description="Document title if available",
    )
    content_preview: str | None = Field(
        default=None,
        description="First 200 characters of content",
    )
    url: str | None = Field(
        default=None,
        description="Original document URL",
    )
    author_name: str | None = Field(
        default=None,
        description="Author display name",
    )
    author_verified: bool = Field(
        default=False,
        description="Whether author is verified",
    )
    tickers: list[str] = Field(
        default_factory=list,
        description="Ticker symbols mentioned",
    )
    authority_score: float | None = Field(
        default=None,
        description="Document authority score (0.0-1.0)",
    )
    timestamp: str | None = Field(
        default=None,
        description="Document creation timestamp (ISO format)",
    )


class SearchResponse(BaseModel):
    """Response model for semantic search."""

    results: list[SearchResultItem] = Field(
        ...,
        description="Search results sorted by similarity",
    )
    total: int = Field(
        ...,
        description="Number of results returned",
    )
    latency_ms: float = Field(
        ...,
        description="Search latency in milliseconds",
    )
