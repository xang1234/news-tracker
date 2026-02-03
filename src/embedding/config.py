"""
Embedding service configuration.

Provides Pydantic settings for the FinBERT embedding service including
model configuration, batching, caching, and Redis stream settings.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """
    Configuration for the embedding generation service.

    Settings can be overridden via environment variables prefixed with EMBEDDING_.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # FinBERT model configuration (768-dim, financial domain)
    model_name: str = Field(
        default="ProsusAI/finbert",
        description="HuggingFace model name for FinBERT embeddings",
    )
    embedding_dim: int = Field(
        default=768,
        description="Embedding vector dimension (768 for FinBERT)",
    )

    # MiniLM model configuration (384-dim, lightweight/fast)
    minilm_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name for MiniLM embeddings",
    )
    minilm_embedding_dim: int = Field(
        default=384,
        description="Embedding vector dimension (384 for MiniLM)",
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum token sequence length for the model",
    )

    # Processing configuration
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Number of documents to embed per batch",
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 (half precision) for GPU acceleration",
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device for model inference (auto detects best available)",
    )

    # Chunking configuration for long documents
    chunk_overlap: int = Field(
        default=50,
        description="Number of overlapping tokens between chunks",
    )
    pooling_strategy: Literal["mean", "max", "cls"] = Field(
        default="mean",
        description="Strategy for pooling chunk embeddings",
    )

    # Redis stream configuration
    stream_name: str = Field(
        default="embedding_queue",
        description="Redis stream name for embedding jobs",
    )
    consumer_group: str = Field(
        default="embedding_workers",
        description="Consumer group name for embedding workers",
    )
    max_stream_length: int = Field(
        default=50_000,
        description="Maximum stream length before trimming",
    )
    dlq_stream_name: str = Field(
        default="embedding_queue:dlq",
        description="Dead letter queue stream name",
    )

    # Caching configuration
    cache_enabled: bool = Field(
        default=True,
        description="Enable Redis caching for embeddings",
    )
    cache_ttl_hours: int = Field(
        default=168,
        ge=1,
        description="Cache TTL in hours (default: 1 week)",
    )
    cache_key_prefix: str = Field(
        default="emb:",
        description="Redis key prefix for cached embeddings",
    )

    # Worker configuration
    worker_batch_timeout: float = Field(
        default=5.0,
        description="Timeout in seconds for batch accumulation",
    )
    worker_idle_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for idle worker shutdown",
    )

    # Queue reclaim configuration
    idle_timeout_ms: int = Field(
        default=30_000,
        ge=1_000,
        le=300_000,
        description="Idle time before reclaiming pending messages (ms)",
    )
    max_delivery_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max delivery attempts before moving to DLQ",
    )

    @property
    def cache_ttl_seconds(self) -> int:
        """Get cache TTL in seconds."""
        return self.cache_ttl_hours * 3600
