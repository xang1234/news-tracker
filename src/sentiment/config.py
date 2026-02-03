"""
Sentiment analysis service configuration.

Provides Pydantic settings for the FinBERT sentiment analysis service including
model configuration, batching, caching, and Redis stream settings.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SentimentConfig(BaseSettings):
    """
    Configuration for the sentiment analysis service.

    Settings can be overridden via environment variables prefixed with SENTIMENT_.

    Example:
        SENTIMENT_MODEL_NAME=ProsusAI/finbert
        SENTIMENT_BATCH_SIZE=16
        SENTIMENT_USE_FP16=true
    """

    model_config = SettingsConfigDict(
        env_prefix="SENTIMENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model configuration
    model_name: str = Field(
        default="ProsusAI/finbert",
        description="HuggingFace model name for sentiment classification",
    )
    max_sequence_length: int = Field(
        default=512,
        ge=128,
        le=512,
        description="Maximum token sequence length for the model",
    )

    # Processing configuration
    batch_size: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Number of documents to classify per batch",
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 (half precision) for GPU acceleration",
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device for model inference (auto detects best available)",
    )

    # Entity-level sentiment configuration
    enable_entity_sentiment: bool = Field(
        default=True,
        description="Extract entity-specific sentiment from context windows",
    )
    entity_context_window: int = Field(
        default=100,
        ge=50,
        le=300,
        description="Character window around entity for entity-level sentiment",
    )

    # Redis stream configuration
    stream_name: str = Field(
        default="sentiment_queue",
        description="Redis stream name for sentiment jobs",
    )
    consumer_group: str = Field(
        default="sentiment_workers",
        description="Consumer group name for sentiment workers",
    )
    max_stream_length: int = Field(
        default=50_000,
        description="Maximum stream length before trimming",
    )
    dlq_stream_name: str = Field(
        default="sentiment_queue:dlq",
        description="Dead letter queue stream name",
    )

    # Emoji modifier configuration
    emoji_modifier_enabled: bool = Field(
        default=True,
        description="Apply emoji-based sentiment modifiers to scores",
    )
    emoji_modifier_max: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum emoji adjustment range [-max, +max]",
    )

    # Temporal aggregation configuration
    aggregation_decay_rate: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Exponential decay rate for recency weighting (0.3 = ~2.3 day half-life)",
    )
    aggregation_authority_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for authority score in aggregation (0-1)",
    )
    aggregation_confidence_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for confidence score in aggregation (0-1)",
    )

    # Extreme sentiment detection thresholds
    extreme_bullish_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Bullish ratio above which sentiment is considered extreme",
    )
    extreme_bearish_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Bullish ratio below which sentiment is considered extreme bearish",
    )

    # Caching configuration
    cache_enabled: bool = Field(
        default=True,
        description="Enable Redis caching for sentiment results",
    )
    cache_ttl_hours: int = Field(
        default=168,  # 1 week
        ge=1,
        description="Cache TTL in hours",
    )
    cache_key_prefix: str = Field(
        default="sentiment:",
        description="Redis key prefix for cached sentiment",
    )

    # Worker configuration
    worker_batch_timeout: float = Field(
        default=3.0,
        ge=1.0,
        le=30.0,
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
