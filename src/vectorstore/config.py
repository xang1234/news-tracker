"""
Configuration for vector store operations.

Uses Pydantic BaseSettings for environment variable support.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreConfig(BaseSettings):
    """
    Configuration for VectorStore and VectorStoreManager.

    All settings can be overridden via environment variables with
    VECTORSTORE_ prefix (e.g., VECTORSTORE_DEFAULT_LIMIT=20).
    """

    # Search defaults
    default_limit: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Default number of results to return",
    )
    default_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default minimum similarity threshold",
    )

    # Centroid search (for BERTopic/clustering)
    centroid_default_limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Default limit for centroid searches",
    )
    centroid_default_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default threshold for centroid searches (lower than normal)",
    )

    # Authority score weights for computation
    authority_verified_bonus: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Authority bonus for verified authors",
    )
    authority_follower_max: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum authority score from follower count",
    )
    authority_engagement_max: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum authority score from engagement",
    )
    authority_spam_penalty_max: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum penalty for spam score",
    )

    # Follower scaling (log base)
    authority_follower_scale: int = Field(
        default=1_000_000,
        ge=1000,
        description="Follower count that gives max follower score",
    )

    # Engagement scaling (log base)
    authority_engagement_scale: int = Field(
        default=10_000,
        ge=100,
        description="Engagement score that gives max engagement bonus",
    )

    # Batch processing
    upsert_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for upsert operations",
    )

    model_config = SettingsConfigDict(env_prefix="VECTORSTORE_")
