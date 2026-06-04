"""Configuration for the claim retrieval substrate."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClaimRetrievalConfig(BaseSettings):
    """Settings for semantic retrieval over the structured claim layer."""

    model_config = SettingsConfigDict(
        env_prefix="CLAIM_RETRIEVAL_",
        case_sensitive=False,
        extra="ignore",
    )

    default_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of claims a retrieval call returns",
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a claim to be returned",
    )
    index_batch_size: int = Field(
        default=128,
        ge=1,
        le=512,
        description="Number of un-embedded claims to embed per backfill batch",
    )
