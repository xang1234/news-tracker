"""Configuration for cited Q&A."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QAConfig(BaseSettings):
    """Q&A knobs. LLM credentials/breaker come from ScoringConfig."""

    model_config = SettingsConfigDict(
        env_prefix="QA_",
        case_sensitive=False,
        extra="ignore",
    )

    max_claims: int = Field(
        default=12,
        ge=1,
        le=50,
        description="How many top claims to retrieve as grounding evidence",
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum claim confidence to include as evidence",
    )
    min_grounding_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Top retrieved-claim similarity below which grounding is "
        "deemed insufficient (the answer refuses)",
    )
    max_segments: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Cap on answer segments (also the templated-fallback cap)",
    )
