"""Configuration for the theme briefing generator."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BriefingConfig(BaseSettings):
    """Briefing-specific knobs. LLM credentials/breaker come from ScoringConfig."""

    model_config = SettingsConfigDict(
        env_prefix="BRIEFING_",
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
    max_clauses: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Cap on clauses in a briefing (also the templated-fallback cap)",
    )
