"""Configuration for the compellingness scoring service.

Provides Pydantic settings for LLM API keys, model selection, tier thresholds,
budget caps, caching, and circuit breaker tuning. All settings can be overridden
via SCORING_* environment variables.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScoringConfig(BaseSettings):
    """Configuration for the LLM compellingness scoring pipeline.

    Settings can be overridden via environment variables prefixed with SCORING_.

    Example:
        SCORING_OPENAI_API_KEY=sk-...
        SCORING_ANTHROPIC_API_KEY=sk-ant-...
        SCORING_TIER2_MIN_RULE_SCORE=4.0
    """

    model_config = SettingsConfigDict(
        env_prefix="SCORING_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM API keys
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for GPT-4o-mini scoring",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key for Claude validation",
    )

    # Model selection
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for Tier 2 scoring",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Anthropic model for Tier 3 validation",
    )

    # Tier gating thresholds
    tier2_min_rule_score: float = Field(
        default=3.0,
        ge=0.0,
        le=10.0,
        description="Minimum rule-based score to advance to Tier 2 (GPT)",
    )
    tier3_min_gpt_score: float = Field(
        default=8.5,
        ge=0.0,
        le=10.0,
        description="Minimum GPT score to advance to Tier 3 (Claude)",
    )
    consensus_tolerance: float = Field(
        default=1.5,
        ge=0.0,
        le=5.0,
        description="Max |claude - gpt| before flagging needs_human_review",
    )

    # Budget caps (USD per day)
    daily_budget_openai: float = Field(
        default=5.0,
        ge=0.0,
        description="Daily spending cap for OpenAI API (USD)",
    )
    daily_budget_anthropic: float = Field(
        default=2.0,
        ge=0.0,
        description="Daily spending cap for Anthropic API (USD)",
    )

    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable Redis caching for scoring results",
    )
    cache_key_prefix: str = Field(
        default="scoring:",
        description="Redis key prefix for cached scores",
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        description="Cache TTL in hours",
    )

    # Circuit breaker settings
    circuit_failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive failures before opening circuit",
    )
    circuit_recovery_timeout: float = Field(
        default=60.0,
        ge=5.0,
        description="Seconds before attempting recovery probe",
    )

    # LLM request timeout
    llm_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Timeout in seconds for LLM API calls",
    )

    @property
    def cache_ttl_seconds(self) -> int:
        """Get cache TTL in seconds."""
        return self.cache_ttl_hours * 3600
