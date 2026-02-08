"""Configuration for the Bayesian authority scoring service.

Controls scoring formula parameters: Beta prior, time decay, probation ramp,
and base weight tiers. All settings can be overridden via ``AUTHORITY_*``
environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthorityConfig(BaseSettings):
    """Configuration for the authority scoring pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="AUTHORITY_",
        case_sensitive=False,
        extra="ignore",
    )

    # Beta prior for Bayesian accuracy smoothing
    prior_alpha: float = Field(
        default=2.0,
        ge=0.1,
        description="Beta prior alpha (pseudo-correct). Higher = more optimistic prior.",
    )
    prior_beta: float = Field(
        default=5.0,
        ge=0.1,
        description="Beta prior beta (pseudo-total - alpha). Higher = stronger regularization.",
    )

    # Time decay on recency of last good call
    decay_lambda: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Exponential decay rate per day for recency multiplier.",
    )

    # Probation ramp for new sources
    probation_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days before a new source reaches full authority weight.",
    )

    # Base weight tiers
    weight_anonymous: float = Field(
        default=1.0,
        ge=0.0,
        description="Base weight for anonymous/unverified authors.",
    )
    weight_verified: float = Field(
        default=5.0,
        ge=0.0,
        description="Base weight for verified professional authors.",
    )
    weight_research: float = Field(
        default=10.0,
        ge=0.0,
        description="Base weight for specialized research outlets.",
    )

    # Follower-based scaling
    follower_log_cap: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Max follower contribution to normalized score.",
    )
    follower_log_base: int = Field(
        default=1_000_000,
        ge=100,
        description="Follower count that maps to follower_log_cap.",
    )

    # Output clamping
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Floor for computed authority score.",
    )
    max_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ceiling for computed authority score.",
    )
