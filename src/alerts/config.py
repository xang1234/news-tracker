"""Alert service configuration.

Controls deduplication windows, per-severity rate limits, and trigger
thresholds for all alert types. All settings can be overridden via
``ALERTS_*`` environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlertConfig(BaseSettings):
    """Configuration for the alert generation system."""

    model_config = SettingsConfigDict(
        env_prefix="ALERTS_",
        case_sensitive=False,
        extra="ignore",
    )

    # Deduplication: suppress duplicate (theme_id, trigger_type) pairs
    dedup_ttl_hours: int = Field(
        default=4,
        ge=1,
        le=168,
        description="Hours to suppress duplicate alerts for the same theme + trigger",
    )

    # Per-severity daily rate limits (0 = unlimited)
    daily_limit_critical: int = Field(
        default=5,
        ge=0,
        description="Max critical alerts per day (0 = unlimited)",
    )
    daily_limit_warning: int = Field(
        default=20,
        ge=0,
        description="Max warning alerts per day (0 = unlimited)",
    )
    daily_limit_info: int = Field(
        default=0,
        ge=0,
        description="Max info alerts per day (0 = unlimited)",
    )

    # Sentiment velocity thresholds
    sentiment_velocity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum abs(delta) in sentiment_score to trigger warning",
    )
    sentiment_velocity_critical: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Abs(delta) in sentiment_score to trigger critical",
    )

    # Extreme sentiment thresholds
    extreme_bullish_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Bullish ratio above which extreme_sentiment fires",
    )
    extreme_bearish_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Bullish ratio below which extreme_sentiment fires",
    )

    # Volume surge thresholds
    volume_surge_threshold: float = Field(
        default=3.0,
        ge=1.0,
        description="Volume z-score above which volume_surge fires (warning)",
    )
    volume_surge_critical: float = Field(
        default=4.0,
        ge=1.0,
        description="Volume z-score above which volume_surge becomes critical",
    )
