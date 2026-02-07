"""Configuration for drift detection and monitoring."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DriftConfig(BaseSettings):
    """Drift detection thresholds and parameters.

    All settings can be overridden via environment variables with the
    ``DRIFT_`` prefix (e.g. ``DRIFT_EMBEDDING_KL_WARNING=0.15``).
    """

    model_config = SettingsConfigDict(
        env_prefix="DRIFT_",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Embedding drift ──────────────────────────────────────
    embedding_baseline_days: int = Field(
        default=30,
        description="Days of historical embeddings for baseline distribution",
    )
    embedding_recent_hours: int = Field(
        default=24,
        description="Hours of recent embeddings to compare against baseline",
    )
    embedding_sample_size: int = Field(
        default=1000,
        description="Max documents to sample per window",
    )
    embedding_kl_warning: float = Field(
        default=0.1,
        description="KL divergence threshold for warning",
    )
    embedding_kl_critical: float = Field(
        default=0.2,
        description="KL divergence threshold for critical",
    )
    embedding_num_bins: int = Field(
        default=50,
        description="Number of histogram bins for L2 norm distribution",
    )

    # ── Theme fragmentation ──────────────────────────────────
    fragmentation_lookback_days: int = Field(
        default=7,
        description="Days to look back for theme creation rate",
    )
    fragmentation_normal_min: int = Field(
        default=5,
        description="Normal minimum themes created per day",
    )
    fragmentation_normal_max: int = Field(
        default=20,
        description="Normal maximum themes created per day",
    )
    fragmentation_warning: int = Field(
        default=30,
        description="Theme creation rate for warning",
    )
    fragmentation_critical: int = Field(
        default=50,
        description="Theme creation rate for critical",
    )

    # ── Sentiment calibration ────────────────────────────────
    sentiment_baseline_days: int = Field(
        default=30,
        description="Days of baseline for sentiment z-score",
    )
    sentiment_zscore_warning: float = Field(
        default=2.0,
        description="Z-score threshold for warning",
    )
    sentiment_zscore_critical: float = Field(
        default=3.0,
        description="Z-score threshold for critical",
    )

    # ── Cluster stability ────────────────────────────────────
    stability_lookback_days: int = Field(
        default=7,
        description="Days of recent docs to compare against stored centroids",
    )
    stability_warning: float = Field(
        default=0.1,
        description="Average cosine distance for warning",
    )
    stability_critical: float = Field(
        default=0.2,
        description="Average cosine distance for critical",
    )
