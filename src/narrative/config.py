"""Configuration for narrative momentum processing."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NarrativeConfig(BaseSettings):
    """Settings for narrative queues, scoring, and maintenance."""

    model_config = SettingsConfigDict(
        env_prefix="NARRATIVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    stream_name: str = Field(default="narrative_queue")
    consumer_group: str = Field(default="narrative_workers")
    dlq_stream_name: str = Field(default="narrative_queue:dlq")
    max_stream_length: int = Field(default=50_000)
    idle_timeout_ms: int = Field(default=30_000, ge=1_000, le=300_000)
    max_delivery_attempts: int = Field(default=3, ge=1, le=10)

    batch_size: int = Field(default=32, ge=1, le=256)
    worker_batch_timeout: float = Field(default=10.0, ge=1.0, le=60.0)
    maintenance_interval_seconds: int = Field(default=900, ge=60, le=86_400)

    similarity_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    merge_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    candidate_limit: int = Field(default=5, ge=1, le=20)
    bucket_minutes: int = Field(default=5, ge=1, le=60)
    cooling_hours: int = Field(default=2, ge=1, le=168)
    close_hours: int = Field(default=24, ge=1, le=720)
    high_authority_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    surge_window_minutes: int = Field(default=30, ge=5, le=180)
    surge_baseline_hours: int = Field(default=6, ge=1, le=48)
    surge_trigger_zscore: float = Field(default=3.0)
    surge_reset_zscore: float = Field(default=1.5)
    surge_trigger_uplift: float = Field(default=2.5)
    surge_min_total_docs: int = Field(default=8, ge=1)
    surge_min_rate_per_hour: float = Field(default=6.0, ge=0.0)

    cross_platform_window_hours: int = Field(default=6, ge=1, le=48)
    cross_platform_min_docs: int = Field(default=5, ge=1)

    authority_window_hours: int = Field(default=12, ge=1, le=72)
    authority_gap_trigger: float = Field(default=0.45)
    authority_gap_reset: float = Field(default=0.20)

    sentiment_recent_hours: int = Field(default=6, ge=1, le=48)
    sentiment_prior_hours: int = Field(default=12, ge=1, le=96)
    sentiment_shift_trigger: float = Field(default=0.35)
    sentiment_shift_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    sentiment_min_docs: int = Field(default=6, ge=1)
