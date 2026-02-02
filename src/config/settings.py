"""Application settings using Pydantic Settings for environment-based configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the news-tracker application.

    All settings can be overridden via environment variables.
    Prefix is not used to allow standard env var names (e.g., DATABASE_URL).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    debug: bool = False

    # Redis
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_stream_name: str = "raw_documents"
    redis_consumer_group: str = "processing_workers"
    redis_max_stream_length: int = 100_000  # Trim old messages

    # PostgreSQL
    database_url: PostgresDsn = Field(
        default="postgresql://postgres:postgres@localhost:5432/news_tracker"
    )
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20

    # Twitter API v2
    twitter_bearer_token: str | None = None
    twitter_api_key: str | None = None
    twitter_api_secret: str | None = None

    # Reddit API
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "news-tracker/0.1.0"

    # News APIs
    finnhub_api_key: str | None = None
    newsapi_api_key: str | None = None
    alpha_vantage_api_key: str | None = None

    # New news sources (comma-separated for multiple keys with rotation)
    newsfilter_api_keys: str | None = None
    marketaux_api_keys: str | None = None
    finlight_api_keys: str | None = None

    # Substack
    substack_cookie: str | None = None

    # Rate limits (requests per minute)
    twitter_rate_limit: int = 30
    reddit_rate_limit: int = 60
    news_rate_limit: int = 60
    substack_rate_limit: int = 10

    # Processing thresholds
    spam_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    poll_interval_seconds: int = 60

    # HTTP retry configuration
    max_http_retries: int = Field(default=3, ge=0, le=10)
    max_backoff_seconds: float = Field(default=60.0, ge=1.0, le=300.0)

    # Observability
    metrics_port: int = 8000
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "news-tracker"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def twitter_configured(self) -> bool:
        """Check if Twitter API is configured."""
        return self.twitter_bearer_token is not None

    @property
    def reddit_configured(self) -> bool:
        """Check if Reddit API is configured."""
        return (
            self.reddit_client_id is not None
            and self.reddit_client_secret is not None
        )

    @property
    def news_api_configured(self) -> bool:
        """Check if at least one news API is configured."""
        return any([
            self.finnhub_api_key,
            self.newsapi_api_key,
            self.alpha_vantage_api_key,
            self.newsfilter_api_keys,
            self.marketaux_api_keys,
            self.finlight_api_keys,
        ])


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    Clear cache with get_settings.cache_clear() if needed.
    """
    return Settings()
