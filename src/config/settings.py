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

    # Sotwe fallback (Twitter alternative when no API key)
    sotwe_enabled: bool = Field(
        default=True,
        description="Enable Sotwe.com fallback when Twitter API unavailable"
    )
    sotwe_usernames: str | None = Field(
        default=None,
        description="Comma-separated Twitter usernames to track via Sotwe (overrides defaults)"
    )
    sotwe_rate_limit: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Requests per minute for Sotwe scraping (conservative)"
    )

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

    # Embedding service
    embedding_model_name: str = "ProsusAI/finbert"
    embedding_batch_size: int = 32
    embedding_use_fp16: bool = True
    embedding_device: str = "auto"  # auto, cpu, cuda, mps
    embedding_stream_name: str = "embedding_queue"
    embedding_consumer_group: str = "embedding_workers"
    embedding_cache_enabled: bool = True
    embedding_cache_ttl_hours: int = 168  # 1 week

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_keys: str | None = None  # Comma-separated API keys, None = no auth (dev mode)

    # Observability
    metrics_port: int = 8000
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "news-tracker"

    # Vector store
    vectorstore_default_limit: int = 10
    vectorstore_default_threshold: float = 0.7
    vectorstore_centroid_limit: int = 100
    vectorstore_centroid_threshold: float = 0.5

    # NER (Named Entity Recognition)
    ner_enabled: bool = Field(default=False, description="Enable NER extraction in preprocessing")
    ner_spacy_model: str = Field(default="en_core_web_trf", description="spaCy model for NER")

    # Keywords extraction
    keywords_enabled: bool = Field(default=False, description="Enable keyword extraction in preprocessing")

    # Event extraction
    events_enabled: bool = Field(default=False, description="Enable event extraction in preprocessing")
    keywords_top_n: int = Field(default=10, ge=1, le=50, description="Max keywords per document")

    # Volume Metrics
    volume_metrics_enabled: bool = Field(default=False, description="Enable volume metrics computation for themes")

    # Theme Ranking
    ranking_enabled: bool = Field(default=False, description="Enable theme ranking engine for actionability scoring")

    # Alerts
    alerts_enabled: bool = Field(default=False, description="Enable alert generation in daily clustering")

    # Notifications
    notifications_enabled: bool = Field(default=False, description="Enable notification delivery for alerts")

    # Causal Graph
    graph_enabled: bool = Field(default=False, description="Enable causal graph for supply chain modeling")

    # Sentiment Propagation
    propagation_enabled: bool = Field(default=False, description="Enable sentiment propagation through causal graph")

    # Backtest
    backtest_enabled: bool = Field(default=False, description="Enable backtest data infrastructure")

    # Scoring (LLM compellingness)
    scoring_enabled: bool = Field(default=False, description="Enable LLM compellingness scoring for themes")

    # Drift Detection
    drift_enabled: bool = Field(default=False, description="Enable drift detection and monitoring")

    # Clustering (BERTopic)
    clustering_enabled: bool = Field(default=False, description="Enable BERTopic clustering in processing")
    clustering_stream_name: str = Field(default="clustering_queue", description="Redis stream for clustering jobs")
    clustering_consumer_group: str = Field(default="clustering_workers", description="Consumer group for clustering workers")

    # Sentiment Analysis
    sentiment_model_name: str = Field(default="ProsusAI/finbert", description="Model for sentiment analysis")
    sentiment_batch_size: int = Field(default=16, ge=1, le=64, description="Batch size for sentiment analysis")
    sentiment_use_fp16: bool = Field(default=True, description="Use FP16 for GPU acceleration")
    sentiment_device: str = Field(default="auto", description="Device for inference (auto, cpu, cuda, mps)")
    sentiment_stream_name: str = Field(default="sentiment_queue", description="Redis stream for sentiment jobs")
    sentiment_consumer_group: str = Field(default="sentiment_workers", description="Consumer group for sentiment workers")
    sentiment_cache_enabled: bool = Field(default=True, description="Enable Redis caching for sentiment")
    sentiment_cache_ttl_hours: int = Field(default=168, description="Cache TTL in hours (1 week)")
    sentiment_enable_entity_sentiment: bool = Field(default=True, description="Enable entity-level sentiment")

    # Queue reclaim configuration (applies to all Redis Streams queues)
    queue_idle_timeout_ms: int = Field(
        default=30_000,
        ge=1_000,
        le=300_000,
        description="Idle time before reclaiming pending messages (ms)",
    )
    queue_max_delivery_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max delivery attempts before moving to DLQ",
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def twitter_configured(self) -> bool:
        """Check if Twitter API is configured."""
        return self.twitter_bearer_token is not None

    @property
    def sotwe_configured(self) -> bool:
        """Check if Sotwe fallback is enabled and available."""
        return self.sotwe_enabled

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
