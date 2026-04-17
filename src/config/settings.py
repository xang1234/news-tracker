"""Application settings using Pydantic Settings for environment-based configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, model_validator
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

    # XUI (Twitter browser ingestion with adaptive guardrails)
    twitter_xui_enabled: bool = Field(
        default=False,
        description="Enable xui browser ingestion for Twitter data",
    )
    twitter_xui_command: str = Field(
        default="xui",
        description="Command to invoke xui CLI (e.g. 'xui' or 'uv run --extra cli xui')",
    )
    twitter_xui_config_path: str | None = Field(
        default=None,
        description="Optional xui config.toml path (defaults to xui CLI standard location)",
    )
    twitter_xui_profile: str = Field(
        default="default",
        description="xui profile name used for authenticated storage state",
    )
    twitter_xui_profile_dir: str | None = Field(
        default=None,
        description="Optional profile directory hint for operators; used for runbook visibility",
    )
    twitter_xui_usernames: str | None = Field(
        default=None,
        description="Comma-separated X usernames to track with xui (overrides defaults)",
    )

    # XUI polling + anti-automation cadence
    twitter_xui_poll_min_seconds: int = Field(default=120, ge=30, le=86_400)
    twitter_xui_poll_max_seconds: int = Field(default=300, ge=30, le=86_400)
    twitter_xui_cycle_jitter_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    twitter_xui_shuffle_sources: bool = Field(default=True)
    twitter_xui_source_cooldown_cycles: int = Field(default=2, ge=0, le=20)

    # XUI collection guardrails
    twitter_xui_limit_per_source: int = Field(default=50, ge=1, le=200)
    twitter_xui_scroll_pause_min_ms: int = Field(default=1400, ge=250, le=60_000)
    twitter_xui_scroll_pause_max_ms: int = Field(default=3200, ge=250, le=60_000)
    twitter_xui_max_scroll_rounds: int = Field(default=4, ge=1, le=200)
    twitter_xui_max_page_loads: int = Field(default=2, ge=1, le=200)
    twitter_xui_timeout_ms: int = Field(default=90_000, ge=10_000, le=600_000)
    twitter_xui_source_pause_min_seconds: float = Field(default=0.8, ge=0.0, le=60.0)
    twitter_xui_source_pause_max_seconds: float = Field(default=2.5, ge=0.0, le=60.0)

    # XUI block/challenge resilience
    twitter_xui_block_backoff_initial_seconds: int = Field(default=300, ge=30, le=86_400)
    twitter_xui_block_backoff_max_seconds: int = Field(default=3600, ge=60, le=86_400)
    twitter_xui_block_circuit_threshold: int = Field(default=3, ge=1, le=50)
    twitter_xui_block_circuit_open_seconds: int = Field(default=7200, ge=60, le=604_800)

    # Rate limits (requests per minute)
    twitter_rate_limit: int = 30
    reddit_rate_limit: int = 60
    news_rate_limit: int = 60
    substack_rate_limit: int = 10

    # Processing thresholds
    spam_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    poll_interval_seconds: int = 3600

    # HTTP retry configuration
    max_http_retries: int = Field(default=3, ge=0, le=10)
    max_backoff_seconds: float = Field(default=60.0, ge=1.0, le=300.0)

    # Embedding service
    embedding_model_name: str = "ProsusAI/finbert"
    embedding_batch_size: int = 32
    embedding_use_fp16: bool = True
    embedding_backend: str = "auto"
    embedding_device: str = "auto"  # auto, cpu, cuda, mps
    embedding_execution_provider: str = "auto"
    embedding_onnx_model_path: str | None = None
    embedding_minilm_onnx_model_path: str | None = None
    embedding_stream_name: str = "embedding_queue"
    embedding_consumer_group: str = "embedding_workers"
    embedding_cache_enabled: bool = True
    embedding_cache_ttl_hours: int = 168  # 1 week

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_keys: str | None = None  # Comma-separated API keys, None = no auth (dev mode)

    # CORS
    cors_origins: str = Field(
        default=(
            "http://localhost:5173,"
            "http://127.0.0.1:5173,"
            "http://localhost:4173,"
            "http://127.0.0.1:4173"
        ),
        description="Comma-separated allowed CORS origins",
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")

    # Request timeout
    request_timeout_seconds: float = Field(
        default=30.0, ge=5.0, le=300.0, description="Request timeout in seconds (0 to disable)"
    )

    # API rate limiting (slowapi)
    rate_limit_enabled: bool = Field(
        default=False, description="Enable API rate limiting via slowapi"
    )
    rate_limit_default: str = Field(
        default="60/minute", description="Default rate limit for all endpoints"
    )
    rate_limit_embed: str = Field(default="30/minute", description="Rate limit for /embed endpoint")
    rate_limit_sentiment: str = Field(
        default="30/minute", description="Rate limit for /sentiment endpoint"
    )
    rate_limit_search: str = Field(
        default="60/minute", description="Rate limit for /search/similar endpoint"
    )
    rate_limit_graph: str = Field(
        default="30/minute", description="Rate limit for graph subgraph endpoint"
    )
    rate_limit_entities: str = Field(
        default="60/minute", description="Rate limit for entity endpoints"
    )
    rate_limit_admin: str = Field(
        default="30/minute", description="Rate limit for admin write operations"
    )

    # Worker resilience
    worker_max_consecutive_failures: int = Field(
        default=10, ge=1, le=100, description="Max consecutive worker failures before exit"
    )
    worker_backoff_base_delay: float = Field(
        default=2.0, ge=0.5, le=30.0, description="Base delay for worker exponential backoff"
    )
    worker_backoff_max_delay: float = Field(
        default=120.0, ge=10.0, le=600.0, description="Max delay for worker exponential backoff"
    )

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
    keywords_enabled: bool = Field(
        default=False, description="Enable keyword extraction in preprocessing"
    )

    # Event extraction
    events_enabled: bool = Field(
        default=False, description="Enable event extraction in preprocessing"
    )
    keywords_top_n: int = Field(default=10, ge=1, le=50, description="Max keywords per document")

    # Volume Metrics
    volume_metrics_enabled: bool = Field(
        default=False, description="Enable volume metrics computation for themes"
    )

    # Theme Ranking
    ranking_enabled: bool = Field(
        default=False, description="Enable theme ranking engine for actionability scoring"
    )

    # Alerts
    alerts_enabled: bool = Field(
        default=False, description="Enable alert generation in daily clustering"
    )

    # Notifications
    notifications_enabled: bool = Field(
        default=False, description="Enable notification delivery for alerts"
    )

    # Causal Graph
    graph_enabled: bool = Field(
        default=False, description="Enable causal graph for supply chain modeling"
    )

    # Sentiment Propagation
    propagation_enabled: bool = Field(
        default=False, description="Enable sentiment propagation through causal graph"
    )

    # Backtest
    backtest_enabled: bool = Field(default=False, description="Enable backtest data infrastructure")

    # Scoring (LLM compellingness)
    scoring_enabled: bool = Field(
        default=False, description="Enable LLM compellingness scoring for themes"
    )

    # Security Master
    security_master_enabled: bool = Field(
        default=False, description="Enable database-backed security master for tickers"
    )

    # Sources
    sources_enabled: bool = Field(
        default=False, description="Enable database-backed source management"
    )
    sources_trigger_lock_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86_400,
        description="TTL for the distributed manual-ingestion lock",
    )

    # Drift Detection
    drift_enabled: bool = Field(default=False, description="Enable drift detection and monitoring")

    # Feedback
    feedback_enabled: bool = Field(
        default=False, description="Enable user feedback collection for quality calibration"
    )

    # Authority Scoring
    authority_enabled: bool = Field(
        default=False, description="Enable Bayesian authority scoring for content authors"
    )

    # Tracing (OpenTelemetry)
    tracing_enabled: bool = Field(
        default=False, description="Enable OpenTelemetry distributed tracing"
    )

    # WebSocket Alerts
    ws_alerts_enabled: bool = Field(
        default=False, description="Enable WebSocket streaming for real-time alerts"
    )
    ws_alerts_max_connections: int = Field(
        default=100, ge=1, le=10000, description="Max concurrent WS connections"
    )
    ws_alerts_heartbeat_seconds: int = Field(
        default=30, ge=5, le=300, description="Heartbeat ping interval"
    )

    # Clustering (BERTopic)
    clustering_enabled: bool = Field(
        default=False, description="Enable BERTopic clustering in processing"
    )
    clustering_stream_name: str = Field(
        default="clustering_queue", description="Redis stream for clustering jobs"
    )
    clustering_consumer_group: str = Field(
        default="clustering_workers", description="Consumer group for clustering workers"
    )

    # Narrative momentum
    narrative_enabled: bool = Field(
        default=False, description="Enable narrative momentum processing"
    )

    # Narrative claim extraction (events/entities → evidence claims)
    narrative_claim_extraction_enabled: bool = Field(
        default=False,
        description="Enable evidence claim extraction from document events and entities",
    )

    # Sentiment Analysis
    sentiment_model_name: str = Field(
        default="ProsusAI/finbert", description="Model for sentiment analysis"
    )
    sentiment_batch_size: int = Field(
        default=16, ge=1, le=64, description="Batch size for sentiment analysis"
    )
    sentiment_use_fp16: bool = Field(default=True, description="Use FP16 for GPU acceleration")
    sentiment_backend: str = Field(
        default="auto", description="Inference backend (auto, torch, onnx)"
    )
    sentiment_device: str = Field(
        default="auto", description="Device for inference (auto, cpu, cuda, mps)"
    )
    sentiment_execution_provider: str = Field(
        default="auto", description="ONNX execution provider override (auto, cpu, cuda, coreml)"
    )
    sentiment_onnx_model_path: str | None = Field(
        default=None, description="Path to exported ONNX sentiment model directory"
    )
    sentiment_stream_name: str = Field(
        default="sentiment_queue", description="Redis stream for sentiment jobs"
    )
    sentiment_consumer_group: str = Field(
        default="sentiment_workers", description="Consumer group for sentiment workers"
    )
    sentiment_cache_enabled: bool = Field(
        default=True, description="Enable Redis caching for sentiment"
    )
    sentiment_cache_ttl_hours: int = Field(default=168, description="Cache TTL in hours (1 week)")
    sentiment_enable_entity_sentiment: bool = Field(
        default=True, description="Enable entity-level sentiment"
    )

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
    def cors_origin_list(self) -> list[str]:
        """Return configured CORS origins as a parsed list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def twitter_configured(self) -> bool:
        """Check if Twitter API is configured."""
        return self.twitter_bearer_token is not None

    @property
    def xui_configured(self) -> bool:
        """Check if xui ingestion is enabled."""
        return self.twitter_xui_enabled

    @property
    def reddit_configured(self) -> bool:
        """Check if Reddit API is configured."""
        return self.reddit_client_id is not None and self.reddit_client_secret is not None

    @property
    def news_api_configured(self) -> bool:
        """Check if at least one news API is configured."""
        return any(
            [
                self.finnhub_api_key,
                self.newsapi_api_key,
                self.alpha_vantage_api_key,
                self.newsfilter_api_keys,
                self.marketaux_api_keys,
                self.finlight_api_keys,
            ]
        )

    @model_validator(mode="after")
    def validate_cors_settings(self) -> "Settings":
        """Reject invalid wildcard-plus-credentials CORS settings."""
        if self.cors_allow_credentials and "*" in self.cors_origin_list:
            raise ValueError(
                "cors_allow_credentials=True is incompatible with wildcard CORS origins"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    Clear cache with get_settings.cache_clear() if needed.
    """
    return Settings()
