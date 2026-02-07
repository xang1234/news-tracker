"""Configuration for the security master service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecurityMasterConfig(BaseSettings):
    """Settings for security master database and caching."""

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_MASTER_",
        case_sensitive=False,
        extra="ignore",
    )

    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="TTL for in-memory ticker/company caches (0 = no caching)",
    )
    fuzzy_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum pg_trgm similarity score for fuzzy name search",
    )
    seed_on_init: bool = Field(
        default=True,
        description="Automatically seed from JSON on first init if table is empty",
    )
