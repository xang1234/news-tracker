"""Configuration for the sources service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SourcesConfig(BaseSettings):
    """Settings for source management database and caching."""

    model_config = SettingsConfigDict(
        env_prefix="SOURCES_",
        case_sensitive=False,
        extra="ignore",
    )

    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="TTL for in-memory source caches (0 = no caching)",
    )
    seed_on_init: bool = Field(
        default=True,
        description="Automatically seed from JSON on first init if table is empty",
    )
