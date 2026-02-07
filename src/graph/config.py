"""Configuration for the causal graph service.

All settings can be overridden via environment variables with GRAPH_ prefix.
Example: GRAPH_MAX_TRAVERSAL_DEPTH=5
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GraphConfig(BaseSettings):
    """Causal graph configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GRAPH_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    max_traversal_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum depth for recursive graph traversals",
    )

    default_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Default confidence for new edges",
    )
