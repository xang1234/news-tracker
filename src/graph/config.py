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

    # Sentiment propagation settings
    propagation_default_decay: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Global decay factor applied per hop during sentiment propagation",
    )
    propagation_max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum hops for sentiment propagation through the graph",
    )
    propagation_min_impact: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum absolute impact to include in propagation results",
    )

    # Edge type weights for propagation (sign indicates direction of effect)
    propagation_weight_depends_on: float = Field(
        default=0.8,
        description="Propagation weight for depends_on edges",
    )
    propagation_weight_supplies_to: float = Field(
        default=0.6,
        description="Propagation weight for supplies_to edges",
    )
    propagation_weight_competes_with: float = Field(
        default=-0.3,
        description="Propagation weight for competes_with edges (negative = inverse)",
    )
    propagation_weight_drives: float = Field(
        default=0.5,
        description="Propagation weight for drives edges",
    )
    propagation_weight_blocks: float = Field(
        default=-0.4,
        description="Propagation weight for blocks edges (negative = inverse)",
    )
