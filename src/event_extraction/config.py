"""Configuration for the event extraction service.

Uses Pydantic settings for environment-based configuration,
following the same pattern as other service configs in the project.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EventExtractionConfig(BaseSettings):
    """
    Configuration for the Event Extraction service.

    All settings can be overridden via environment variables with EVENTS_ prefix.
    Example: EVENTS_MIN_CONFIDENCE=0.6

    Attributes:
        extractor_version: Version string for extractor provenance tracking.
        min_confidence: Minimum confidence threshold for event inclusion.
        max_events_per_doc: Maximum events to extract from a single document.
    """

    model_config = SettingsConfigDict(
        env_prefix="EVENTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    extractor_version: str = Field(
        default="1.0.0",
        description="Version string for extractor provenance tracking.",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for event inclusion.",
    )
    max_events_per_doc: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum events to extract from a single document.",
    )
