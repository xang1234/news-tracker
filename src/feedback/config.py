"""Feedback service configuration.

Controls constraints on feedback submission. All settings can be
overridden via ``FEEDBACK_*`` environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeedbackConfig(BaseSettings):
    """Configuration for the feedback system."""

    model_config = SettingsConfigDict(
        env_prefix="FEEDBACK_",
        case_sensitive=False,
        extra="ignore",
    )

    max_comment_length: int = Field(
        default=2000,
        ge=0,
        le=10000,
        description="Maximum length for feedback comments",
    )
