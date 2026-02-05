"""Configuration for the keywords service.

Uses Pydantic settings for environment-based configuration,
following the same pattern as other service configs in the project.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KeywordsConfig(BaseSettings):
    """
    Configuration for the Keyword Extraction service.

    All settings can be overridden via environment variables with KEYWORDS_ prefix.
    Example: KEYWORDS_TOP_N=15

    Attributes:
        top_n: Maximum number of keywords to extract per document.
        language: Language for stopword filtering (default: English).
        min_score: Minimum TextRank score for keyword inclusion.
        max_text_length: Maximum text length to process (truncate longer texts).
    """

    model_config = SettingsConfigDict(
        env_prefix="KEYWORDS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model configuration
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model for tokenization and POS tagging.",
    )

    # Extraction configuration
    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum keywords to extract per document.",
    )
    language: str = Field(
        default="en",
        description="Language code for stopword filtering.",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum TextRank score for keyword inclusion.",
    )

    # Performance configuration
    max_text_length: int = Field(
        default=10000,
        ge=100,
        description="Maximum text length to process. Longer texts are truncated.",
    )
