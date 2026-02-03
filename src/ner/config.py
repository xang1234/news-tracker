"""Configuration for the NER service.

Uses Pydantic settings for environment-based configuration,
following the same pattern as other service configs in the project.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NERConfig(BaseSettings):
    """
    Configuration for the Named Entity Recognition service.

    All settings can be overridden via environment variables with NER_ prefix.
    Example: NER_SPACY_MODEL=en_core_web_sm

    Attributes:
        spacy_model: spaCy model to use. Transformer model (en_core_web_trf)
            provides higher accuracy but requires more memory (~500MB).
            Use en_core_web_sm for faster, lighter processing.
        patterns_dir: Directory containing EntityRuler JSONL pattern files.
        fuzzy_threshold: Minimum score (0-100) for fuzzy entity matching.
        enable_coreference: Whether to resolve coreferences ("the company" -> "Nvidia").
        batch_size: Number of texts to process in parallel.
        max_text_length: Maximum text length to process (truncate longer texts).
        confidence_threshold: Minimum confidence score for entity inclusion.
    """

    model_config = SettingsConfigDict(
        env_prefix="NER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model configuration
    spacy_model: str = Field(
        default="en_core_web_trf",
        description="spaCy model name. Use 'en_core_web_sm' for faster processing.",
    )
    fallback_model: str = Field(
        default="en_core_web_sm",
        description="Fallback model if primary model fails to load.",
    )

    # Pattern configuration
    patterns_dir: Path = Field(
        default=Path(__file__).parent / "patterns",
        description="Directory containing EntityRuler JSONL pattern files.",
    )

    # Matching configuration
    fuzzy_threshold: int = Field(
        default=85,
        ge=0,
        le=100,
        description="Minimum fuzzy match score (0-100) for entity variations.",
    )
    enable_coreference: bool = Field(
        default=True,
        description="Enable coreference resolution to link pronouns to entities.",
    )

    # Performance configuration
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Number of texts to process in parallel.",
    )
    max_text_length: int = Field(
        default=10000,
        ge=100,
        description="Maximum text length to process. Longer texts are truncated.",
    )

    # Quality configuration
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for including an entity.",
    )

    # Entity types to extract
    extract_types: list[Literal["TICKER", "COMPANY", "PRODUCT", "TECHNOLOGY", "METRIC"]] = Field(
        default=["TICKER", "COMPANY", "PRODUCT", "TECHNOLOGY", "METRIC"],
        description="Entity types to extract.",
    )

    # Theme linking configuration
    enable_semantic_linking: bool = Field(
        default=False,
        description="Enable embedding-based semantic similarity for entity-theme linking. "
        "Requires EmbeddingService to be injected.",
    )
    semantic_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for semantic theme matching.",
    )
    semantic_base_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Base score weight for semantic similarity in theme linking.",
    )
