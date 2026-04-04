"""Configuration for the filing lane."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class FilingConfig(BaseSettings):
    """Filing lane configuration.

    Attributes:
        primary_provider: Which provider to use as primary ('edgartools' or 'sec_api').
        fallback_enabled: Whether to try the fallback provider on primary failure.
        filing_types: Comma-separated list of filing types to process.
        max_sections_per_filing: Maximum sections to extract per filing.
        min_section_words: Minimum word count to keep a section.
    """

    model_config = SettingsConfigDict(
        env_prefix="FILING_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    primary_provider: str = "edgartools"
    fallback_enabled: bool = True
    filing_types: str = "10-K,10-Q,8-K"
    max_sections_per_filing: int = 50
    min_section_words: int = 20

    @property
    def filing_type_list(self) -> list[str]:
        """Parse comma-separated filing types into a list."""
        return [t.strip() for t in self.filing_types.split(",") if t.strip()]
