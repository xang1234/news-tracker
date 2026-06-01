"""Admin API models for sources and securities."""

from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from src.security_master.schemas import normalize_sec_cik

_ALLOWED_SOURCE_PLATFORMS = {"twitter", "reddit", "substack", "rss"}
_RSS_REQUIRED_METADATA = ("url", "category")


def _validate_source_platform(value: str) -> str:
    normalized = value.lower()
    if normalized not in _ALLOWED_SOURCE_PLATFORMS:
        allowed = ", ".join(sorted(_ALLOWED_SOURCE_PLATFORMS))
        raise ValueError(f"platform must be one of: {allowed}")
    return normalized


def _validate_rss_source_metadata(metadata: dict[str, object]) -> None:
    for key in _RSS_REQUIRED_METADATA:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"rss sources require metadata.{key}")

    authority = metadata.get("authority")
    if authority is not None and not isinstance(authority, str):
        raise ValueError("rss metadata.authority must be a string when provided")

    full_text = metadata.get("full_text")
    if full_text is not None and not isinstance(full_text, bool):
        raise ValueError("rss metadata.full_text must be a boolean when provided")


def _normalize_optional_sec_cik(value: str | None) -> str | None:
    if value is None:
        return None
    return normalize_sec_cik(value)


class SecurityIdentifierLineageItem(BaseModel):
    """Auditable provenance for one security identifier."""

    identifier_type: str = Field(..., min_length=1, description="Identifier kind")
    value: str = Field(..., min_length=1, description="Identifier value")
    source: str = Field(..., min_length=1, description="Source that supplied the identifier")
    observed_at: str | None = Field(default=None, description="Observation date or timestamp")
    valid_from: str | None = Field(default=None, description="Known validity start")
    valid_to: str | None = Field(default=None, description="Known validity end")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, object] = Field(default_factory=dict)


class SecurityItem(BaseModel):
    """Single security record."""

    ticker: str = Field(..., description="Ticker symbol")
    exchange: str = Field(..., description="Exchange code (e.g. US, KRX)")
    name: str = Field(..., description="Company name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    sector: str = Field(default="", description="Sector classification")
    country: str = Field(default="US", description="Country code")
    currency: str = Field(default="USD", description="Trading currency")
    sec_cik: str | None = Field(default=None, description="10-digit SEC CIK")
    issuer_name: str = Field(default="", description="Issuer name reported to SEC")
    former_names: list[str] = Field(default_factory=list, description="Former issuer names")
    external_identifiers: dict[str, object] = Field(
        default_factory=dict,
        description="External identifiers such as SEC tickers, LEIs, and FIGI aliases",
    )
    identifier_lineage: list[SecurityIdentifierLineageItem] = Field(default_factory=list)
    is_active: bool = Field(default=True, description="Whether security is active")
    created_at: str | None = Field(default=None, description="Creation timestamp (ISO)")
    updated_at: str | None = Field(default=None, description="Last update timestamp (ISO)")


class SecuritiesListResponse(BaseModel):
    """Paginated securities list."""

    securities: list[SecurityItem] = Field(..., description="Security records")
    total: int = Field(..., description="Total matching securities")
    has_more: bool = Field(..., description="Whether more pages exist")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class CreateSecurityRequest(BaseModel):
    """Request to create a new security."""

    ticker: str = Field(..., min_length=1, max_length=20, description="Ticker symbol")
    exchange: str = Field(default="US", max_length=10, description="Exchange code")
    name: str = Field(..., min_length=1, max_length=200, description="Company name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    sector: str = Field(default="", description="Sector classification")
    country: str = Field(default="US", description="Country code")
    currency: str = Field(default="USD", description="Trading currency")
    sec_cik: str | None = Field(default=None, description="10-digit SEC CIK")
    issuer_name: str = Field(default="", description="Issuer name reported to SEC")
    former_names: list[str] = Field(default_factory=list, description="Former issuer names")
    external_identifiers: dict[str, object] = Field(default_factory=dict)
    identifier_lineage: list[SecurityIdentifierLineageItem] = Field(default_factory=list)

    @field_validator("sec_cik")
    @classmethod
    def validate_sec_cik(cls, value: str | None) -> str | None:
        return _normalize_optional_sec_cik(value)


class UpdateSecurityRequest(BaseModel):
    """Request to update a security."""

    name: str | None = Field(default=None, description="Company name")
    aliases: list[str] | None = Field(default=None, description="Alternative names")
    sector: str | None = Field(default=None, description="Sector classification")
    country: str | None = Field(default=None, description="Country code")
    currency: str | None = Field(default=None, description="Trading currency")
    sec_cik: str | None = Field(default=None, description="10-digit SEC CIK")
    issuer_name: str | None = Field(default=None, description="Issuer name reported to SEC")
    former_names: list[str] | None = Field(default=None, description="Former issuer names")
    external_identifiers: dict[str, object] | None = Field(default=None)
    identifier_lineage: list[SecurityIdentifierLineageItem] | None = Field(default=None)

    @field_validator("sec_cik")
    @classmethod
    def validate_sec_cik(cls, value: str | None) -> str | None:
        return _normalize_optional_sec_cik(value)


class SourceItem(BaseModel):
    """Single source record."""

    platform: str = Field(..., description="Platform: twitter, reddit, substack, rss")
    identifier: str = Field(
        ..., description="Handle, subreddit name, publication slug, or feed slug"
    )
    display_name: str = Field(default="", description="Human-readable display name")
    description: str = Field(default="", description="Short description")
    is_active: bool = Field(default=True, description="Whether source is active")
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Platform-specific metadata",
    )
    created_at: str | None = Field(default=None, description="Creation timestamp (ISO)")
    updated_at: str | None = Field(default=None, description="Last update timestamp (ISO)")


class SourcesListResponse(BaseModel):
    """Paginated sources list."""

    sources: list[SourceItem] = Field(..., description="Source records")
    total: int = Field(..., description="Total matching sources")
    has_more: bool = Field(..., description="Whether more pages exist")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class CreateSourceRequest(BaseModel):
    """Request to create a new source."""

    platform: str = Field(..., description="Platform: twitter, reddit, substack, rss")
    identifier: str = Field(
        ..., min_length=1, max_length=200, description="Handle, subreddit name, slug, or feed slug"
    )
    display_name: str = Field(default="", max_length=200, description="Human-readable name")
    description: str = Field(default="", max_length=500, description="Short description")
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Platform-specific metadata",
    )

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, value: str) -> str:
        return _validate_source_platform(value)

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("identifier must contain at least one non-empty character")
        return cleaned

    @model_validator(mode="after")
    def validate_rss_metadata(self) -> Self:
        if self.platform == "rss":
            _validate_rss_source_metadata(self.metadata)
        return self


class BulkCreateSourcesRequest(BaseModel):
    """Request to bulk-create sources for a single platform."""

    platform: str = Field(
        ...,
        description=(
            "Platform: twitter, reddit, substack. RSS requires metadata; use create source."
        ),
    )
    identifiers: list[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of identifiers to add (max 500)",
    )

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, value: str) -> str:
        platform = _validate_source_platform(value)
        if platform == "rss":
            raise ValueError("RSS sources require metadata; use create source instead")
        return platform

    @field_validator("identifiers")
    @classmethod
    def clean_identifiers(cls, value: list[str]) -> list[str]:
        cleaned = [identifier.strip() for identifier in value if identifier.strip()]
        if not cleaned:
            raise ValueError("identifiers must contain at least one non-empty value")
        for index, identifier in enumerate(cleaned):
            if len(identifier) > 200:
                raise ValueError(f"identifiers[{index}] exceeds maximum length of 200 characters")
        return cleaned


class BulkCreateSourcesResponse(BaseModel):
    """Response for bulk source creation."""

    created: int = Field(..., description="Number of new sources created")
    skipped: int = Field(..., description="Number of identifiers that already existed")
    total: int = Field(..., description="Total identifiers submitted")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class TriggerIngestionResponse(BaseModel):
    """Response model for triggering manual ingestion."""

    status: Literal["started"] = Field(..., description="Trigger status: started")
    message: str = Field(..., description="Human-readable status message")


class UpdateSourceRequest(BaseModel):
    """Request to update a source."""

    display_name: str | None = Field(
        default=None,
        max_length=200,
        description="Human-readable name",
    )
    description: str | None = Field(default=None, max_length=500, description="Short description")
    is_active: bool | None = Field(default=None, description="Whether source is active")
    metadata: dict[str, object] | None = Field(
        default=None,
        description="Platform-specific metadata",
    )
