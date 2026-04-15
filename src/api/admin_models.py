"""Admin API models for sources and securities."""

from pydantic import BaseModel, Field, field_validator


class SecurityItem(BaseModel):
    """Single security record."""

    ticker: str = Field(..., description="Ticker symbol")
    exchange: str = Field(..., description="Exchange code (e.g. US, KRX)")
    name: str = Field(..., description="Company name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    sector: str = Field(default="", description="Sector classification")
    country: str = Field(default="US", description="Country code")
    currency: str = Field(default="USD", description="Trading currency")
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


class UpdateSecurityRequest(BaseModel):
    """Request to update a security."""

    name: str | None = Field(default=None, description="Company name")
    aliases: list[str] | None = Field(default=None, description="Alternative names")
    sector: str | None = Field(default=None, description="Sector classification")
    country: str | None = Field(default=None, description="Country code")
    currency: str | None = Field(default=None, description="Trading currency")


class SourceItem(BaseModel):
    """Single source record."""

    platform: str = Field(..., description="Platform: twitter, reddit, substack")
    identifier: str = Field(..., description="Handle, subreddit name, or publication slug")
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

    platform: str = Field(..., description="Platform: twitter, reddit, substack")
    identifier: str = Field(
        ..., min_length=1, max_length=200, description="Handle, subreddit name, or slug"
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
        allowed = {"twitter", "reddit", "substack"}
        if value.lower() not in allowed:
            raise ValueError(f"platform must be one of: {', '.join(sorted(allowed))}")
        return value.lower()


class BulkCreateSourcesRequest(BaseModel):
    """Request to bulk-create sources for a single platform."""

    platform: str = Field(..., description="Platform: twitter, reddit, substack")
    identifiers: list[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of identifiers to add (max 500)",
    )

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, value: str) -> str:
        allowed = {"twitter", "reddit", "substack"}
        if value.lower() not in allowed:
            raise ValueError(f"platform must be one of: {', '.join(sorted(allowed))}")
        return value.lower()

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

    status: str = Field(..., description="Trigger status: started or already_running")
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
