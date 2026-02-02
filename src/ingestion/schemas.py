"""
Canonical document schema for the news-tracker pipeline.

CRITICAL: This schema flows through the entire pipeline - do not modify field names
without updating all downstream services. All platform adapters MUST output this
exact structure.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _utc_now() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


class Platform(str, Enum):
    """Supported data source platforms."""

    TWITTER = "twitter"
    REDDIT = "reddit"
    SUBSTACK = "substack"
    NEWS = "news"


class EngagementMetrics(BaseModel):
    """
    Platform-normalized engagement signals.

    Different platforms have different engagement metrics, but we normalize
    to a common structure. Platform-specific fields are optional.
    """

    likes: int = Field(default=0, ge=0, description="Likes, favorites, or upvotes")
    shares: int = Field(default=0, ge=0, description="Retweets, crossposts, shares")
    comments: int = Field(default=0, ge=0, description="Comment or reply count")
    views: int | None = Field(default=None, ge=0, description="View count if available")

    # Platform-specific metrics
    upvote_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reddit-specific: ratio of upvotes to total votes",
    )
    read_time_minutes: float | None = Field(
        default=None,
        ge=0.0,
        description="Substack-specific: estimated read time",
    )

    @property
    def engagement_score(self) -> float:
        """
        Calculate a normalized engagement score.

        This is a simple heuristic that can be tuned based on backtesting.
        """
        score = self.likes + (self.shares * 2) + self.comments
        if self.upvote_ratio is not None:
            score *= self.upvote_ratio
        return float(score)


class NormalizedDocument(BaseModel):
    """
    CANONICAL DOCUMENT SCHEMA

    All platform adapters MUST output this exact structure.
    Downstream services (embedding, storage, analysis) depend on these field names.
    """

    # Identity
    id: str = Field(
        ...,
        description="Unique ID in format: {platform}_{native_id}",
        examples=["twitter_1234567890", "reddit_abc123"],
    )
    platform: Platform = Field(..., description="Source platform")
    url: str | None = Field(default=None, description="Original content URL")

    # Timestamps
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of content creation on the platform",
    )
    fetched_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when document was fetched",
    )

    # Author information
    author_id: str = Field(..., description="Platform-specific author identifier")
    author_name: str = Field(..., description="Display name of the author")
    author_followers: int | None = Field(
        default=None,
        ge=0,
        description="Follower count at time of fetch",
    )
    author_verified: bool = Field(
        default=False,
        description="Whether author is verified/premium",
    )

    # Content
    content: str = Field(
        ...,
        min_length=1,
        description="Full text content, cleaned and normalized",
    )
    content_type: Literal["post", "comment", "article"] = Field(
        default="post",
        description="Type of content",
    )
    title: str | None = Field(
        default=None,
        description="Title for Reddit posts, Substack articles, news",
    )

    # Engagement
    engagement: EngagementMetrics = Field(
        default_factory=EngagementMetrics,
        description="Platform-normalized engagement metrics",
    )

    # Extracted entities (populated during preprocessing)
    tickers_mentioned: list[str] = Field(
        default_factory=list,
        description="Extracted $TICKER cashtags and company references",
    )
    urls_mentioned: list[str] = Field(
        default_factory=list,
        description="URLs found in content",
    )

    # Quality signals (computed during preprocessing)
    spam_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Spam probability from 0.0 (not spam) to 1.0 (spam)",
    )
    bot_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Bot probability from 0.0 (human) to 1.0 (bot)",
    )

    # Set by downstream services
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding (set by embedding service)",
    )
    sentiment: dict[str, Any] | None = Field(
        default=None,
        description="Sentiment analysis results (set by NLP service)",
    )
    theme_ids: list[str] = Field(
        default_factory=list,
        description="Theme cluster IDs (set by clustering service)",
    )

    # Metadata
    raw_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Original platform response for debugging",
    )

    model_config = {
        "use_enum_values": True,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure ID follows platform_nativeid format."""
        if "_" not in v:
            raise ValueError("ID must be in format: {platform}_{native_id}")
        return v

    @field_validator("content")
    @classmethod
    def normalize_content(cls, v: str) -> str:
        """Basic content normalization."""
        # Remove excessive whitespace
        v = " ".join(v.split())
        return v.strip()

    @field_validator("tickers_mentioned")
    @classmethod
    def normalize_tickers(cls, v: list[str]) -> list[str]:
        """Normalize ticker symbols to uppercase without $ prefix."""
        normalized = []
        for ticker in v:
            t = ticker.upper().strip()
            if t.startswith("$"):
                t = t[1:]
            if t and t not in normalized:
                normalized.append(t)
        return normalized

    @property
    def is_high_engagement(self) -> bool:
        """Check if document has above-average engagement."""
        return self.engagement.engagement_score > 100

    @property
    def is_spam(self) -> bool:
        """Check if document is likely spam (default threshold: 0.7)."""
        return self.spam_score >= 0.7

    @property
    def is_likely_bot(self) -> bool:
        """Check if author is likely a bot (default threshold: 0.7)."""
        return self.bot_probability >= 0.7

    @property
    def should_filter(self) -> bool:
        """Check if document should be filtered out."""
        return self.is_spam or self.is_likely_bot

    def to_storage_dict(self) -> dict[str, Any]:
        """
        Convert to dict for database storage.

        Excludes embedding and raw_data to reduce storage size.
        """
        data = self.model_dump()
        data.pop("embedding", None)
        data.pop("raw_data", None)
        return data
