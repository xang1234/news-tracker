"""
Request and response models for the embedding API.
"""

import datetime as dt
from enum import Enum

from pydantic import BaseModel, Field


class APIModelType(str, Enum):
    """Model selection for API requests."""

    AUTO = "auto"
    FINBERT = "finbert"
    MINILM = "minilm"


class EmbedRequest(BaseModel):
    """Request model for batch embedding."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="List of texts to embed (1-64)",
    )
    model: APIModelType = Field(
        default=APIModelType.AUTO,
        description="Model to use: auto (select by content), finbert, or minilm",
    )
    cache: bool = Field(
        default=True,
        description="Whether to use caching",
    )


class EmbedResponse(BaseModel):
    """Response model for batch embedding."""

    embeddings: list[list[float]] = Field(
        ...,
        description="List of embedding vectors",
    )
    model_used: str = Field(
        ...,
        description="Model that was used for embedding",
    )
    dimensions: int = Field(
        ...,
        description="Embedding dimensions (768 for finbert, 384 for minilm)",
    )
    latency_ms: float = Field(
        ...,
        description="Processing latency in milliseconds",
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(
        ...,
        description="Overall service status: healthy, degraded, or unhealthy",
    )
    models_loaded: dict[str, bool] = Field(
        default_factory=dict,
        description="Which models are currently loaded",
    )
    cache_available: bool = Field(
        default=False,
        description="Whether Redis cache is available",
    )
    gpu_available: bool = Field(
        default=False,
        description="Whether GPU (CUDA) is available for inference",
    )
    service_stats: dict = Field(
        default_factory=dict,
        description="Embedding service statistics",
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(
        ...,
        description="Error message",
    )
    error_type: str = Field(
        default="error",
        description="Error type",
    )


# Search models


class SearchRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text query to find similar documents",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0)",
    )

    # Filters (all optional)
    platforms: list[str] | None = Field(
        default=None,
        description="Filter to documents from these platforms (twitter, reddit, news, substack)",
    )
    tickers: list[str] | None = Field(
        default=None,
        description="Filter to documents mentioning these ticker symbols",
    )
    theme_ids: list[str] | None = Field(
        default=None,
        description="Filter to documents with these theme cluster IDs",
    )
    min_authority_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum authority score threshold",
    )


class SearchResultItem(BaseModel):
    """Single search result item."""

    document_id: str = Field(
        ...,
        description="Unique document identifier",
    )
    score: float = Field(
        ...,
        description="Similarity score (0.0-1.0, higher is more similar)",
    )
    platform: str | None = Field(
        default=None,
        description="Source platform",
    )
    title: str | None = Field(
        default=None,
        description="Document title if available",
    )
    content_preview: str | None = Field(
        default=None,
        description="First 200 characters of content",
    )
    url: str | None = Field(
        default=None,
        description="Original document URL",
    )
    author_name: str | None = Field(
        default=None,
        description="Author display name",
    )
    author_verified: bool = Field(
        default=False,
        description="Whether author is verified",
    )
    tickers: list[str] = Field(
        default_factory=list,
        description="Ticker symbols mentioned",
    )
    authority_score: float | None = Field(
        default=None,
        description="Document authority score (0.0-1.0)",
    )
    timestamp: str | None = Field(
        default=None,
        description="Document creation timestamp (ISO format)",
    )


class SearchResponse(BaseModel):
    """Response model for semantic search."""

    results: list[SearchResultItem] = Field(
        ...,
        description="Search results sorted by similarity",
    )
    total: int = Field(
        ...,
        description="Number of results returned",
    )
    latency_ms: float = Field(
        ...,
        description="Search latency in milliseconds",
    )


# Sentiment models


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of texts to analyze (1-32)",
    )


class SentimentScores(BaseModel):
    """Sentiment class probabilities."""

    positive: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    negative: float = Field(..., ge=0.0, le=1.0)


class EntitySentimentItem(BaseModel):
    """Entity-level sentiment result."""

    entity: str = Field(..., description="Entity text or normalized form")
    type: str = Field(..., description="Entity type (COMPANY, PRODUCT, etc.)")
    label: str = Field(..., description="Sentiment label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: SentimentScores
    context: str | None = Field(default=None, description="Context window used")


class SentimentResultItem(BaseModel):
    """Single sentiment analysis result."""

    label: str = Field(
        ...,
        description="Sentiment label: positive, neutral, or negative",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence",
    )
    scores: SentimentScores = Field(
        ...,
        description="All class probabilities",
    )
    entity_sentiments: list[EntitySentimentItem] = Field(
        default_factory=list,
        description="Reserved for future entity-level sentiment (currently always empty)",
    )


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""

    results: list[SentimentResultItem] = Field(
        ...,
        description="Sentiment results for each input text",
    )
    total: int = Field(
        ...,
        description="Number of results",
    )
    model: str = Field(
        ...,
        description="Model used for analysis",
    )
    latency_ms: float = Field(
        ...,
        description="Processing latency in milliseconds",
    )


# Theme models


class ThemeItem(BaseModel):
    """Shared theme representation for list and detail responses."""

    theme_id: str = Field(..., description="Deterministic theme identifier")
    name: str = Field(..., description="Human-readable theme name")
    top_keywords: list[str] = Field(default_factory=list, description="Ranked topic keywords")
    top_tickers: list[str] = Field(default_factory=list, description="Most-mentioned ticker symbols")
    top_entities: list[dict] = Field(default_factory=list, description="Entity objects with scores")
    lifecycle_stage: str = Field(..., description="One of: emerging, accelerating, mature, fading")
    document_count: int = Field(..., description="Number of documents assigned to this theme")
    description: str | None = Field(default=None, description="Optional human-readable summary")
    created_at: str = Field(..., description="Theme creation timestamp (ISO format)")
    updated_at: str = Field(..., description="Last modification timestamp (ISO format)")
    metadata: dict = Field(default_factory=dict, description="Flexible metadata storage")
    centroid: list[float] | None = Field(
        default=None,
        description="768-dim FinBERT centroid vector (omitted by default, opt-in via include_centroid=true)",
    )


class ThemeListResponse(BaseModel):
    """Response model for listing themes."""

    themes: list[ThemeItem] = Field(..., description="List of themes")
    total: int = Field(..., description="Number of themes returned")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class ThemeDetailResponse(BaseModel):
    """Response model for a single theme."""

    theme: ThemeItem = Field(..., description="Theme details")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class ThemeDocumentItem(BaseModel):
    """Document item within a theme's document list."""

    document_id: str = Field(..., description="Unique document identifier")
    platform: str | None = Field(default=None, description="Source platform")
    title: str | None = Field(default=None, description="Document title if available")
    content_preview: str | None = Field(default=None, description="First 300 characters of content")
    url: str | None = Field(default=None, description="Original document URL")
    author_name: str | None = Field(default=None, description="Author display name")
    tickers: list[str] = Field(default_factory=list, description="Ticker symbols mentioned")
    authority_score: float | None = Field(default=None, description="Document authority score (0.0-1.0)")
    sentiment_label: str | None = Field(default=None, description="Sentiment label if available")
    sentiment_confidence: float | None = Field(default=None, description="Sentiment confidence if available")
    timestamp: str | None = Field(default=None, description="Document creation timestamp (ISO format)")


class ThemeDocumentsResponse(BaseModel):
    """Response model for documents in a theme."""

    documents: list[ThemeDocumentItem] = Field(..., description="Documents in this theme")
    total: int = Field(..., description="Number of documents returned")
    theme_id: str = Field(..., description="Theme identifier")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class ThemeSentimentResponse(BaseModel):
    """Response model for theme sentiment aggregation."""

    theme_id: str = Field(..., description="Theme identifier")
    bullish_ratio: float = Field(..., description="Proportion of positive sentiment (0-1)")
    bearish_ratio: float = Field(..., description="Proportion of negative sentiment (0-1)")
    neutral_ratio: float = Field(..., description="Proportion of neutral sentiment (0-1)")
    avg_confidence: float = Field(..., description="Average sentiment confidence")
    avg_authority: float | None = Field(default=None, description="Average authority score of documents")
    sentiment_velocity: float | None = Field(
        default=None, description="Rate of sentiment change (positive = more bullish)"
    )
    extreme_sentiment: str | None = Field(
        default=None, description="Extreme sentiment flag: extreme_bullish or extreme_bearish"
    )
    document_count: int = Field(..., description="Number of documents in aggregation window")
    window_start: str = Field(..., description="Aggregation window start (ISO format)")
    window_end: str = Field(..., description="Aggregation window end (ISO format)")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class ThemeMetricsItem(BaseModel):
    """Single day of theme metrics."""

    date: dt.date = Field(..., description="Calendar date for this metrics snapshot")
    document_count: int = Field(..., description="Number of documents on this date")
    sentiment_score: float | None = Field(default=None, description="Aggregate sentiment (-1.0 to 1.0)")
    volume_zscore: float | None = Field(default=None, description="Standard deviations from mean volume")
    velocity: float | None = Field(default=None, description="Rate of volume change")
    acceleration: float | None = Field(default=None, description="Rate of velocity change")
    avg_authority: float | None = Field(default=None, description="Mean authority score of documents")
    bullish_ratio: float | None = Field(default=None, description="Fraction of positive sentiment documents")


class ThemeMetricsResponse(BaseModel):
    """Response model for theme metrics time series."""

    metrics: list[ThemeMetricsItem] = Field(..., description="Daily metrics time series")
    total: int = Field(..., description="Number of metric data points")
    theme_id: str = Field(..., description="Theme identifier")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


# Event models


class ThemeEventItem(BaseModel):
    """Single event linked to a theme."""

    event_id: str = Field(..., description="Unique event identifier")
    doc_id: str = Field(..., description="Source document identifier")
    event_type: str = Field(..., description="Event category (e.g., capacity_expansion)")
    actor: str | None = Field(default=None, description="Entity performing the action")
    action: str = Field(..., description="Action phrase from the text")
    object: str | None = Field(default=None, description="Target of the action")
    time_ref: str | None = Field(default=None, description="Temporal reference (e.g., Q3 2026)")
    quantity: str | None = Field(default=None, description="Numeric quantity mentioned")
    tickers: list[str] = Field(default_factory=list, description="Ticker symbols linked to this event")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score")
    source_doc_ids: list[str] = Field(
        default_factory=list, description="Document IDs confirming this event (after dedup)"
    )
    created_at: str | None = Field(default=None, description="Event extraction timestamp (ISO format)")


class ThemeEventsResponse(BaseModel):
    """Response model for events linked to a theme."""

    events: list[ThemeEventItem] = Field(..., description="Events linked to this theme")
    total: int = Field(..., description="Number of events returned")
    theme_id: str = Field(..., description="Theme identifier")
    event_counts: dict[str, int] = Field(
        default_factory=dict, description="Event counts by type"
    )
    investment_signal: str | None = Field(
        default=None,
        description="Derived investment signal: supply_increasing, supply_decreasing, product_momentum, product_risk, or null",
    )
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


# Ranking models


class RankedThemeItem(BaseModel):
    """Single ranked theme with score and tier."""

    theme: ThemeItem = Field(..., description="Theme details")
    score: float = Field(..., description="Composite ranking score (higher = more actionable)")
    tier: int = Field(..., ge=1, le=3, description="Tier: 1 (top 5%), 2 (top 20%), 3 (rest)")
    components: dict = Field(
        default_factory=dict,
        description="Score component breakdown (volume_component, compellingness_component, lifecycle_multiplier, volume_zscore, strategy)",
    )


class RankedThemesResponse(BaseModel):
    """Response model for ranked themes."""

    themes: list[RankedThemeItem] = Field(..., description="Ranked themes sorted by score")
    total: int = Field(..., description="Number of ranked themes returned")
    strategy: str = Field(..., description="Ranking strategy used (swing or position)")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


# Alert models


class AlertItem(BaseModel):
    """Single alert record."""

    alert_id: str = Field(..., description="Unique alert identifier")
    theme_id: str = Field(..., description="Theme that triggered the alert")
    trigger_type: str = Field(..., description="Alert trigger type")
    severity: str = Field(..., description="Severity level: critical, warning, info")
    title: str = Field(..., description="Short human-readable summary")
    message: str = Field(..., description="Detailed alert description")
    trigger_data: dict = Field(default_factory=dict, description="Trigger-specific context")
    acknowledged: bool = Field(default=False, description="Whether the alert has been reviewed")
    created_at: str = Field(..., description="Alert creation timestamp (ISO format)")


class AlertsResponse(BaseModel):
    """Response model for listing alerts."""

    alerts: list[AlertItem] = Field(..., description="List of alerts")
    total: int = Field(..., description="Number of alerts returned")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class AlertAcknowledgeResponse(BaseModel):
    """Response model for acknowledging an alert."""

    alert_id: str = Field(..., description="Acknowledged alert identifier")
    acknowledged: bool = Field(..., description="New acknowledgement status")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


# Graph propagation models


class PropagateRequest(BaseModel):
    """Request model for sentiment propagation."""

    source_node: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Node ID where sentiment changed (e.g., 'theme_hbm_demand')",
    )
    sentiment_delta: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Magnitude of sentiment change (-1.0 to 1.0)",
    )


class PropagationImpactItem(BaseModel):
    """Single propagation impact result."""

    node_id: str = Field(..., description="Affected downstream node ID")
    impact: float = Field(..., description="Propagated sentiment impact (signed)")
    depth: int = Field(..., ge=1, description="Hops from source node")
    relation: str = Field(..., description="Edge type of the first hop reaching this node")
    edge_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of that edge")


class PropagateResponse(BaseModel):
    """Response model for sentiment propagation."""

    source_node: str = Field(..., description="Source node where sentiment changed")
    sentiment_delta: float = Field(..., description="Input sentiment delta")
    impacts: list[PropagationImpactItem] = Field(
        ..., description="Affected nodes sorted by absolute impact (descending)"
    )
    total_affected: int = Field(..., description="Number of affected downstream nodes")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


# Feedback models


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""

    entity_type: str = Field(
        ...,
        description="Type of entity being rated: theme, alert, document",
    )
    entity_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Identifier of the entity being rated",
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Quality rating from 1 (poor) to 5 (excellent)",
    )
    quality_label: str | None = Field(
        default=None,
        description="Optional categorical label: useful, noise, too_late, wrong_direction",
    )
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional free-text comment",
    )


class FeedbackItem(BaseModel):
    """Single feedback record."""

    feedback_id: str = Field(..., description="Unique feedback identifier")
    entity_type: str = Field(..., description="Entity type: theme, alert, document")
    entity_id: str = Field(..., description="Entity identifier")
    rating: int = Field(..., ge=1, le=5, description="Quality rating (1-5)")
    quality_label: str | None = Field(default=None, description="Categorical quality label")
    comment: str | None = Field(default=None, description="Free-text comment")
    user_id: str | None = Field(default=None, description="User who submitted the feedback")
    created_at: str = Field(..., description="Submission timestamp (ISO format)")


class FeedbackResponse(BaseModel):
    """Response model for creating feedback."""

    feedback: FeedbackItem = Field(..., description="Created feedback record")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class FeedbackStatsItem(BaseModel):
    """Aggregated feedback statistics for an entity type."""

    entity_type: str = Field(..., description="Entity type: theme, alert, document")
    total_count: int = Field(..., description="Total number of feedback records")
    avg_rating: float = Field(..., description="Average rating (1.0-5.0)")
    label_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each quality label",
    )


class FeedbackStatsResponse(BaseModel):
    """Response model for feedback statistics."""

    stats: list[FeedbackStatsItem] = Field(..., description="Statistics grouped by entity type")
    total: int = Field(..., description="Number of entity type groups")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
