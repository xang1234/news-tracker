"""
Temporal sentiment aggregation for alpha generation.

Provides rolling averages, recency weighting, sentiment velocity detection,
and extreme sentiment (crowded trade) identification.

Usage:
    from src.sentiment.aggregation import SentimentAggregator, DocumentSentiment

    aggregator = SentimentAggregator(decay_rate=0.3)
    docs = [
        DocumentSentiment("d1", now, "positive", 0.9, scores, authority=0.8),
        DocumentSentiment("d2", now - timedelta(days=1), "negative", 0.7, scores),
    ]
    result = aggregator.aggregate_theme_sentiment("AI_chips", None, docs)
    print(f"Bullish: {result.bullish_ratio:.1%}, Velocity: {result.sentiment_velocity}")
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.sentiment.config import SentimentConfig


@dataclass
class DocumentSentiment:
    """
    Sentiment data for a single document used in aggregation.

    Lightweight container holding only the fields needed for temporal aggregation.
    """

    document_id: str
    timestamp: datetime
    label: str  # "positive", "negative", "neutral"
    confidence: float
    scores: Dict[str, float]  # {"positive": p, "negative": n, "neutral": u}
    authority_score: Optional[float] = None
    platform: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.label not in ("positive", "negative", "neutral"):
            raise ValueError(f"Invalid label: {self.label}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass
class AggregatedSentiment:
    """
    Aggregated sentiment metrics for a theme or ticker over a time window.

    Contains weighted sentiment ratios, velocity metrics, and extreme detection flags.
    """

    theme_id: Optional[str]
    ticker: Optional[str]
    window_start: datetime
    window_end: datetime
    document_count: int

    # Weighted sentiment ratios (sum to ~1.0)
    bullish_ratio: float  # Proportion of positive sentiment
    bearish_ratio: float  # Proportion of negative sentiment
    neutral_ratio: float  # Proportion of neutral sentiment

    # Quality metrics
    avg_confidence: float
    avg_authority: Optional[float]

    # Alpha signals
    sentiment_velocity: Optional[float] = None  # Rate of change (slope)
    extreme_sentiment: Optional[str] = None  # "extreme_bullish" or "extreme_bearish"

    # Metadata
    daily_bullish_ratios: List[Tuple[datetime, float]] = field(default_factory=list)


class SentimentAggregator:
    """
    Aggregates document sentiments over time with recency weighting.

    Uses exponential decay to weight recent documents more heavily,
    incorporates authority and confidence scores, and detects sentiment
    velocity (rate of change) and extreme crowded positions.

    Weighting formula:
        weight = recency_decay * authority_factor * confidence_factor

    Where:
        recency_decay = exp(-decay_rate * days_ago)
        authority_factor = 1 + authority_weight * (authority_score or 0)
        confidence_factor = 1 + confidence_weight * confidence

    At decay_rate=0.3, the half-life is ~2.3 days:
        exp(-0.3 * 2.3) ≈ 0.5
    """

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        decay_rate: Optional[float] = None,
        authority_weight: Optional[float] = None,
        confidence_weight: Optional[float] = None,
    ):
        """
        Initialize the aggregator.

        Args:
            config: SentimentConfig with aggregation settings (optional)
            decay_rate: Override config's decay rate (optional)
            authority_weight: Override config's authority weight (optional)
            confidence_weight: Override config's confidence weight (optional)
        """
        self._config = config or SentimentConfig()

        # Allow parameter overrides for testing/customization
        self._decay_rate = (
            decay_rate if decay_rate is not None
            else self._config.aggregation_decay_rate
        )
        self._authority_weight = (
            authority_weight if authority_weight is not None
            else self._config.aggregation_authority_weight
        )
        self._confidence_weight = (
            confidence_weight if confidence_weight is not None
            else self._config.aggregation_confidence_weight
        )

    def _compute_weight(
        self,
        doc: DocumentSentiment,
        reference_time: datetime,
    ) -> float:
        """
        Compute weight for a single document.

        Args:
            doc: Document sentiment to weight
            reference_time: Reference timestamp (typically window_end or now)

        Returns:
            Composite weight > 0
        """
        # Recency decay
        delta = reference_time - doc.timestamp
        days_ago = max(0, delta.total_seconds() / 86400)
        recency_decay = math.exp(-self._decay_rate * days_ago)

        # Authority factor (1.0 to 1.0 + authority_weight)
        authority = doc.authority_score if doc.authority_score is not None else 0.0
        authority_factor = 1.0 + self._authority_weight * authority

        # Confidence factor (1.0 to 1.0 + confidence_weight)
        confidence_factor = 1.0 + self._confidence_weight * doc.confidence

        return recency_decay * authority_factor * confidence_factor

    def aggregate_theme_sentiment(
        self,
        theme_id: Optional[str],
        ticker: Optional[str],
        document_sentiments: List[DocumentSentiment],
        window_days: int = 7,
        reference_time: Optional[datetime] = None,
    ) -> AggregatedSentiment:
        """
        Aggregate sentiment for a theme or ticker over a time window.

        Args:
            theme_id: Theme identifier (optional)
            ticker: Ticker symbol (optional)
            document_sentiments: List of document sentiments to aggregate
            window_days: Number of days to include in aggregation
            reference_time: End of window (defaults to now UTC)

        Returns:
            AggregatedSentiment with weighted ratios and alpha signals
        """
        ref_time = reference_time or datetime.now(timezone.utc)
        window_start = ref_time - timedelta(days=window_days)

        # Filter to window
        in_window = [
            doc for doc in document_sentiments
            if doc.timestamp >= window_start and doc.timestamp <= ref_time
        ]

        # Handle empty case
        if not in_window:
            return AggregatedSentiment(
                theme_id=theme_id,
                ticker=ticker,
                window_start=window_start,
                window_end=ref_time,
                document_count=0,
                bullish_ratio=0.0,
                bearish_ratio=0.0,
                neutral_ratio=0.0,
                avg_confidence=0.0,
                avg_authority=None,
                sentiment_velocity=None,
                extreme_sentiment=None,
            )

        # Compute weighted sums
        total_weight = 0.0
        weighted_positive = 0.0
        weighted_negative = 0.0
        weighted_neutral = 0.0
        weighted_confidence = 0.0
        authority_sum = 0.0
        authority_count = 0

        for doc in in_window:
            weight = self._compute_weight(doc, ref_time)
            total_weight += weight

            # Weight by label
            if doc.label == "positive":
                weighted_positive += weight
            elif doc.label == "negative":
                weighted_negative += weight
            else:
                weighted_neutral += weight

            weighted_confidence += doc.confidence * weight

            if doc.authority_score is not None:
                authority_sum += doc.authority_score
                authority_count += 1

        # Normalize ratios
        if total_weight > 0:
            bullish_ratio = weighted_positive / total_weight
            bearish_ratio = weighted_negative / total_weight
            neutral_ratio = weighted_neutral / total_weight
            avg_confidence = weighted_confidence / total_weight
        else:
            bullish_ratio = bearish_ratio = neutral_ratio = 0.0
            avg_confidence = 0.0

        avg_authority = authority_sum / authority_count if authority_count > 0 else None

        # Compute daily bullish ratios for velocity calculation
        daily_ratios = self._compute_daily_ratios(in_window, window_start, ref_time)

        # Compute sentiment velocity (rate of change)
        sentiment_velocity = self.compute_sentiment_velocity(daily_ratios)

        # Detect extreme sentiment
        extreme_sentiment = self.detect_extreme_sentiment(bullish_ratio)

        return AggregatedSentiment(
            theme_id=theme_id,
            ticker=ticker,
            window_start=window_start,
            window_end=ref_time,
            document_count=len(in_window),
            bullish_ratio=round(bullish_ratio, 4),
            bearish_ratio=round(bearish_ratio, 4),
            neutral_ratio=round(neutral_ratio, 4),
            avg_confidence=round(avg_confidence, 4),
            avg_authority=round(avg_authority, 4) if avg_authority is not None else None,
            sentiment_velocity=round(sentiment_velocity, 4) if sentiment_velocity is not None else None,
            extreme_sentiment=extreme_sentiment,
            daily_bullish_ratios=daily_ratios,
        )

    def _compute_daily_ratios(
        self,
        docs: List[DocumentSentiment],
        window_start: datetime,
        window_end: datetime,
    ) -> List[Tuple[datetime, float]]:
        """
        Compute daily bullish ratios for velocity calculation.

        Groups documents by date and computes bullish ratio per day.
        """
        # Group by date
        by_date: Dict[datetime.date, List[DocumentSentiment]] = {}
        for doc in docs:
            date = doc.timestamp.date()
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(doc)

        # Compute ratio for each day
        daily_ratios = []
        for date in sorted(by_date.keys()):
            day_docs = by_date[date]
            positive_count = sum(1 for d in day_docs if d.label == "positive")
            total = len(day_docs)
            ratio = positive_count / total if total > 0 else 0.5
            # Use noon of the date as timestamp
            dt = datetime.combine(date, datetime.min.time().replace(hour=12), tzinfo=timezone.utc)
            daily_ratios.append((dt, ratio))

        return daily_ratios

    def compute_sentiment_velocity(
        self,
        daily_ratios: List[Tuple[datetime, float]],
    ) -> Optional[float]:
        """
        Compute sentiment velocity using linear regression.

        Measures the rate of change in bullish ratio over time.
        Positive velocity = sentiment becoming more bullish.
        Negative velocity = sentiment becoming more bearish.

        Args:
            daily_ratios: List of (datetime, bullish_ratio) tuples

        Returns:
            Slope of linear fit (change in bullish_ratio per day),
            or None if insufficient data
        """
        if len(daily_ratios) < 2:
            return None

        # Convert to arrays
        # Use day index relative to first date
        base_date = daily_ratios[0][0]
        x = np.array([(dt - base_date).days for dt, _ in daily_ratios], dtype=float)
        y = np.array([ratio for _, ratio in daily_ratios], dtype=float)

        # Check for variance in x (need at least 2 distinct days)
        if np.std(x) < 0.1:
            return None

        # Linear regression: y = slope * x + intercept
        try:
            slope, _ = np.polyfit(x, y, 1)
            return float(slope)
        except (np.linalg.LinAlgError, ValueError):
            return None

    def detect_extreme_sentiment(
        self,
        bullish_ratio: float,
    ) -> Optional[str]:
        """
        Detect extreme sentiment conditions (crowded trades).

        Extreme bullish indicates potential overbought condition.
        Extreme bearish indicates potential oversold condition.

        Args:
            bullish_ratio: Proportion of positive sentiment (0-1)

        Returns:
            "extreme_bullish", "extreme_bearish", or None
        """
        if bullish_ratio >= self._config.extreme_bullish_threshold:
            return "extreme_bullish"
        elif bullish_ratio <= self._config.extreme_bearish_threshold:
            return "extreme_bearish"
        return None

    def get_config_summary(self) -> Dict[str, float]:
        """Get current aggregation configuration."""
        return {
            "decay_rate": self._decay_rate,
            "authority_weight": self._authority_weight,
            "confidence_weight": self._confidence_weight,
            "extreme_bullish_threshold": self._config.extreme_bullish_threshold,
            "extreme_bearish_threshold": self._config.extreme_bearish_threshold,
        }
