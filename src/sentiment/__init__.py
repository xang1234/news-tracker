"""
Sentiment analysis service using FinBERT.

Provides financial sentiment classification for documents with:
- Document-level sentiment (positive/neutral/negative)
- Entity-level sentiment (optional, nested in results)
- Emoji-based sentiment modifiers (bullish/bearish signals)
- Temporal aggregation with recency weighting
- Sentiment velocity (rate of change) detection
- Extreme sentiment (crowded trade) identification
- Lazy model loading for efficient resource usage
- Redis caching to avoid recomputation
- Async worker for pipeline integration

Usage:
    # Direct analysis
    from src.sentiment import SentimentService

    service = SentimentService()
    result = await service.analyze("NVIDIA stock surged on AI demand 🚀")
    print(f"{result['label']}: {result['confidence']:.2f}")
    if "emoji_modifier" in result:
        print(f"Emoji modifier: {result['emoji_modifier']}")

    # Temporal aggregation
    from src.sentiment import SentimentAggregator, DocumentSentiment
    from datetime import datetime, timezone

    aggregator = SentimentAggregator()
    docs = [
        DocumentSentiment("d1", datetime.now(timezone.utc), "positive", 0.9, {...}),
    ]
    agg = aggregator.aggregate_theme_sentiment("AI_chips", None, docs)
    print(f"Bullish: {agg.bullish_ratio:.1%}, Velocity: {agg.sentiment_velocity}")
"""

from src.sentiment.aggregation import (
    AggregatedSentiment,
    DocumentSentiment,
    SentimentAggregator,
)
from src.sentiment.config import SentimentConfig
from src.sentiment.emoji_lookup import (
    EMOJI_SENTIMENT,
    compute_emoji_modifier,
    extract_emojis,
    get_emoji_breakdown,
)
from src.sentiment.service import SentimentService

__all__ = [
    # Core service
    "SentimentConfig",
    "SentimentService",
    # Aggregation
    "AggregatedSentiment",
    "DocumentSentiment",
    "SentimentAggregator",
    # Emoji utilities
    "EMOJI_SENTIMENT",
    "compute_emoji_modifier",
    "extract_emojis",
    "get_emoji_breakdown",
]
