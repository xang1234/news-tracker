"""
Mock adapter for testing and development.

Generates synthetic financial documents that mimic real platform data.
Useful for:
- Testing the pipeline without API credentials
- Load testing
- Development and debugging
"""

import random
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any

from src.config.tickers import SEMICONDUCTOR_TICKERS
from src.ingestion.base_adapter import BaseAdapter
from src.ingestion.schemas import (
    EngagementMetrics,
    NormalizedDocument,
    Platform,
)

# Sample content templates for realistic mock data
BULLISH_TEMPLATES = [
    "{ticker} looking strong heading into earnings. Expecting a beat based on channel checks.",
    "Just loaded up on {ticker} calls. This semiconductor cycle is just getting started.",
    "DD on {ticker}: Supply constraints easing, demand still robust. PT $XXX.",
    "{ticker} new orders accelerating per my sources. HBM demand insane.",
    "Breaking: {ticker} wins major AI infrastructure contract. Massive revenue implications.",
    "{ticker} CEO on CNBC crushing it. Guidance raise incoming.",
]

BEARISH_TEMPLATES = [
    "{ticker} looking weak here. Inventory build concerns are real.",
    "Cutting {ticker} position. Seeing demand softening in consumer segment.",
    "{ticker} PT cut by Goldman to $XXX. Margin compression narrative gaining steam.",
    "Warning signs in {ticker} channel checks. Booking rates declining.",
    "{ticker} guidance likely to disappoint. Competition heating up.",
]

NEUTRAL_TEMPLATES = [
    "Watching {ticker} closely. Waiting for more clarity on AI demand trajectory.",
    "{ticker} trading in a range. Need catalyst to break out.",
    "Mixed signals on {ticker}. Strong datacenter, weak mobile.",
    "Holding {ticker} but trimmed position size. Valuation stretched.",
    "{ticker} earnings next week. Could go either way.",
]

NEWS_TEMPLATES = [
    "{ticker} Announces Q{q} Earnings: Revenue {revenue}B, EPS {eps}",
    "{ticker} Expands Manufacturing Capacity with New {location} Fab",
    "{ticker} Partners with {partner} on Next-Gen AI Chips",
    "Analyst Roundup: {ticker} Price Targets Range from ${low} to ${high}",
    "{ticker} CEO Discusses AI Strategy at Industry Conference",
]

SAMPLE_AUTHORS = [
    ("fintwit_alpha", 15000, True),
    ("semiconductor_daily", 8500, True),
    ("chip_trader_99", 2100, False),
    ("value_investor_mike", 45000, True),
    ("tech_analyst_jen", 12000, True),
    ("random_retail_123", 150, False),
    ("wsb_degen", 500, False),
    ("institutional_view", 95000, True),
    ("quant_signals", 3200, False),
    ("macro_mike", 28000, True),
]


class MockAdapter(BaseAdapter):
    """
    Mock adapter that generates synthetic financial documents.

    Useful for testing the ingestion pipeline without real API access.
    Generates a mix of bullish, bearish, and neutral sentiment across
    different content types.
    """

    def __init__(
        self,
        platform: Platform = Platform.TWITTER,
        documents_per_fetch: int = 10,
        include_spam: bool = True,
        rate_limit: int = 100,
    ):
        """
        Initialize mock adapter.

        Args:
            platform: Which platform to mimic
            documents_per_fetch: Number of documents to generate per fetch
            include_spam: Whether to include spam-like content (for testing filters)
            rate_limit: Rate limit (not really needed for mock, but for consistency)
        """
        super().__init__(rate_limit=rate_limit)
        self._platform = platform
        self._documents_per_fetch = documents_per_fetch
        self._include_spam = include_spam
        self._doc_counter = 0

    @property
    def platform(self) -> Platform:
        return self._platform

    async def _fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        """Generate mock raw data."""
        tickers = list(SEMICONDUCTOR_TICKERS)

        for _ in range(self._documents_per_fetch):
            self._doc_counter += 1

            # Select random ticker and author
            ticker = random.choice(tickers)
            author_name, followers, verified = random.choice(SAMPLE_AUTHORS)

            # Generate content based on sentiment distribution
            # Roughly: 40% bullish, 30% neutral, 20% bearish, 10% spam
            rand = random.random()
            if rand < 0.4:
                templates = BULLISH_TEMPLATES
            elif rand < 0.7:
                templates = NEUTRAL_TEMPLATES
            elif rand < 0.9:
                templates = BEARISH_TEMPLATES
            else:
                # Generate spam-like content
                if self._include_spam:
                    yield self._generate_spam(ticker)
                    continue
                templates = NEUTRAL_TEMPLATES

            # Select and populate template
            template = random.choice(templates)
            content = template.format(
                ticker=f"${ticker}",
                q=random.randint(1, 4),
                revenue=round(random.uniform(5, 50), 1),
                eps=round(random.uniform(0.5, 5.0), 2),
                location=random.choice(["Arizona", "Texas", "Ohio", "Germany"]),
                partner=random.choice(["Microsoft", "Google", "Meta", "Amazon"]),
                low=random.randint(50, 150),
                high=random.randint(200, 400),
            )

            # Generate timestamp within last 24 hours
            timestamp = datetime.utcnow() - timedelta(
                hours=random.randint(0, 24),
                minutes=random.randint(0, 59),
            )

            yield {
                "id": f"mock_{self._doc_counter}",
                "content": content,
                "author_id": f"user_{hash(author_name) % 10000}",
                "author_name": author_name,
                "author_followers": followers + random.randint(-100, 500),
                "author_verified": verified,
                "timestamp": timestamp.isoformat(),
                "ticker": ticker,
                "engagement": {
                    "likes": random.randint(0, 500) if verified else random.randint(0, 50),
                    "shares": random.randint(0, 100) if verified else random.randint(0, 10),
                    "comments": random.randint(0, 50),
                },
            }

    def _generate_spam(self, ticker: str) -> dict[str, Any]:
        """Generate spam-like content for testing spam filters."""
        self._doc_counter += 1

        spam_templates = [
            f"🚀🚀🚀 ${ticker} TO THE MOON!!! 🚀🚀🚀 JOIN MY DISCORD FOR FREE SIGNALS!!!",
            f"${ticker} $AAPL $TSLA $GME $AMC GUARANTEED 1000% GAINS DM ME NOW!!!",
            f"I made $50,000 on ${ticker} using this ONE SIMPLE TRICK! Link in bio!",
            f"FREE STOCK PICKS! ${ticker} is just the beginning! Join telegram @scamchannel",
        ]

        return {
            "id": f"mock_spam_{self._doc_counter}",
            "content": random.choice(spam_templates),
            "author_id": f"spam_bot_{random.randint(1000, 9999)}",
            "author_name": f"crypto_guru_{random.randint(100, 999)}",
            "author_followers": random.randint(0, 50),
            "author_verified": False,
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "engagement": {
                "likes": random.randint(0, 10),
                "shares": random.randint(0, 5),
                "comments": random.randint(0, 3),
            },
        }

    def _transform(self, raw: dict[str, Any]) -> NormalizedDocument | None:
        """Transform mock raw data to NormalizedDocument."""
        try:
            # Parse timestamp
            if isinstance(raw["timestamp"], str):
                timestamp = datetime.fromisoformat(raw["timestamp"])
            else:
                timestamp = raw["timestamp"]

            # Build engagement metrics
            eng_data = raw.get("engagement", {})
            engagement = EngagementMetrics(
                likes=eng_data.get("likes", 0),
                shares=eng_data.get("shares", 0),
                comments=eng_data.get("comments", 0),
            )

            # Determine content type based on platform
            if self._platform == Platform.SUBSTACK:
                content_type = "article"
            elif self._platform == Platform.REDDIT:
                content_type = "post"
            else:
                content_type = "post"

            return NormalizedDocument(
                id=f"{self._platform.value}_{raw['id']}",
                platform=self._platform,
                timestamp=timestamp,
                author_id=raw["author_id"],
                author_name=raw["author_name"],
                author_followers=raw.get("author_followers"),
                author_verified=raw.get("author_verified", False),
                content=raw["content"],
                content_type=content_type,
                engagement=engagement,
                tickers_mentioned=[raw["ticker"]] if raw.get("ticker") else [],
                raw_data=raw,
            )

        except Exception:
            return None

    async def health_check(self) -> bool:
        """Mock adapter is always healthy."""
        return True


def create_mock_adapters(
    documents_per_fetch: int = 10,
) -> dict[Platform, MockAdapter]:
    """
    Create mock adapters for all platforms.

    Args:
        documents_per_fetch: Number of documents each adapter generates

    Returns:
        Dictionary mapping Platform to MockAdapter
    """
    return {
        Platform.TWITTER: MockAdapter(
            platform=Platform.TWITTER,
            documents_per_fetch=documents_per_fetch,
        ),
        Platform.REDDIT: MockAdapter(
            platform=Platform.REDDIT,
            documents_per_fetch=documents_per_fetch,
        ),
        Platform.SUBSTACK: MockAdapter(
            platform=Platform.SUBSTACK,
            documents_per_fetch=documents_per_fetch // 2,  # Lower volume
        ),
        Platform.NEWS: MockAdapter(
            platform=Platform.NEWS,
            documents_per_fetch=documents_per_fetch,
        ),
    }
