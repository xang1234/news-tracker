"""Pytest fixtures for news-tracker tests."""

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator

import pytest

from src.config.settings import Settings
from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Settings configured for testing."""
    return Settings(
        environment="development",
        log_level="DEBUG",
        redis_url="redis://localhost:6379/1",  # Use DB 1 for tests
        database_url="postgresql://postgres:postgres@localhost:5432/news_tracker_test",
    )


@pytest.fixture
def sample_document() -> NormalizedDocument:
    """Create a sample document for testing."""
    return NormalizedDocument(
        id="twitter_123456789",
        platform=Platform.TWITTER,
        url="https://twitter.com/user/status/123456789",
        timestamp=datetime.now(timezone.utc),
        author_id="user_123",
        author_name="test_user",
        author_followers=1000,
        author_verified=True,
        content="$NVDA looking strong heading into earnings. Bullish on AI demand.",
        content_type="post",
        engagement=EngagementMetrics(
            likes=50,
            shares=10,
            comments=5,
        ),
        tickers_mentioned=["NVDA"],
    )


@pytest.fixture
def spam_document() -> NormalizedDocument:
    """Create a spam-like document for testing."""
    return NormalizedDocument(
        id="twitter_spam_001",
        platform=Platform.TWITTER,
        url="https://twitter.com/spammer/status/001",
        timestamp=datetime.now(timezone.utc),
        author_id="spam_bot_123",
        author_name="crypto_guru_42069",
        author_followers=5,
        author_verified=False,
        content="🚀🚀🚀 $NVDA $AMD $INTC $TSM $QCOM $AVGO TO THE MOON!!! JOIN MY DISCORD FOR FREE SIGNALS!!! 🚀🚀🚀",
        content_type="post",
        engagement=EngagementMetrics(likes=2, shares=0, comments=1),
        tickers_mentioned=["NVDA", "AMD", "INTC", "TSM", "QCOM", "AVGO"],
    )


@pytest.fixture
def duplicate_documents() -> list[NormalizedDocument]:
    """Create near-duplicate documents for testing deduplication."""
    # Base content - use longer text for better MinHash accuracy
    base_content = (
        "NVIDIA reports strong Q4 earnings results today, beating analyst expectations "
        "with record datacenter revenue growth driven by unprecedented AI demand. "
        "The company's GPU products continue to dominate the artificial intelligence "
        "training market as customers expand their computing infrastructure."
    )

    return [
        NormalizedDocument(
            id="news_1",
            platform=Platform.NEWS,
            timestamp=datetime.now(timezone.utc),
            author_id="reuters",
            author_name="Reuters",
            content=base_content,
            content_type="article",
            title="NVIDIA Q4 Earnings Beat",
        ),
        NormalizedDocument(
            id="news_2",
            platform=Platform.NEWS,
            timestamp=datetime.now(timezone.utc),
            author_id="yahoo",
            author_name="Yahoo Finance",
            # Near-identical content with minor differences (same words, slight reordering)
            # MinHash with 3-word shingles should detect this as duplicate
            content=(
                "NVIDIA reports strong Q4 earnings results today, beating analyst expectations "
                "with record datacenter revenue growth driven by unprecedented AI demand. "
                "The company's GPU products continue to dominate the artificial intelligence "
                "training market as customers expand their computing infrastructure rapidly."
            ),
            content_type="article",
            title="NVIDIA Q4 Earnings Beat Expectations",
        ),
        NormalizedDocument(
            id="news_3",
            platform=Platform.NEWS,
            timestamp=datetime.now(timezone.utc),
            author_id="bloomberg",
            author_name="Bloomberg",
            # Completely different content (should NOT be duplicate)
            content=(
                "AMD announces new MI300X accelerator chip, directly challenging NVIDIA's "
                "dominance in the AI training chip market. The new product features advanced "
                "memory bandwidth capabilities targeting large language model workloads."
            ),
            content_type="article",
            title="AMD Launches MI300X",
        ),
    ]


@pytest.fixture
def batch_documents() -> list[NormalizedDocument]:
    """Create a batch of documents for testing."""
    platforms = [Platform.TWITTER, Platform.REDDIT, Platform.NEWS, Platform.SUBSTACK]
    tickers = ["NVDA", "AMD", "INTC", "TSM", "QCOM"]

    # Use substantially different content templates to avoid deduplication
    content_templates = [
        "Breaking: {ticker} announces new product line targeting enterprise customers.",
        "Earnings report: {ticker} beats expectations with strong quarterly growth numbers.",
        "Market analysis shows {ticker} gaining market share in competitive landscape.",
        "Investor sentiment positive on {ticker} following analyst upgrade recommendation.",
        "Supply chain improvements help {ticker} address manufacturing capacity challenges.",
        "New partnership announcement between {ticker} and major technology provider.",
        "Research team releases updated price target for {ticker} stock valuation.",
        "Industry trends favor {ticker} positioning in emerging technology sectors.",
        "Management team at {ticker} discusses long-term growth strategy outlook.",
        "Technical indicators suggest {ticker} momentum building among active traders.",
        "Quarterly revenue growth at {ticker} exceeds Wall Street consensus estimates.",
        "Product roadmap reveals {ticker} plans for next generation offerings.",
        "Competitive analysis positions {ticker} as leader in key market segments.",
        "Institutional investors increase holdings in {ticker} during recent period.",
        "Innovation pipeline at {ticker} shows promising research developments.",
        "Cost reduction initiatives help {ticker} improve operating margin metrics.",
        "Customer feedback indicates strong satisfaction with {ticker} solutions.",
        "Global expansion strategy drives {ticker} international revenue growth.",
        "Technology leadership positions {ticker} well for future market opportunities.",
        "Strong balance sheet enables {ticker} to pursue strategic growth initiatives.",
    ]

    docs = []
    for i in range(20):
        platform = platforms[i % len(platforms)]
        ticker = tickers[i % len(tickers)]
        content = content_templates[i].format(ticker=ticker)

        docs.append(
            NormalizedDocument(
                id=f"{platform.value}_{i}",
                platform=platform,
                timestamp=datetime.now(timezone.utc),
                author_id=f"author_{i}",
                author_name=f"user_{i}",
                content=content,
                content_type="post" if platform in [Platform.TWITTER, Platform.REDDIT] else "article",
                tickers_mentioned=[ticker],
            )
        )

    return docs
