"""Tests for the curated RSS/Atom feed catalog."""

from __future__ import annotations

from src.config.feeds import FEED_CATEGORIES, FEEDS, Feed, validate_feed_catalog


def test_curated_catalog_has_starter_stock_feed_coverage() -> None:
    enabled_feeds = [feed for feed in FEEDS if feed.enabled]
    slugs = {feed.slug for feed in enabled_feeds}
    categories = {feed.category for feed in enabled_feeds}

    assert 15 <= len(enabled_feeds) <= 25
    assert {"company_ir", "trade_press", "tech_press"}.issubset(categories)
    assert {
        "nvidia-press-releases",
        "amd-press-releases",
        "intel-newsroom",
        "micron-press-releases",
        "samsung-semiconductor-newsroom",
        "kla-press-releases",
        "semiconductor-engineering",
        "ee-times",
        "semiconductor-digest",
        "semiconductor-today-news",
        "semiwiki",
    }.issubset(slugs)


def test_curated_catalog_entries_are_complete_and_valid() -> None:
    assert validate_feed_catalog(FEEDS) == []
    assert frozenset({"company_ir", "company_tech", "trade_press", "tech_press"}) == (
        FEED_CATEGORIES
    )

    for feed in FEEDS:
        assert feed.slug
        assert feed.name
        assert feed.url
        assert feed.category in FEED_CATEGORIES
        assert isinstance(feed.enabled, bool)
        assert isinstance(feed.full_text, bool)


def test_catalog_validation_catches_duplicate_and_malformed_urls() -> None:
    feeds = [
        Feed(
            slug="valid-feed",
            name="Valid Feed",
            url="https://example.com/feed.xml",
            category="trade_press",
        ),
        Feed(
            slug="valid-feed",
            name="Duplicate Slug",
            url="https://example.com/other.xml",
            category="trade_press",
        ),
        Feed(
            slug="duplicate-url",
            name="Duplicate URL",
            url="https://example.com/feed.xml",
            category="trade_press",
        ),
        Feed(
            slug="malformed-url",
            name="Malformed URL",
            url="ftp://example.com/feed.xml",
            category="trade_press",
        ),
        Feed(
            slug="unknown-category",
            name="Unknown Category",
            url="https://example.com/unknown.xml",
            category="general_news",
        ),
    ]

    issues = validate_feed_catalog(feeds)

    assert {issue.code for issue in issues} >= {
        "duplicate_slug",
        "duplicate_url",
        "malformed_url",
        "unknown_category",
    }
