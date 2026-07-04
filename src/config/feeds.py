"""Static RSS/Atom feed configuration for generic feed ingestion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

FEED_CATEGORIES = frozenset({"company_ir", "company_tech", "trade_press", "tech_press"})
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Feed:
    """Configuration for one RSS/Atom feed source."""

    slug: str
    name: str
    url: str
    category: str
    authority: str = "standard"
    full_text: bool = False
    enabled: bool = True

    def to_metadata(self) -> dict[str, object]:
        """Serialize stable source metadata for raw document lineage."""
        return {
            "slug": self.slug,
            "name": self.name,
            "url": self.url,
            "category": self.category,
            "authority": self.authority,
            "full_text": self.full_text,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class FeedCatalogIssue:
    """One static feed catalog validation issue."""

    code: str
    feed_slug: str
    message: str


def validate_feed_catalog(feeds: Sequence[Feed]) -> list[FeedCatalogIssue]:
    """Validate static feed metadata without making network calls."""
    issues: list[FeedCatalogIssue] = []
    seen_slugs: dict[str, str] = {}
    seen_urls: dict[str, str] = {}

    for feed in feeds:
        slug = feed.slug.strip()
        if not slug or not _SLUG_RE.fullmatch(slug):
            issues.append(
                FeedCatalogIssue(
                    code="malformed_slug",
                    feed_slug=feed.slug,
                    message="Feed slug must be lower kebab-case ASCII.",
                )
            )
        elif slug in seen_slugs:
            issues.append(
                FeedCatalogIssue(
                    code="duplicate_slug",
                    feed_slug=feed.slug,
                    message=f"Feed slug duplicates {seen_slugs[slug]}.",
                )
            )
        else:
            seen_slugs[slug] = feed.slug

        if not feed.name.strip():
            issues.append(
                FeedCatalogIssue(
                    code="missing_name",
                    feed_slug=feed.slug,
                    message="Feed display name is required.",
                )
            )

        if feed.category not in FEED_CATEGORIES:
            issues.append(
                FeedCatalogIssue(
                    code="unknown_category",
                    feed_slug=feed.slug,
                    message=f"Feed category must be one of {sorted(FEED_CATEGORIES)}.",
                )
            )

        normalized_url = _normalize_url(feed.url)
        if normalized_url is None:
            issues.append(
                FeedCatalogIssue(
                    code="malformed_url",
                    feed_slug=feed.slug,
                    message="Feed URL must be an absolute HTTPS URL.",
                )
            )
        elif normalized_url in seen_urls:
            issues.append(
                FeedCatalogIssue(
                    code="duplicate_url",
                    feed_slug=feed.slug,
                    message=f"Feed URL duplicates {seen_urls[normalized_url]}.",
                )
            )
        else:
            seen_urls[normalized_url] = feed.slug

        if not isinstance(feed.enabled, bool):
            issues.append(
                FeedCatalogIssue(
                    code="invalid_enabled",
                    feed_slug=feed.slug,
                    message="Feed enabled flag must be boolean.",
                )
            )
        if not isinstance(feed.full_text, bool):
            issues.append(
                FeedCatalogIssue(
                    code="invalid_full_text",
                    feed_slug=feed.slug,
                    message="Feed full-text policy must be boolean.",
                )
            )

    return issues


def _normalize_url(url: str) -> str | None:
    parsed = urlsplit(url.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


FEEDS: list[Feed] = [
    Feed(
        slug="nvidia-press-releases",
        name="NVIDIA Newsroom Press Releases",
        url="https://nvidianews.nvidia.com/cats/press_release.xml",
        category="company_ir",
        authority="official",
        full_text=True,
    ),
    Feed(
        slug="nvidia-technical-blog",
        name="NVIDIA Technical Blog",
        url="https://developer.nvidia.com/blog/feed/",
        category="company_tech",
        authority="official",
        full_text=False,
    ),
    Feed(
        slug="amd-press-releases",
        name="AMD Press Releases",
        url="https://ir.amd.com/news-events/press-releases/rss",
        category="company_ir",
        authority="official",
        full_text=True,
    ),
    Feed(
        slug="intel-newsroom",
        name="Intel Newsroom",
        url="https://newsroom.intel.com/feed/",
        category="company_ir",
        authority="official",
        full_text=True,
    ),
    Feed(
        slug="micron-press-releases",
        name="Micron Technology News Releases",
        url="https://investors.micron.com/rss/news-releases.xml",
        category="company_ir",
        authority="official",
        full_text=True,
    ),
    Feed(
        slug="kla-press-releases",
        name="KLA Corporation Press Releases",
        url="https://ir.kla.com/news-events/press-releases/rss",
        category="company_ir",
        authority="official",
        full_text=True,
    ),
    Feed(
        slug="samsung-global-press-releases",
        name="Samsung Global Newsroom Press Releases",
        url="https://news.samsung.com/global/category/corporate/press-release/feed",
        category="company_ir",
        authority="official",
        full_text=True,
    ),
    Feed(
        slug="samsung-global-semiconductor",
        name="Samsung Global Newsroom Semiconductor",
        url="https://news.samsung.com/global/tag/semiconductor/feed",
        category="company_tech",
        authority="official",
        full_text=True,
    ),
    Feed(
        slug="samsung-semiconductor-newsroom",
        name="Samsung Semiconductor Global Newsroom",
        url="https://news.samsungsemiconductor.com/global/feed/",
        category="company_tech",
        authority="official",
        full_text=True,
    ),
    Feed(
        slug="semiconductor-engineering",
        name="Semiconductor Engineering",
        url="https://semiengineering.com/feed/",
        category="trade_press",
        authority="specialist",
        full_text=True,
    ),
    Feed(
        slug="ee-times",
        name="EE Times",
        url="https://www.eetimes.com/feed/",
        category="trade_press",
        authority="specialist",
        full_text=True,
    ),
    Feed(
        slug="semiconductor-digest",
        name="Semiconductor Digest",
        url="https://sst.semiconductor-digest.com/feed/",
        category="trade_press",
        authority="specialist",
        full_text=True,
    ),
    Feed(
        slug="semiconductor-today-news",
        name="Semiconductor Today News",
        url="https://www.semiconductor-today.com/rss/news.xml",
        category="trade_press",
        authority="specialist",
        full_text=True,
    ),
    Feed(
        slug="semiwiki",
        name="SemiWiki",
        url="https://semiwiki.com/feed/",
        category="trade_press",
        authority="specialist",
        full_text=True,
    ),
]
