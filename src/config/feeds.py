"""Static RSS/Atom feed configuration for generic feed ingestion."""

from __future__ import annotations

from dataclasses import dataclass


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


# Curated source seeding is tracked by news-tracker-781.2. Tests and operators
# can inject feeds into FeedAdapter before the static catalog lands.
FEEDS: list[Feed] = []
