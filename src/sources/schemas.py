"""Data models for the sources module."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Source:
    """A tracked ingestion source (Twitter account, subreddit, Substack publication).

    Uses composite key (platform, identifier) to uniquely identify sources
    across different platforms.
    """

    platform: str
    identifier: str
    display_name: str = ""
    description: str = ""
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class RssSourceHealth:
    """Operator-facing health summary for one RSS source."""

    slug: str
    name: str
    url: str
    category: str
    is_active: bool
    status: str
    is_producing: bool
    recent_document_count: int = 0
    last_fetch_at: str | None = None
    last_success_at: str | None = None
    last_error_at: str | None = None
    last_error: str = ""
    health_status: str = ""
