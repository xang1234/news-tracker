"""Data models for the sources module."""

from dataclasses import dataclass, field
from datetime import datetime


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
    metadata: dict = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
