"""Schema definitions for authority profiles.

Maps 1:1 to the ``authority_profiles`` database table. Each profile
tracks a source's Bayesian accuracy, base weight tier, and topic-specific
expertise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AuthorTier(str, Enum):
    """Classification tier for author base weight."""

    ANONYMOUS = "anonymous"
    VERIFIED = "verified"
    RESEARCH = "research"


@dataclass
class AuthorityProfile:
    """A persisted authority profile from the authority_profiles table.

    Attributes:
        author_id: Platform-specific author identifier.
        platform: Source platform (twitter, reddit, substack, news).
        tier: Author classification for base weight lookup.
        base_weight: Platform/verification-derived base weight.
        total_calls: Total feedback-rated items by this author.
        correct_calls: Items rated 'useful' (rating >= 4).
        first_seen: When this author was first ingested.
        last_good_call: Timestamp of most recent correct call.
        topic_scores: Topic-specific accuracy {topic: {correct, total}}.
        centrality_score: PageRank centrality (future: citation graph).
        updated_at: Last profile update timestamp.
    """

    author_id: str
    platform: str
    tier: str = AuthorTier.ANONYMOUS.value
    base_weight: float = 1.0
    total_calls: int = 0
    correct_calls: int = 0
    first_seen: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_good_call: datetime | None = None
    topic_scores: dict[str, dict[str, int]] = field(default_factory=dict)
    centrality_score: float = 0.0
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        valid_tiers = {t.value for t in AuthorTier}
        if self.tier not in valid_tiers:
            raise ValueError(
                f"Invalid tier {self.tier!r}. Must be one of: {sorted(valid_tiers)}"
            )
        if self.total_calls < 0:
            raise ValueError("total_calls must be non-negative")
        if self.correct_calls < 0:
            raise ValueError("correct_calls must be non-negative")
        if self.correct_calls > self.total_calls:
            raise ValueError("correct_calls cannot exceed total_calls")
