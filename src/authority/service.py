"""Bayesian authority scoring service.

Computes author authority scores using:
- Beta-prior smoothed accuracy: (correct + alpha) / (total + alpha + beta)
- Exponential time decay on recency of last good call
- Probation ramp for new sources (linear over first N days)
- Base weight from author tier (anonymous / verified / research)
- Log-scaled follower normalization

The final score is clamped to [0, 1] for storage on NormalizedDocument.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.authority.config import AuthorityConfig
from src.authority.schemas import AuthorTier, AuthorityProfile

if TYPE_CHECKING:
    from src.authority.repository import AuthorityRepository

logger = logging.getLogger(__name__)


# Known research / specialized outlets (author_name patterns)
_RESEARCH_OUTLETS: frozenset[str] = frozenset({
    "semianalysis",
    "techinsights",
    "ic insights",
    "yole développement",
    "yole group",
    "trendforce",
    "counterpoint",
    "omdia",
    "gartner",
    "idc semiconductor",
})


class AuthorityService:
    """Compute and manage Bayesian authority scores for content authors.

    Usage:
        service = AuthorityService()
        score = service.compute_score(profile, followers=50000)

        # With repository for persistence:
        service = AuthorityService(repository=repo)
        await service.update_track_record(author_id, platform, is_correct=True)
    """

    def __init__(
        self,
        config: AuthorityConfig | None = None,
        repository: "AuthorityRepository | None" = None,
    ) -> None:
        self._config = config or AuthorityConfig()
        self._repo = repository

    def classify_tier(
        self,
        *,
        author_verified: bool = False,
        author_name: str = "",
        platform: str = "",
    ) -> AuthorTier:
        """Classify an author into a base weight tier.

        Args:
            author_verified: Whether the author is platform-verified.
            author_name: Author display name for research outlet matching.
            platform: Source platform.

        Returns:
            AuthorTier classification.
        """
        name_lower = author_name.lower().strip()

        # Check research outlets first (highest tier)
        if any(outlet in name_lower for outlet in _RESEARCH_OUTLETS):
            return AuthorTier.RESEARCH

        # Substack and news platforms get verified tier if verified
        if platform in ("substack", "news") and author_verified:
            return AuthorTier.RESEARCH

        if author_verified:
            return AuthorTier.VERIFIED

        return AuthorTier.ANONYMOUS

    def base_weight_for_tier(self, tier: AuthorTier) -> float:
        """Look up the configured base weight for a tier."""
        weights = {
            AuthorTier.ANONYMOUS: self._config.weight_anonymous,
            AuthorTier.VERIFIED: self._config.weight_verified,
            AuthorTier.RESEARCH: self._config.weight_research,
        }
        return weights[tier]

    def compute_score(
        self,
        profile: AuthorityProfile,
        *,
        followers: int | None = None,
        now: datetime | None = None,
    ) -> float:
        """Compute the authority score for a profile.

        Formula:
            accuracy_smoothed = (correct + alpha) / (total + alpha + beta)
            recency = exp(-lambda * days_since_last_good_call)
            probation = min(1.0, days_active / probation_days)
            follower_norm = min(cap, cap * ln(followers+1) / ln(base+1))
            score = clamp(base_weight * accuracy * recency * probation + follower_norm)

        The score is normalized to [0, 1] by dividing by max possible
        (research_weight * 1.0 * 1.0 * 1.0 + follower_cap).

        Args:
            profile: Author's authority profile.
            followers: Current follower count (optional, for follower bonus).
            now: Reference time (default: UTC now).

        Returns:
            Authority score clamped to [min_score, max_score].
        """
        if now is None:
            now = datetime.now(timezone.utc)

        cfg = self._config

        # 1. Bayesian-smoothed accuracy
        accuracy = (profile.correct_calls + cfg.prior_alpha) / (
            profile.total_calls + cfg.prior_alpha + cfg.prior_beta
        )

        # 2. Recency multiplier (time since last good call)
        if profile.last_good_call is not None:
            days_since = max(0.0, (now - profile.last_good_call).total_seconds() / 86400)
            recency = math.exp(-cfg.decay_lambda * days_since)
        else:
            # No good calls yet — use moderate recency
            recency = 0.5

        # 3. Probation ramp
        days_active = max(0.0, (now - profile.first_seen).total_seconds() / 86400)
        probation = min(1.0, days_active / cfg.probation_days)

        # 4. Base weight from tier
        tier = AuthorTier(profile.tier)
        base_weight = self.base_weight_for_tier(tier)

        # 5. Follower normalization bonus
        follower_bonus = 0.0
        if followers is not None and followers > 0:
            follower_bonus = min(
                cfg.follower_log_cap,
                cfg.follower_log_cap * math.log(followers + 1) / math.log(cfg.follower_log_base + 1),
            )

        # Combine: raw_score = base_weight * accuracy * recency * probation + follower
        raw_score = base_weight * accuracy * recency * probation + follower_bonus

        # Normalize to [0, 1] using max possible raw score
        max_raw = cfg.weight_research * 1.0 * 1.0 * 1.0 + cfg.follower_log_cap
        normalized = raw_score / max_raw if max_raw > 0 else 0.0

        # Clamp
        return max(cfg.min_score, min(cfg.max_score, normalized))

    def compute_score_simple(
        self,
        *,
        author_verified: bool = False,
        author_name: str = "",
        author_followers: int | None = None,
        platform: str = "",
        first_seen: datetime | None = None,
        now: datetime | None = None,
    ) -> float:
        """Compute authority score without a persisted profile.

        Creates an ephemeral profile from document metadata.
        Useful during preprocessing when no DB is available.

        Args:
            author_verified: Whether author is verified.
            author_name: Author display name.
            author_followers: Follower count.
            platform: Source platform.
            first_seen: When this author was first seen (default: now, i.e. new).
            now: Reference time.

        Returns:
            Authority score in [0, 1].
        """
        if now is None:
            now = datetime.now(timezone.utc)
        if first_seen is None:
            first_seen = now

        tier = self.classify_tier(
            author_verified=author_verified,
            author_name=author_name,
            platform=platform,
        )

        profile = AuthorityProfile(
            author_id="ephemeral",
            platform=platform,
            tier=tier.value,
            base_weight=self.base_weight_for_tier(tier),
            first_seen=first_seen,
        )

        return self.compute_score(
            profile,
            followers=author_followers,
            now=now,
        )

    async def get_profile(
        self,
        author_id: str,
        platform: str,
    ) -> AuthorityProfile | None:
        """Retrieve an author's authority profile.

        Args:
            author_id: Author identifier.
            platform: Source platform.

        Returns:
            AuthorityProfile if found, None otherwise.
        """
        if self._repo is None:
            return None
        return await self._repo.get(author_id, platform)

    async def upsert_profile(self, profile: AuthorityProfile) -> AuthorityProfile:
        """Create or update an authority profile.

        Args:
            profile: Profile to persist.

        Returns:
            The persisted profile.

        Raises:
            RuntimeError: If no repository is configured.
        """
        if self._repo is None:
            raise RuntimeError("AuthorityService has no repository configured")
        return await self._repo.upsert(profile)

    async def update_track_record(
        self,
        author_id: str,
        platform: str,
        *,
        is_correct: bool,
        topic: str | None = None,
    ) -> AuthorityProfile | None:
        """Update an author's track record from feedback.

        Increments total_calls, and correct_calls if is_correct.
        Updates last_good_call timestamp on correct calls.
        Optionally updates topic-specific accuracy.

        Args:
            author_id: Author identifier.
            platform: Source platform.
            is_correct: Whether the author's call was correct.
            topic: Optional topic for topic-specific tracking.

        Returns:
            Updated profile, or None if no repository.
        """
        if self._repo is None:
            return None

        profile = await self._repo.get(author_id, platform)
        if profile is None:
            logger.debug(
                "No authority profile for %s/%s, skipping track record update",
                author_id, platform,
            )
            return None

        profile.total_calls += 1
        if is_correct:
            profile.correct_calls += 1
            profile.last_good_call = datetime.now(timezone.utc)

        # Topic-specific tracking
        if topic is not None:
            if topic not in profile.topic_scores:
                profile.topic_scores[topic] = {"correct": 0, "total": 0}
            profile.topic_scores[topic]["total"] += 1
            if is_correct:
                profile.topic_scores[topic]["correct"] += 1

        profile.updated_at = datetime.now(timezone.utc)
        return await self._repo.upsert(profile)

    def compute_topic_score(
        self,
        profile: AuthorityProfile,
        topic: str,
        *,
        followers: int | None = None,
        now: datetime | None = None,
    ) -> float | None:
        """Compute authority score for a specific topic.

        Uses topic-specific correct/total instead of global counts
        when available.

        Args:
            profile: Author's authority profile.
            topic: Topic key to look up in topic_scores.
            followers: Current follower count.
            now: Reference time.

        Returns:
            Topic-specific authority score, or None if no topic data.
        """
        topic_data = profile.topic_scores.get(topic)
        if topic_data is None:
            return None

        # Create a temporary profile with topic-specific counts
        topic_profile = AuthorityProfile(
            author_id=profile.author_id,
            platform=profile.platform,
            tier=profile.tier,
            base_weight=profile.base_weight,
            total_calls=topic_data["total"],
            correct_calls=topic_data["correct"],
            first_seen=profile.first_seen,
            last_good_call=profile.last_good_call,
            centrality_score=profile.centrality_score,
        )

        return self.compute_score(topic_profile, followers=followers, now=now)
