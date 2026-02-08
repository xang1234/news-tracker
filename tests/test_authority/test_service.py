"""Tests for the Bayesian authority scoring service."""

from datetime import timedelta

import pytest

from src.authority.config import AuthorityConfig
from src.authority.schemas import AuthorityProfile, AuthorTier
from src.authority.service import AuthorityService


class TestClassifyTier:
    """Tests for author tier classification."""

    def test_anonymous_default(self, service):
        tier = service.classify_tier()
        assert tier == AuthorTier.ANONYMOUS

    def test_verified_author(self, service):
        tier = service.classify_tier(author_verified=True)
        assert tier == AuthorTier.VERIFIED

    def test_research_outlet_by_name(self, service):
        tier = service.classify_tier(author_name="SemiAnalysis")
        assert tier == AuthorTier.RESEARCH

    def test_research_outlet_case_insensitive(self, service):
        tier = service.classify_tier(author_name="TRENDFORCE Research")
        assert tier == AuthorTier.RESEARCH

    def test_verified_substack_is_research(self, service):
        tier = service.classify_tier(
            author_verified=True, platform="substack"
        )
        assert tier == AuthorTier.RESEARCH

    def test_verified_news_is_research(self, service):
        tier = service.classify_tier(
            author_verified=True, platform="news"
        )
        assert tier == AuthorTier.RESEARCH

    def test_research_name_takes_priority_over_verified(self, service):
        tier = service.classify_tier(
            author_verified=True, author_name="SemiAnalysis"
        )
        assert tier == AuthorTier.RESEARCH


class TestBaseWeight:
    """Tests for base weight lookup."""

    def test_anonymous_weight(self, service):
        assert service.base_weight_for_tier(AuthorTier.ANONYMOUS) == 1.0

    def test_verified_weight(self, service):
        assert service.base_weight_for_tier(AuthorTier.VERIFIED) == 5.0

    def test_research_weight(self, service):
        assert service.base_weight_for_tier(AuthorTier.RESEARCH) == 10.0


class TestComputeScore:
    """Tests for the main scoring formula."""

    def test_score_in_valid_range(self, service, established_profile, now):
        score = service.compute_score(established_profile, now=now)
        assert 0.0 <= score <= 1.0

    def test_research_scores_higher_than_anonymous(
        self, service, established_profile, research_profile, now
    ):
        anon_score = service.compute_score(established_profile, now=now)
        research_score = service.compute_score(research_profile, now=now)
        assert research_score > anon_score

    def test_verified_scores_between_anon_and_research(
        self, service, established_profile, verified_profile, research_profile, now
    ):
        anon = service.compute_score(established_profile, now=now)
        verified = service.compute_score(verified_profile, now=now)
        research = service.compute_score(research_profile, now=now)
        assert anon < verified < research

    def test_new_source_probation_penalty(self, service, new_profile, now):
        """A 5-day-old source should score lower than a 90-day-old one."""
        old_profile = AuthorityProfile(
            author_id="old",
            platform="twitter",
            tier=AuthorTier.ANONYMOUS.value,
            first_seen=now - timedelta(days=90),
        )
        new_score = service.compute_score(new_profile, now=now)
        old_score = service.compute_score(old_profile, now=now)
        assert new_score < old_score

    def test_probation_ramp_at_30_days(self, service, now):
        """After 30 days, probation should be 1.0 (no penalty)."""
        profile_30d = AuthorityProfile(
            author_id="user",
            platform="twitter",
            first_seen=now - timedelta(days=30),
        )
        profile_60d = AuthorityProfile(
            author_id="user",
            platform="twitter",
            first_seen=now - timedelta(days=60),
        )
        score_30d = service.compute_score(profile_30d, now=now)
        score_60d = service.compute_score(profile_60d, now=now)
        # Both past probation period — should be equal
        assert abs(score_30d - score_60d) < 0.01

    def test_stale_source_decays(self, service, stale_profile, now):
        """A source with last_good_call 200 days ago should decay significantly."""
        fresh = AuthorityProfile(
            author_id="fresh",
            platform="reddit",
            tier=AuthorTier.ANONYMOUS.value,
            total_calls=10,
            correct_calls=5,
            first_seen=now - timedelta(days=365),
            last_good_call=now - timedelta(hours=6),
        )
        stale_score = service.compute_score(stale_profile, now=now)
        fresh_score = service.compute_score(fresh, now=now)
        assert stale_score < fresh_score

    def test_no_good_calls_gets_moderate_recency(self, service, now):
        """Source with no last_good_call gets recency=0.5."""
        profile = AuthorityProfile(
            author_id="never_good",
            platform="twitter",
            first_seen=now - timedelta(days=60),
            last_good_call=None,
        )
        score = service.compute_score(profile, now=now)
        assert 0.0 < score < 1.0

    def test_followers_boost_score(self, service, established_profile, now):
        no_followers = service.compute_score(
            established_profile, followers=0, now=now
        )
        with_followers = service.compute_score(
            established_profile, followers=100_000, now=now
        )
        assert with_followers > no_followers

    def test_zero_followers_no_crash(self, service, established_profile, now):
        score = service.compute_score(
            established_profile, followers=0, now=now
        )
        assert 0.0 <= score <= 1.0

    def test_none_followers_no_crash(self, service, established_profile, now):
        score = service.compute_score(
            established_profile, followers=None, now=now
        )
        assert 0.0 <= score <= 1.0

    def test_perfect_accuracy_high_score(self, service, now):
        perfect = AuthorityProfile(
            author_id="perfect",
            platform="twitter",
            tier=AuthorTier.RESEARCH.value,
            base_weight=10.0,
            total_calls=100,
            correct_calls=100,
            first_seen=now - timedelta(days=365),
            last_good_call=now - timedelta(hours=1),
        )
        score = service.compute_score(perfect, followers=500_000, now=now)
        assert score > 0.8

    def test_zero_accuracy_low_score(self, service, now):
        bad = AuthorityProfile(
            author_id="bad",
            platform="twitter",
            tier=AuthorTier.ANONYMOUS.value,
            total_calls=100,
            correct_calls=0,
            first_seen=now - timedelta(days=365),
            last_good_call=None,
        )
        score = service.compute_score(bad, now=now)
        assert score < 0.1


class TestComputeScoreSimple:
    """Tests for the simplified (no profile) scoring path."""

    def test_anonymous_unverified(self, service, now):
        score = service.compute_score_simple(
            author_verified=False,
            author_name="randomuser123",
            platform="twitter",
            now=now,
        )
        assert 0.0 <= score <= 1.0

    def test_verified_twitter_user(self, service, now):
        first_seen = now - timedelta(days=60)
        anon_score = service.compute_score_simple(
            author_verified=False, platform="twitter",
            first_seen=first_seen, now=now,
        )
        verified_score = service.compute_score_simple(
            author_verified=True, platform="twitter",
            first_seen=first_seen, now=now,
        )
        assert verified_score > anon_score

    def test_research_outlet(self, service, now):
        first_seen = now - timedelta(days=60)
        score = service.compute_score_simple(
            author_name="SemiAnalysis",
            platform="substack",
            first_seen=first_seen, now=now,
        )
        anon = service.compute_score_simple(
            author_name="random",
            platform="twitter",
            first_seen=first_seen, now=now,
        )
        assert score > anon

    def test_new_source_default_is_now(self, service, now):
        """When first_seen is None, default is now → probation=0."""
        score = service.compute_score_simple(
            author_verified=False,
            platform="twitter",
            first_seen=None,
            now=now,
        )
        # Brand new source has 0 days → probation = 0/30 = 0
        assert score == 0.0

    def test_established_source(self, service, now):
        score = service.compute_score_simple(
            author_verified=True,
            author_followers=50000,
            platform="twitter",
            first_seen=now - timedelta(days=60),
            now=now,
        )
        assert score > 0.0


class TestTopicScore:
    """Tests for topic-specific authority scoring."""

    def test_topic_with_data(self, service, now):
        profile = AuthorityProfile(
            author_id="topic_expert",
            platform="twitter",
            tier=AuthorTier.VERIFIED.value,
            base_weight=5.0,
            total_calls=50,
            correct_calls=30,
            first_seen=now - timedelta(days=180),
            last_good_call=now - timedelta(days=1),
            topic_scores={
                "ai_chips": {"correct": 18, "total": 20},
                "memory": {"correct": 2, "total": 15},
            },
        )
        ai_score = service.compute_topic_score(profile, "ai_chips", now=now)
        memory_score = service.compute_topic_score(profile, "memory", now=now)
        assert ai_score is not None
        assert memory_score is not None
        assert ai_score > memory_score

    def test_topic_missing_returns_none(self, service, established_profile, now):
        result = service.compute_topic_score(
            established_profile, "nonexistent_topic", now=now
        )
        assert result is None


class TestUpdateTrackRecord:
    """Tests for track record updates (async)."""

    @pytest.mark.asyncio
    async def test_no_repo_returns_none(self, service):
        result = await service.update_track_record(
            "user", "twitter", is_correct=True
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_profile_returns_none(self, service_with_repo, repository):
        repository.get = AsyncMock(return_value=None)
        result = await service_with_repo.update_track_record(
            "unknown", "twitter", is_correct=True
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_correct_call_increments(
        self, service_with_repo, repository, established_profile
    ):
        repository.get = AsyncMock(return_value=established_profile)
        repository.upsert = AsyncMock(side_effect=lambda p: p)

        result = await service_with_repo.update_track_record(
            "user_123", "twitter", is_correct=True
        )
        assert result is not None
        assert result.total_calls == 21
        assert result.correct_calls == 16
        assert result.last_good_call is not None

    @pytest.mark.asyncio
    async def test_incorrect_call_increments_total_only(
        self, service_with_repo, repository, established_profile
    ):
        original_good_call = established_profile.last_good_call
        repository.get = AsyncMock(return_value=established_profile)
        repository.upsert = AsyncMock(side_effect=lambda p: p)

        result = await service_with_repo.update_track_record(
            "user_123", "twitter", is_correct=False
        )
        assert result.total_calls == 21
        assert result.correct_calls == 15  # unchanged
        assert result.last_good_call == original_good_call

    @pytest.mark.asyncio
    async def test_topic_tracking(
        self, service_with_repo, repository, established_profile
    ):
        repository.get = AsyncMock(return_value=established_profile)
        repository.upsert = AsyncMock(side_effect=lambda p: p)

        result = await service_with_repo.update_track_record(
            "user_123", "twitter", is_correct=True, topic="ai_chips"
        )
        assert "ai_chips" in result.topic_scores
        assert result.topic_scores["ai_chips"]["correct"] == 1
        assert result.topic_scores["ai_chips"]["total"] == 1


class TestConfigOverrides:
    """Tests for custom configuration."""

    def test_custom_prior(self, now):
        # Stronger prior = more conservative
        conservative = AuthorityConfig(prior_alpha=1.0, prior_beta=10.0)
        liberal = AuthorityConfig(prior_alpha=5.0, prior_beta=2.0)

        profile = AuthorityProfile(
            author_id="test",
            platform="twitter",
            total_calls=5,
            correct_calls=4,
            first_seen=now - timedelta(days=60),
            last_good_call=now - timedelta(days=1),
        )

        conservative_score = AuthorityService(config=conservative).compute_score(
            profile, now=now
        )
        liberal_score = AuthorityService(config=liberal).compute_score(
            profile, now=now
        )
        assert liberal_score > conservative_score

    def test_custom_probation_days(self, now):
        short_probation = AuthorityConfig(probation_days=7)
        long_probation = AuthorityConfig(probation_days=90)

        profile = AuthorityProfile(
            author_id="test",
            platform="twitter",
            first_seen=now - timedelta(days=10),
        )

        short_score = AuthorityService(config=short_probation).compute_score(
            profile, now=now
        )
        long_score = AuthorityService(config=long_probation).compute_score(
            profile, now=now
        )
        assert short_score > long_score


# Import at bottom to avoid issues with conftest fixtures
from unittest.mock import AsyncMock
