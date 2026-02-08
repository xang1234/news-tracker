"""Tests for authority schema validation."""

import pytest

from src.authority.schemas import AuthorityProfile, AuthorTier


class TestAuthorTier:
    """Tests for AuthorTier enum."""

    def test_all_tiers_are_strings(self):
        for tier in AuthorTier:
            assert isinstance(tier.value, str)

    def test_tier_values(self):
        assert AuthorTier.ANONYMOUS.value == "anonymous"
        assert AuthorTier.VERIFIED.value == "verified"
        assert AuthorTier.RESEARCH.value == "research"


class TestAuthorityProfileCreation:
    """Tests for valid AuthorityProfile construction."""

    def test_minimal_profile(self):
        p = AuthorityProfile(author_id="user1", platform="twitter")
        assert p.author_id == "user1"
        assert p.platform == "twitter"
        assert p.tier == "anonymous"
        assert p.base_weight == 1.0
        assert p.total_calls == 0
        assert p.correct_calls == 0
        assert p.last_good_call is None
        assert p.topic_scores == {}
        assert p.centrality_score == 0.0

    def test_full_profile(self, established_profile):
        assert established_profile.author_id == "user_123"
        assert established_profile.total_calls == 20
        assert established_profile.correct_calls == 15

    def test_all_valid_tiers(self):
        for tier in AuthorTier:
            p = AuthorityProfile(
                author_id="x", platform="twitter", tier=tier.value
            )
            assert p.tier == tier.value

    def test_topic_scores_dict(self):
        p = AuthorityProfile(
            author_id="x",
            platform="twitter",
            topic_scores={"ai_chips": {"correct": 5, "total": 8}},
        )
        assert p.topic_scores["ai_chips"]["correct"] == 5


class TestAuthorityProfileValidation:
    """Tests for invalid AuthorityProfile construction."""

    def test_invalid_tier(self):
        with pytest.raises(ValueError, match="Invalid tier"):
            AuthorityProfile(
                author_id="x", platform="twitter", tier="expert"
            )

    def test_negative_total_calls(self):
        with pytest.raises(ValueError, match="total_calls must be non-negative"):
            AuthorityProfile(
                author_id="x", platform="twitter", total_calls=-1
            )

    def test_negative_correct_calls(self):
        with pytest.raises(ValueError, match="correct_calls must be non-negative"):
            AuthorityProfile(
                author_id="x", platform="twitter", correct_calls=-1
            )

    def test_correct_exceeds_total(self):
        with pytest.raises(ValueError, match="correct_calls cannot exceed total_calls"):
            AuthorityProfile(
                author_id="x",
                platform="twitter",
                total_calls=5,
                correct_calls=10,
            )

    def test_equal_correct_and_total_is_valid(self):
        p = AuthorityProfile(
            author_id="x", platform="twitter",
            total_calls=10, correct_calls=10,
        )
        assert p.correct_calls == p.total_calls
