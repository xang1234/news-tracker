"""
Tests for Sotwe client and Twitter adapter Sotwe fallback.

Tests are split into:
- Unit tests: No external dependencies, fast, always run
- Integration tests: Require Node.js and network access, marked @pytest.mark.integration
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.twitter_accounts import (
    DEFAULT_USERNAMES,
    get_default_usernames,
    parse_usernames,
)
from src.ingestion.sotwe_client import (
    SotweClient,
    SotweClientError,
    SotweTweet,
    check_node_available,
)
from src.ingestion.twitter_adapter import TwitterAdapter
from src.ingestion.schemas import Platform


class TestTwitterAccounts:
    """Tests for twitter_accounts.py configuration."""

    def test_default_usernames_not_empty(self):
        """Default usernames list should have accounts."""
        usernames = get_default_usernames()
        assert len(usernames) > 0
        assert "SemiAnalysis" in usernames

    def test_default_usernames_returns_copy(self):
        """get_default_usernames should return a copy, not the original."""
        usernames1 = get_default_usernames()
        usernames2 = get_default_usernames()
        usernames1.append("test")
        assert "test" not in usernames2

    def test_parse_usernames_with_valid_string(self):
        """parse_usernames should split comma-separated string."""
        result = parse_usernames("user1,user2,user3")
        assert result == ["user1", "user2", "user3"]

    def test_parse_usernames_strips_whitespace(self):
        """parse_usernames should strip whitespace from usernames."""
        result = parse_usernames(" user1 , user2 , user3 ")
        assert result == ["user1", "user2", "user3"]

    def test_parse_usernames_strips_at_prefix(self):
        """parse_usernames should remove @ prefix."""
        result = parse_usernames("@user1,@user2,user3")
        assert result == ["user1", "user2", "user3"]

    def test_parse_usernames_with_none_returns_defaults(self):
        """parse_usernames should return defaults when None."""
        result = parse_usernames(None)
        assert result == DEFAULT_USERNAMES

    def test_parse_usernames_with_empty_string_returns_defaults(self):
        """parse_usernames should return defaults for empty string."""
        result = parse_usernames("")
        assert result == DEFAULT_USERNAMES

    def test_parse_usernames_filters_empty(self):
        """parse_usernames should filter out empty strings."""
        result = parse_usernames("user1,,user2,")
        assert result == ["user1", "user2"]


class TestSotweTweet:
    """Tests for SotweTweet dataclass."""

    def test_sotwe_tweet_creation(self):
        """SotweTweet should store all fields correctly."""
        tweet = SotweTweet(
            id="123456",
            text="Test tweet about $NVDA",
            created_at=datetime.now(timezone.utc),
            username="testuser",
            author_name="Test User",
            likes=100,
            retweets=50,
            replies=25,
            views=1000,
        )
        assert tweet.id == "123456"
        assert tweet.text == "Test tweet about $NVDA"
        assert tweet.username == "testuser"
        assert tweet.likes == 100
        assert tweet.views == 1000

    def test_sotwe_tweet_default_metrics(self):
        """SotweTweet should default metrics to 0/None."""
        tweet = SotweTweet(
            id="123",
            text="Test",
            created_at=datetime.now(timezone.utc),
            username="user",
            author_name="User",
        )
        assert tweet.likes == 0
        assert tweet.retweets == 0
        assert tweet.replies == 0
        assert tweet.views is None


class TestSotweClient:
    """Unit tests for SotweClient (no network/Node.js required)."""

    def test_check_node_available_always_true(self):
        """check_node_available returns True (Node.js no longer required)."""
        # Node.js is no longer required - the function is kept for compatibility
        assert check_node_available() is True

    def test_is_tweet_like_with_valid_tweet(self):
        """_is_tweet_like should identify valid tweet data."""
        client = SotweClient()
        data = {
            "id": "123456",
            "text": "This is a tweet",
        }
        assert client._is_tweet_like(data) is True

    def test_is_tweet_like_with_full_text(self):
        """_is_tweet_like should recognize full_text field."""
        client = SotweClient()
        data = {
            "id_str": "123456",
            "full_text": "This is a tweet",
        }
        assert client._is_tweet_like(data) is True

    def test_is_tweet_like_without_text(self):
        """_is_tweet_like should reject data without text."""
        client = SotweClient()
        data = {
            "id": "123456",
        }
        assert client._is_tweet_like(data) is False

    def test_is_tweet_like_without_id(self):
        """_is_tweet_like should reject data without ID."""
        client = SotweClient()
        data = {
            "text": "This is a tweet",
        }
        assert client._is_tweet_like(data) is False

    def test_parse_single_tweet_valid_data(self):
        """_parse_single_tweet should parse valid tweet data."""
        client = SotweClient()
        data = {
            "id_str": "123456789",
            "full_text": "NVIDIA announces new GPU $NVDA",
            "created_at": "2024-01-15T10:30:00Z",
            "favorite_count": 100,
            "retweet_count": 50,
            "reply_count": 25,
            "user": {
                "screen_name": "SemiAnalysis",
                "name": "SemiAnalysis",
            },
        }
        tweet = client._parse_single_tweet(data, "default_user")
        assert tweet is not None
        assert tweet.id == "123456789"
        assert tweet.text == "NVIDIA announces new GPU $NVDA"
        assert tweet.username == "SemiAnalysis"
        assert tweet.likes == 100
        assert tweet.retweets == 50
        assert tweet.replies == 25

    def test_parse_single_tweet_uses_default_username(self):
        """_parse_single_tweet should use default username when not in data."""
        client = SotweClient()
        data = {
            "id": "123",
            "text": "Test tweet",
        }
        tweet = client._parse_single_tweet(data, "fallback_user")
        assert tweet is not None
        assert tweet.username == "fallback_user"

    def test_parse_single_tweet_missing_id(self):
        """_parse_single_tweet should return None for missing ID."""
        client = SotweClient()
        data = {
            "text": "Test tweet",
        }
        tweet = client._parse_single_tweet(data, "user")
        assert tweet is None

    def test_parse_single_tweet_missing_text(self):
        """_parse_single_tweet should return None for missing text."""
        client = SotweClient()
        data = {
            "id": "123",
        }
        tweet = client._parse_single_tweet(data, "user")
        assert tweet is None

    def test_extract_metric_with_various_keys(self):
        """_extract_metric should try multiple key names."""
        client = SotweClient()

        # Test favorite_count
        assert client._extract_metric({"favorite_count": 100}, ["favorite_count", "likes"]) == 100

        # Test likes
        assert client._extract_metric({"likes": 50}, ["favorite_count", "likes"]) == 50

        # Test missing key
        assert client._extract_metric({"other": 10}, ["favorite_count", "likes"]) == 0

    def test_parse_timestamp_iso_format(self):
        """_parse_timestamp should handle ISO format."""
        client = SotweClient()
        data = {"created_at": "2024-01-15T10:30:00Z"}
        ts = client._parse_timestamp(data)
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15

    def test_parse_timestamp_twitter_format(self):
        """_parse_timestamp should handle Twitter's format."""
        client = SotweClient()
        data = {"created_at": "Mon Jan 15 10:30:00 +0000 2024"}
        ts = client._parse_timestamp(data)
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15

    def test_parse_timestamp_unix(self):
        """_parse_timestamp should handle Unix timestamp."""
        client = SotweClient()
        data = {"created_at": 1705315800}  # 2024-01-15T10:30:00Z
        ts = client._parse_timestamp(data)
        assert ts.year == 2024

    def test_parse_timestamp_missing(self):
        """_parse_timestamp should return now for missing timestamp."""
        client = SotweClient()
        data = {}
        ts = client._parse_timestamp(data)
        # Should be close to now
        assert (datetime.now(timezone.utc) - ts).total_seconds() < 5

    def test_find_tweets_in_data_flat_list(self):
        """_find_tweets_in_data should find tweets in flat list."""
        client = SotweClient()
        data = [
            {"id": "1", "text": "This is a longer tweet about NVIDIA stock"},
            {"id": "2", "text": "Another substantial tweet about AMD chips"},
        ]
        tweets = client._find_tweets_in_data(data)
        assert len(tweets) == 2

    def test_find_tweets_in_data_nested(self):
        """_find_tweets_in_data should find tweets in nested structure."""
        client = SotweClient()
        data = {
            "user": {
                "timeline": [
                    {"id": "1", "text": "This is a longer tweet about NVIDIA stock"},
                    {"id": "2", "text": "Another substantial tweet about AMD chips"},
                ]
            }
        }
        tweets = client._find_tweets_in_data(data)
        assert len(tweets) == 2


class TestTwitterAdapterSotweFallback:
    """Tests for TwitterAdapter with Sotwe fallback."""

    def test_adapter_prefers_api_when_configured(self):
        """TwitterAdapter should prefer API when bearer token is set."""
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                twitter_bearer_token="test_token",
                sotwe_enabled=True,
                sotwe_usernames=None,
                sotwe_rate_limit=10,
            )
            adapter = TwitterAdapter(bearer_token="test_token")
            assert adapter._bearer_token == "test_token"

    def test_adapter_uses_sotwe_when_no_token(self):
        """TwitterAdapter should enable Sotwe when no bearer token."""
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                twitter_bearer_token=None,
                sotwe_enabled=True,
                sotwe_usernames=None,
                sotwe_rate_limit=10,
            )
            adapter = TwitterAdapter()
            assert adapter._bearer_token is None
            assert adapter._sotwe_enabled is True
            assert len(adapter._sotwe_usernames) > 0

    def test_adapter_sotwe_custom_usernames(self):
        """TwitterAdapter should use custom Sotwe usernames from settings."""
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                twitter_bearer_token=None,
                sotwe_enabled=True,
                sotwe_usernames="user1,user2",
                sotwe_rate_limit=10,
            )
            adapter = TwitterAdapter()
            assert adapter._sotwe_usernames == ["user1", "user2"]

    def test_adapter_transform_sotwe_tweet(self):
        """TwitterAdapter should transform Sotwe tweet correctly."""
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                twitter_bearer_token=None,
                sotwe_enabled=True,
                sotwe_usernames=None,
                sotwe_rate_limit=10,
            )
            adapter = TwitterAdapter()

            tweet = SotweTweet(
                id="123456789",
                text="NVIDIA stock up 5% today $NVDA bullish",
                created_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
                username="SemiAnalysis",
                author_name="SemiAnalysis",
                likes=100,
                retweets=50,
                replies=25,
                views=1000,
            )

            raw = {
                "source": "sotwe",
                "tweet": tweet,
                "username": "SemiAnalysis",
            }

            doc = adapter._transform(raw)
            assert doc is not None
            assert doc.id == "twitter_123456789"
            assert doc.platform == Platform.TWITTER
            assert "NVDA" in doc.tickers_mentioned
            assert doc.engagement.likes == 100
            assert doc.engagement.shares == 50
            assert doc.engagement.views == 1000
            assert doc.author_name == "SemiAnalysis"
            assert doc.raw_data["source"] == "sotwe"
            assert "https://twitter.com/SemiAnalysis/status/123456789" in doc.url

    def test_adapter_transform_api_tweet(self):
        """TwitterAdapter should transform API tweet correctly."""
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                twitter_bearer_token="test_token",
                sotwe_enabled=True,
                sotwe_usernames=None,
                sotwe_rate_limit=10,
            )
            adapter = TwitterAdapter(bearer_token="test_token")

            raw = {
                "source": "twitter_api",
                "tweet": {
                    "id": "987654321",
                    "text": "$AMD announces new chips",
                    "created_at": "2024-01-15T10:30:00Z",
                    "author_id": "12345",
                    "public_metrics": {
                        "like_count": 200,
                        "retweet_count": 75,
                        "reply_count": 30,
                    },
                },
                "author": {
                    "username": "AMD",
                    "verified": True,
                    "public_metrics": {"followers_count": 1000000},
                },
            }

            doc = adapter._transform(raw)
            assert doc is not None
            assert doc.id == "twitter_987654321"
            assert doc.platform == Platform.TWITTER
            assert "AMD" in doc.tickers_mentioned
            assert doc.engagement.likes == 200
            assert doc.raw_data["source"] == "twitter_api"

    async def test_adapter_health_check_with_sotwe(self):
        """health_check should check Sotwe when API not configured."""
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                twitter_bearer_token=None,
                sotwe_enabled=True,
                sotwe_usernames=None,
                sotwe_rate_limit=10,
            )
            adapter = TwitterAdapter()

            # Mock Sotwe client availability check
            adapter._sotwe_client = MagicMock()
            adapter._sotwe_client.check_available = AsyncMock(return_value=True)

            result = await adapter.health_check()
            assert result is True
            adapter._sotwe_client.check_available.assert_called_once()

    async def test_adapter_health_check_sotwe_unavailable(self):
        """health_check should return False when Sotwe unavailable."""
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                twitter_bearer_token=None,
                sotwe_enabled=True,
                sotwe_usernames=None,
                sotwe_rate_limit=10,
            )
            adapter = TwitterAdapter()

            # Mock Sotwe client unavailable
            adapter._sotwe_client = MagicMock()
            adapter._sotwe_client.check_available = AsyncMock(return_value=False)

            result = await adapter.health_check()
            assert result is False


@pytest.mark.integration
class TestSotweIntegration:
    """
    Integration tests that require Node.js and network access.

    Skip with: pytest -m "not integration"
    """

    async def test_sotwe_client_real_fetch(self):
        """Test actual Sotwe fetch (requires Node.js and network)."""
        client = SotweClient()

        if not await client.check_available():
            pytest.skip("Sotwe not available (Node.js missing or blocked)")

        tweets = await client.fetch_user_tweets("nvidia", max_tweets=5)

        # Should get some tweets (nvidia account is active)
        assert len(tweets) > 0

        # Check tweet structure
        for tweet in tweets:
            assert tweet.id
            assert tweet.text
            assert tweet.username

    async def test_twitter_adapter_sotwe_fetch(self):
        """Test TwitterAdapter Sotwe fallback (requires Node.js and network)."""
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                twitter_bearer_token=None,
                sotwe_enabled=True,
                sotwe_usernames="nvidia",  # Single account for speed
                sotwe_rate_limit=10,
            )
            adapter = TwitterAdapter()

            if not await adapter.health_check():
                pytest.skip("Sotwe not available")

            docs = []
            async for doc in adapter.fetch():
                docs.append(doc)
                if len(docs) >= 3:  # Limit for test speed
                    break

            assert len(docs) > 0
            for doc in docs:
                assert doc.platform == Platform.TWITTER
                assert doc.id.startswith("twitter_")
                assert doc.content
