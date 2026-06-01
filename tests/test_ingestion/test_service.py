"""Tests for ingestion service configuration behavior."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.config.settings import Settings
from src.ingestion.schemas import Platform
from src.services.ingestion_service import IngestionConfigurationError, IngestionService


class TestIngestionServiceConfiguration:
    """Validate startup configuration behavior for ingestion."""

    def test_raises_when_all_selected_sources_are_empty(self) -> None:
        settings = SimpleNamespace(
            poll_interval_seconds=60,
            twitter_configured=False,
            xui_configured=False,
            twitter_rate_limit=10,
            reddit_configured=False,
            reddit_rate_limit=10,
            substack_rate_limit=10,
            news_api_configured=False,
            news_rate_limit=10,
            rss_enabled=False,
        )

        with (
            patch("src.services.ingestion_service.get_settings", return_value=settings),
            pytest.raises(
                IngestionConfigurationError,
                match="No ingestion sources are configured",
            ),
        ):
            IngestionService(
                use_mock=False,
                twitter_sources=[],
                reddit_sources=[],
                substack_sources=[],
            )

    def test_default_substack_adapter_counts_as_real_source(self) -> None:
        settings = SimpleNamespace(
            poll_interval_seconds=60,
            twitter_configured=False,
            xui_configured=False,
            twitter_rate_limit=10,
            reddit_configured=False,
            reddit_rate_limit=10,
            substack_rate_limit=10,
            news_api_configured=False,
            news_rate_limit=10,
        )

        with patch("src.services.ingestion_service.get_settings", return_value=settings):
            service = IngestionService(use_mock=False)

        assert Platform.SUBSTACK in service._adapters

    def test_rss_adapter_enabled_when_configured_feeds_exist(self) -> None:
        settings = SimpleNamespace(
            poll_interval_seconds=60,
            twitter_configured=False,
            xui_configured=False,
            twitter_rate_limit=10,
            reddit_configured=False,
            reddit_rate_limit=10,
            substack_rate_limit=10,
            news_api_configured=False,
            news_rate_limit=10,
            rss_enabled=True,
            rss_rate_limit=20,
            rss_max_items_per_feed=50,
            rss_recency_days=7,
            rss_fetch_timeout=15.0,
            rss_full_text_enabled=True,
        )
        feed = SimpleNamespace(enabled=True)

        with (
            patch("src.services.ingestion_service.get_settings", return_value=settings),
            patch("src.services.ingestion_service.FEEDS", [feed]),
            patch("src.services.ingestion_service.FeedAdapter") as feed_adapter,
        ):
            service = IngestionService(use_mock=False, substack_sources=[])

        assert Platform.RSS in service._adapters
        feed_adapter.assert_called_once_with(
            rate_limit=20,
            max_items_per_feed=50,
            recency_days=7,
            fetch_timeout=15.0,
            full_text_enabled=True,
        )

    def test_blank_twitter_bearer_token_does_not_enable_twitter_adapter(self) -> None:
        settings = Settings(
            _env_file=None,
            twitter_bearer_token="   ",
            twitter_xui_enabled=False,
            reddit_client_id=None,
            reddit_client_secret=None,
            news_api_key=None,
            rss_enabled=False,
        )

        with (
            patch("src.services.ingestion_service.get_settings", return_value=settings),
            patch("src.services.ingestion_service.TwitterAdapter") as twitter_adapter,
            pytest.raises(
                IngestionConfigurationError,
                match="No ingestion sources are configured",
            ),
        ):
            IngestionService(
                use_mock=False,
                twitter_sources=["nvidia"],
                reddit_sources=[],
                substack_sources=[],
            )

        assert settings.twitter_configured is False
        twitter_adapter.assert_not_called()

    def test_mock_mode_still_uses_mock_adapters(self) -> None:
        service = IngestionService(use_mock=True)
        assert service._adapters
