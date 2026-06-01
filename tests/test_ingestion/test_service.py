"""Tests for ingestion service configuration behavior."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from src.config.feeds import Feed
from src.config.settings import Settings
from src.ingestion.base_adapter import BaseAdapter
from src.ingestion.schemas import NormalizedDocument, Platform
from src.services.ingestion_service import IngestionConfigurationError, IngestionService


class RecordingQueue:
    def __init__(self) -> None:
        self.published: list[NormalizedDocument] = []
        self.closed = False

    async def connect(self) -> None:
        return None

    async def publish(self, doc: NormalizedDocument) -> None:
        self.published.append(doc)

    async def close(self) -> None:
        self.closed = True


class RssHealthAdapter:
    def __init__(self, snapshots: list[dict[str, object]]) -> None:
        self._snapshots = snapshots

    async def fetch(self) -> AsyncIterator[NormalizedDocument]:
        if False:
            yield cast(NormalizedDocument, object())

    async def health_check(self) -> bool:
        return True

    def feed_health_snapshots(self) -> list[dict[str, object]]:
        return [dict(snapshot) for snapshot in self._snapshots]


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
            feeds=[feed],
            rate_limit=20,
            max_items_per_feed=50,
            recency_days=7,
            fetch_timeout=15.0,
            full_text_enabled=True,
        )

    def test_rss_adapter_uses_db_backed_feeds_when_supplied(self) -> None:
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
        db_feed = Feed(
            slug="semiwiki",
            name="SemiWiki",
            url="https://semiwiki.com/feed/",
            category="trade_press",
        )

        with (
            patch("src.services.ingestion_service.get_settings", return_value=settings),
            patch("src.services.ingestion_service.FeedAdapter") as feed_adapter,
        ):
            service = IngestionService(
                use_mock=False,
                substack_sources=[],
                rss_feeds=[db_feed],
            )

        assert Platform.RSS in service._adapters
        feed_adapter.assert_called_once_with(
            feeds=[db_feed],
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

    @pytest.mark.asyncio
    async def test_run_once_publishes_rss_health_snapshots(self) -> None:
        snapshot = {
            "slug": "semiwiki",
            "status": "ok",
            "recent_document_count": 0,
        }
        recorded: list[tuple[str, dict[str, object]]] = []

        async def sink(slug: str, health: dict[str, object]) -> bool:
            recorded.append((slug, dict(health)))
            return True

        service = IngestionService(
            adapters={Platform.RSS: cast(BaseAdapter, RssHealthAdapter([snapshot]))},
            queue=cast(Any, RecordingQueue()),
            rss_health_sink=sink,
        )

        results = await service.run_once()

        assert results == {Platform.RSS: 0}
        assert recorded == [("semiwiki", snapshot)]

    @pytest.mark.asyncio
    async def test_run_once_ignores_rss_health_sink_failure(self) -> None:
        async def failing_sink(slug: str, health: dict[str, object]) -> bool:
            raise RuntimeError(f"cannot persist {slug}")

        service = IngestionService(
            adapters={
                Platform.RSS: cast(
                    BaseAdapter,
                    RssHealthAdapter([{"slug": "semiwiki", "status": "ok"}]),
                )
            },
            queue=cast(Any, RecordingQueue()),
            rss_health_sink=failing_sink,
        )

        assert await service.run_once() == {Platform.RSS: 0}
