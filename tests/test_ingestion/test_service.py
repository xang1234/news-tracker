"""Tests for ingestion service configuration behavior."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

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

    def test_mock_mode_still_uses_mock_adapters(self) -> None:
        service = IngestionService(use_mock=True)
        assert service._adapters
