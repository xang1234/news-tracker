"""Unit tests for sources admin routes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.admin_models import BulkCreateSourcesRequest, CreateSourceRequest
from src.api.routes.sources import (
    _INGESTION_LOCK_KEY,
    create_source,
    deactivate_source,
    list_sources,
    trigger_ingestion,
)
from src.config.feeds import Feed
from src.ingestion.schemas import Platform
from src.services.ingestion_service import IngestionConfigurationError
from src.sources.schemas import Source


def _build_request() -> SimpleNamespace:
    services = SimpleNamespace(
        get_database=AsyncMock(return_value=object()),
        close=AsyncMock(),
    )
    state = SimpleNamespace(background_tasks=set(), services=services)
    return SimpleNamespace(app=SimpleNamespace(state=state))


class _FakeSourcesService:
    def __init__(self, db) -> None:
        self._db = db

    async def get_twitter_sources(self) -> list[str]:
        return ["SemiAnalysis"]

    async def get_reddit_sources(self) -> list[str]:
        return ["wallstreetbets"]

    async def get_substack_sources(self) -> list[tuple[str, str, str]]:
        return [("semianalysis", "SemiAnalysis", "Semiconductor deep dives")]

    async def get_rss_feeds(self) -> list[Feed]:
        return [
            Feed(
                slug="semiwiki",
                name="SemiWiki",
                url="https://semiwiki.com/feed/",
                category="trade_press",
            )
        ]


class _EmptySourcesService:
    def __init__(self, db) -> None:
        self._db = db

    async def get_twitter_sources(self) -> list[str]:
        return []

    async def get_reddit_sources(self) -> list[str]:
        return []

    async def get_substack_sources(self) -> list[tuple[str, str, str]]:
        return []

    async def get_rss_feeds(self) -> list[Feed]:
        return []


class _FakeIngestionService:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def run_once(self) -> dict[Platform, int]:
        return {Platform.TWITTER: 1}


class _FakeSourcesRepository:
    def __init__(self, sources: list[Source] | None = None) -> None:
        self.sources = sources or []
        self.upserted: Source | None = None
        self.deactivated: tuple[str, str] | None = None

    async def list_sources(
        self,
        platform: str | None = None,
        search: str | None = None,
        active_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Source], int]:
        return self.sources, len(self.sources)

    async def upsert(self, source: Source) -> None:
        self.upserted = source

    async def get_by_key(self, platform: str, identifier: str) -> Source | None:
        if (
            self.upserted
            and self.upserted.platform == platform
            and self.upserted.identifier == identifier
        ):
            return self.upserted
        return next(
            (s for s in self.sources if s.platform == platform and s.identifier == identifier),
            None,
        )

    async def deactivate(self, platform: str, identifier: str) -> bool:
        self.deactivated = (platform, identifier)
        return True


@pytest.mark.asyncio
async def test_list_sources_returns_rss_feed_metadata() -> None:
    request = _build_request()
    repo = _FakeSourcesRepository(
        [
            Source(
                platform="rss",
                identifier="semiwiki",
                display_name="SemiWiki",
                description="Semiconductor trade analysis",
                metadata={
                    "url": "https://semiwiki.com/feed/",
                    "category": "trade_press",
                    "authority": "specialist",
                    "full_text": True,
                },
            )
        ]
    )
    settings = SimpleNamespace(sources_enabled=True)

    with patch("src.api.routes.sources._get_settings", return_value=settings):
        response = await list_sources(
            request=request,
            platform="rss",
            search=None,
            active_only=False,
            limit=50,
            offset=0,
            api_key="test-key",
            repo=repo,
        )

    assert response.total == 1
    assert response.sources[0].platform == "rss"
    assert response.sources[0].identifier == "semiwiki"
    assert response.sources[0].metadata["url"] == "https://semiwiki.com/feed/"


@pytest.mark.asyncio
async def test_create_source_accepts_rss_feed_metadata() -> None:
    request = _build_request()
    repo = _FakeSourcesRepository()
    settings = SimpleNamespace(sources_enabled=True)
    body = CreateSourceRequest(
        platform="rss",
        identifier="nvidia-press-releases",
        display_name="NVIDIA Newsroom Press Releases",
        description="Official NVIDIA press releases",
        metadata={
            "url": "https://nvidianews.nvidia.com/cats/press_release.xml",
            "category": "company_ir",
            "authority": "official",
            "full_text": True,
        },
    )

    with patch("src.api.routes.sources._get_settings", return_value=settings):
        response = await create_source(
            request=request,
            body=body,
            api_key="test-key",
            repo=repo,
        )

    assert response.platform == "rss"
    assert response.identifier == "nvidia-press-releases"
    assert repo.upserted is not None
    assert repo.upserted.metadata["category"] == "company_ir"
    assert repo.upserted.metadata["full_text"] is True


def test_create_source_requires_rss_feed_url_and_category() -> None:
    with pytest.raises(ValidationError, match="metadata.url"):
        CreateSourceRequest(
            platform="rss",
            identifier="bad-feed",
            display_name="Bad Feed",
            metadata={"category": "trade_press"},
        )


def test_bulk_create_rejects_rss_sources_without_metadata() -> None:
    with pytest.raises(ValidationError, match="RSS sources require metadata"):
        BulkCreateSourcesRequest(platform="rss", identifiers=["bad-feed"])


@pytest.mark.asyncio
async def test_deactivate_source_supports_rss_feeds() -> None:
    request = _build_request()
    repo = _FakeSourcesRepository()
    settings = SimpleNamespace(sources_enabled=True)

    with patch("src.api.routes.sources._get_settings", return_value=settings):
        await deactivate_source(
            request=request,
            platform="rss",
            identifier="semiwiki",
            api_key="test-key",
            repo=repo,
        )

    assert repo.deactivated == ("rss", "semiwiki")


@pytest.mark.asyncio
async def test_trigger_ingestion_returns_conflict_when_lock_exists() -> None:
    request = _build_request()
    redis_client = AsyncMock()
    redis_client.set.return_value = None

    settings = SimpleNamespace(
        sources_enabled=True,
        sources_trigger_lock_ttl_seconds=3600,
    )

    with (
        patch("src.api.routes.sources._get_settings", return_value=settings),
        pytest.raises(HTTPException) as exc_info,
    ):
        await trigger_ingestion(request=request, api_key="test-key", redis_client=redis_client)

    assert exc_info.value.status_code == 409
    redis_client.execute_command.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_ingestion_returns_error_when_preflight_fails() -> None:
    request = _build_request()
    redis_client = AsyncMock()
    redis_client.set.return_value = True

    settings = SimpleNamespace(
        sources_enabled=True,
        sources_trigger_lock_ttl_seconds=3600,
    )

    with (
        patch("src.api.routes.sources._get_settings", return_value=settings),
        patch("src.sources.service.SourcesService", _FakeSourcesService),
        patch(
            "src.services.ingestion_service.IngestionService",
            side_effect=IngestionConfigurationError("No ingestion sources are configured."),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await trigger_ingestion(request=request, api_key="test-key", redis_client=redis_client)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "No ingestion sources are configured."
    assert not request.app.state.background_tasks
    redis_client.execute_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_trigger_ingestion_returns_503_when_all_active_sources_are_disabled() -> None:
    request = _build_request()
    redis_client = AsyncMock()
    redis_client.set.return_value = True

    route_settings = SimpleNamespace(
        sources_enabled=True,
        sources_trigger_lock_ttl_seconds=3600,
    )
    service_settings = SimpleNamespace(
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
        patch("src.api.routes.sources._get_settings", return_value=route_settings),
        patch("src.services.ingestion_service.get_settings", return_value=service_settings),
        patch("src.sources.service.SourcesService", _EmptySourcesService),
        pytest.raises(HTTPException) as exc_info,
    ):
        await trigger_ingestion(request=request, api_key="test-key", redis_client=redis_client)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == (
        "No ingestion sources are configured. Set real source credentials "
        "or run with --mock for synthetic data."
    )
    assert not request.app.state.background_tasks
    redis_client.execute_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_trigger_ingestion_releases_distributed_lock() -> None:
    request = _build_request()
    redis_client = AsyncMock()
    redis_client.set.return_value = True

    settings = SimpleNamespace(
        sources_enabled=True,
        sources_trigger_lock_ttl_seconds=3600,
    )

    with (
        patch("src.api.routes.sources._get_settings", return_value=settings),
        patch("src.sources.service.SourcesService", _FakeSourcesService),
        patch("src.services.ingestion_service.IngestionService", _FakeIngestionService),
    ):
        response = await trigger_ingestion(
            request=request,
            api_key="test-key",
            redis_client=redis_client,
        )
        await asyncio.gather(*tuple(request.app.state.background_tasks))

    assert response.status == "started"
    redis_client.set.assert_awaited_once()
    set_args = redis_client.set.await_args.args
    set_kwargs = redis_client.set.await_args.kwargs
    assert set_args[0] == _INGESTION_LOCK_KEY
    assert set_kwargs == {"ex": 3600, "nx": True}

    redis_client.execute_command.assert_awaited_once()
    command_args = redis_client.execute_command.await_args.args
    assert command_args[0] == "EVAL"
    assert command_args[2] == 1
    assert command_args[3] == _INGESTION_LOCK_KEY
