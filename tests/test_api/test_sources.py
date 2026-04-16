"""Unit tests for sources admin routes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.api.routes.sources import _INGESTION_LOCK_KEY, trigger_ingestion
from src.ingestion.schemas import Platform
from src.services.ingestion_service import IngestionConfigurationError


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


class _FakeIngestionService:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def run_once(self) -> dict[Platform, int]:
        return {Platform.TWITTER: 1}


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
