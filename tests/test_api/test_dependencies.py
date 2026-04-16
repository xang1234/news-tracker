"""Tests for app-scoped API dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.api.dependencies import AppServices


@pytest.mark.asyncio
async def test_get_database_does_not_cache_failed_connection(monkeypatch) -> None:
    attempts = 0

    class FakeDatabase:
        def __init__(self) -> None:
            self.closed = False

        async def connect(self) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("boom")

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("src.api.dependencies.Database", FakeDatabase)

    services = AppServices()

    with pytest.raises(RuntimeError, match="boom"):
        await services.get_database()

    assert services.database is None

    database = await services.get_database()

    assert isinstance(database, FakeDatabase)
    assert attempts == 2


@pytest.mark.asyncio
async def test_close_attempts_all_resources_even_if_one_fails() -> None:
    services = AppServices()

    failing_broadcaster = type(
        "Broadcaster",
        (),
        {"stop": AsyncMock(side_effect=RuntimeError("stop failed"))},
    )()
    closing_embedding = type("Embedding", (), {"close": AsyncMock()})()
    closing_sentiment = type(
        "Sentiment",
        (),
        {"close": AsyncMock(side_effect=RuntimeError("close failed"))},
    )()
    closing_redis = type("Redis", (), {"close": AsyncMock()})()
    closing_database = type("Database", (), {"close": AsyncMock()})()

    services.alert_broadcaster = failing_broadcaster
    services.embedding_service = closing_embedding
    services.sentiment_service = closing_sentiment
    services.redis_client = closing_redis
    services.database = closing_database

    await services.close()

    failing_broadcaster.stop.assert_awaited_once()
    closing_embedding.close.assert_awaited_once()
    closing_sentiment.close.assert_awaited_once()
    closing_redis.close.assert_awaited_once()
    closing_database.close.assert_awaited_once()
    assert services.alert_broadcaster is None
    assert services.embedding_service is None
    assert services.sentiment_service is None
    assert services.redis_client is None
    assert services.database is None
