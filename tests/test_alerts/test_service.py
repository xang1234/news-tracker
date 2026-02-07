"""Tests for AlertService with mocked Redis and repository."""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from src.alerts.config import AlertConfig
from src.alerts.repository import AlertRepository
from src.alerts.schemas import Alert
from src.alerts.service import AlertService
from src.themes.schemas import Theme, ThemeMetrics
from src.themes.transitions import LifecycleTransition


@pytest.fixture
def config():
    return AlertConfig()


@pytest.fixture
def mock_repo():
    repo = AsyncMock(spec=AlertRepository)
    repo.create_batch.return_value = []
    repo.count_today_by_severity.return_value = 0
    return repo


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.set.return_value = True  # SET NX succeeds (no duplicate)
    return r


@pytest.fixture
def service(config, mock_repo, mock_redis):
    return AlertService(
        config=config,
        alert_repo=mock_repo,
        redis_client=mock_redis,
    )


@pytest.fixture
def theme():
    return Theme(
        theme_id="theme_svc_test",
        name="service_test_theme",
        centroid=np.zeros(768, dtype=np.float32),
        lifecycle_stage="accelerating",
        document_count=100,
    )


def _make_metrics(
    theme_id: str = "theme_svc_test",
    target_date: date = date(2026, 2, 7),
    **kwargs,
) -> ThemeMetrics:
    defaults = {
        "document_count": 10,
        "sentiment_score": 0.5,
        "bullish_ratio": 0.6,
        "volume_zscore": 1.0,
    }
    defaults.update(kwargs)
    return ThemeMetrics(theme_id=theme_id, date=target_date, **defaults)


# ── Deduplication ────────────────────────────────────────


class TestDeduplication:
    """Test Redis SET NX dedup behavior."""

    @pytest.mark.asyncio
    async def test_not_duplicate_when_key_set(self, service, mock_redis):
        mock_redis.set.return_value = True  # SET NX succeeded
        assert await service._is_duplicate("t1", "volume_surge") is False

    @pytest.mark.asyncio
    async def test_duplicate_when_key_exists(self, service, mock_redis):
        mock_redis.set.return_value = None  # SET NX failed (key exists)
        assert await service._is_duplicate("t1", "volume_surge") is True

    @pytest.mark.asyncio
    async def test_graceful_on_redis_error(self, service, mock_redis):
        mock_redis.set.side_effect = ConnectionError("Redis down")
        assert await service._is_duplicate("t1", "volume_surge") is False

    @pytest.mark.asyncio
    async def test_no_redis_client(self, config, mock_repo):
        svc = AlertService(config=config, alert_repo=mock_repo, redis_client=None)
        assert await svc._is_duplicate("t1", "volume_surge") is False


# ── Rate Limiting ────────────────────────────────────────


class TestRateLimiting:
    """Test DB count rate limiting."""

    @pytest.mark.asyncio
    async def test_not_limited_below_threshold(self, service, mock_repo):
        mock_repo.count_today_by_severity.return_value = 3
        assert await service._is_rate_limited("critical") is False  # limit=5

    @pytest.mark.asyncio
    async def test_limited_at_threshold(self, service, mock_repo):
        mock_repo.count_today_by_severity.return_value = 5
        assert await service._is_rate_limited("critical") is True  # limit=5

    @pytest.mark.asyncio
    async def test_unlimited_info(self, service, mock_repo):
        mock_repo.count_today_by_severity.return_value = 1000
        assert await service._is_rate_limited("info") is False  # limit=0

    @pytest.mark.asyncio
    async def test_graceful_on_db_error(self, service, mock_repo):
        mock_repo.count_today_by_severity.side_effect = Exception("DB error")
        assert await service._is_rate_limited("critical") is False


# ── generate_alerts ──────────────────────────────────────


class TestGenerateAlerts:
    """Test end-to-end alert generation."""

    @pytest.mark.asyncio
    async def test_no_alerts_when_no_metrics(self, service, mock_repo, theme):
        mock_repo.create_batch.return_value = []
        result = await service.generate_alerts(
            themes=[theme],
            today_metrics_map={},
            yesterday_metrics_map={},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_volume_surge_alert_generated(self, service, mock_repo, theme):
        today = _make_metrics(volume_zscore=4.5)
        mock_repo.create_batch.side_effect = lambda alerts: alerts

        result = await service.generate_alerts(
            themes=[theme],
            today_metrics_map={theme.theme_id: today},
            yesterday_metrics_map={},
        )
        assert len(result) == 1
        assert result[0].trigger_type == "volume_surge"
        assert result[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_lifecycle_alert_generated(self, service, mock_repo, theme):
        today = _make_metrics()
        transition = LifecycleTransition(
            theme_id=theme.theme_id,
            from_stage="emerging",
            to_stage="accelerating",
            confidence=0.8,
        )
        mock_repo.create_batch.side_effect = lambda alerts: alerts

        result = await service.generate_alerts(
            themes=[theme],
            today_metrics_map={theme.theme_id: today},
            yesterday_metrics_map={},
            lifecycle_transitions=[transition],
        )
        types = {a.trigger_type for a in result}
        assert "lifecycle_change" in types

    @pytest.mark.asyncio
    async def test_new_theme_alert_generated(self, service, mock_repo, theme):
        mock_repo.create_batch.side_effect = lambda alerts: alerts

        result = await service.generate_alerts(
            themes=[theme],
            today_metrics_map={},
            yesterday_metrics_map={},
            new_theme_ids=[theme.theme_id],
        )
        assert len(result) == 1
        assert result[0].trigger_type == "new_theme"
        assert result[0].severity == "info"

    @pytest.mark.asyncio
    async def test_dedup_filters_duplicates(self, service, mock_repo, mock_redis, theme):
        today = _make_metrics(volume_zscore=4.5)
        mock_redis.set.return_value = None  # duplicate
        mock_repo.create_batch.side_effect = lambda alerts: alerts

        result = await service.generate_alerts(
            themes=[theme],
            today_metrics_map={theme.theme_id: today},
            yesterday_metrics_map={},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_rate_limit_filters_alerts(self, service, mock_repo, theme):
        today = _make_metrics(volume_zscore=4.5)
        mock_repo.count_today_by_severity.return_value = 100
        mock_repo.create_batch.side_effect = lambda alerts: alerts

        result = await service.generate_alerts(
            themes=[theme],
            today_metrics_map={theme.theme_id: today},
            yesterday_metrics_map={},
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_themes(self, service, mock_repo):
        themes = [
            Theme(
                theme_id=f"theme_{i}",
                name=f"theme_{i}",
                centroid=np.zeros(768, dtype=np.float32),
            )
            for i in range(3)
        ]
        today_map = {
            t.theme_id: _make_metrics(
                theme_id=t.theme_id, volume_zscore=3.5,
            )
            for t in themes
        }
        mock_repo.create_batch.side_effect = lambda alerts: alerts

        result = await service.generate_alerts(
            themes=themes,
            today_metrics_map=today_map,
            yesterday_metrics_map={},
        )
        assert len(result) == 3
        assert all(a.trigger_type == "volume_surge" for a in result)
