"""Tests for VolumeMetricsService — pure computation and async orchestrator."""

import math
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.themes.metrics import (
    DEFAULT_PLATFORM_WEIGHTS,
    VolumeMetricsConfig,
    VolumeMetricsService,
)
from src.themes.schemas import ThemeMetrics


# ── Helpers ──────────────────────────────────────────────────


def _make_doc(
    timestamp: datetime,
    platform: str = "twitter",
    authority_score: float = 0.5,
) -> SimpleNamespace:
    """Create a lightweight mock document with required attributes."""
    return SimpleNamespace(
        timestamp=timestamp,
        platform=platform,
        authority_score=authority_score,
    )


@pytest.fixture
def service() -> VolumeMetricsService:
    """VolumeMetricsService with default config."""
    return VolumeMetricsService()


@pytest.fixture
def ref_time() -> datetime:
    """Fixed reference time for deterministic tests."""
    return datetime(2026, 2, 7, 12, 0, 0, tzinfo=timezone.utc)


# ── TestComputeWeightedVolume ────────────────────────────────


class TestComputeWeightedVolume:
    """Tests for platform weighting, recency decay, and authority scaling."""

    def test_empty_docs_returns_zero(self, service: VolumeMetricsService) -> None:
        assert service.compute_weighted_volume([]) == 0.0

    def test_single_twitter_doc(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """A single Twitter doc at reference time → weight * 1.0 * authority."""
        doc = _make_doc(ref_time, platform="twitter", authority_score=1.0)
        result = service.compute_weighted_volume([doc], ref_time)
        # recency_decay at age=0 → exp(0) = 1.0
        expected = DEFAULT_PLATFORM_WEIGHTS["twitter"] * 1.0 * 1.0
        assert result == pytest.approx(expected)

    def test_platform_weights_applied(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """News doc contributes more than Twitter doc (same age & authority)."""
        twitter_doc = _make_doc(ref_time, platform="twitter", authority_score=1.0)
        news_doc = _make_doc(ref_time, platform="news", authority_score=1.0)

        twitter_vol = service.compute_weighted_volume([twitter_doc], ref_time)
        news_vol = service.compute_weighted_volume([news_doc], ref_time)

        assert news_vol > twitter_vol
        assert news_vol / twitter_vol == pytest.approx(
            DEFAULT_PLATFORM_WEIGHTS["news"] / DEFAULT_PLATFORM_WEIGHTS["twitter"]
        )

    def test_recency_decay(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """Older documents contribute less than recent ones."""
        recent = _make_doc(ref_time, authority_score=1.0)
        old = _make_doc(
            ref_time - timedelta(days=3), authority_score=1.0
        )

        vol_recent = service.compute_weighted_volume([recent], ref_time)
        vol_old = service.compute_weighted_volume([old], ref_time)

        assert vol_recent > vol_old

    def test_authority_scaling(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """Higher authority → higher contribution."""
        low = _make_doc(ref_time, authority_score=0.2)
        high = _make_doc(ref_time, authority_score=0.8)

        vol_low = service.compute_weighted_volume([low], ref_time)
        vol_high = service.compute_weighted_volume([high], ref_time)

        assert vol_high > vol_low

    def test_docs_outside_window_excluded(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """Documents older than window_days are excluded."""
        # Default window is 7 days; doc at 8 days ago should be excluded
        old_doc = _make_doc(
            ref_time - timedelta(days=8), authority_score=1.0
        )
        assert service.compute_weighted_volume([old_doc], ref_time) == 0.0

    def test_doc_at_window_boundary_included(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """Document at exactly window_days edge is included (> cutoff check)."""
        # 6 days 23 hours is within window
        doc = _make_doc(
            ref_time - timedelta(days=6, hours=23),
            authority_score=1.0,
        )
        assert service.compute_weighted_volume([doc], ref_time) > 0.0

    def test_none_authority_uses_floor(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """Doc with None authority_score still contributes (floor of 0.1)."""
        doc = SimpleNamespace(
            timestamp=ref_time, platform="twitter", authority_score=None
        )
        vol = service.compute_weighted_volume([doc], ref_time)
        expected = DEFAULT_PLATFORM_WEIGHTS["twitter"] * 1.0 * 0.1
        assert vol == pytest.approx(expected)

    def test_unknown_platform_defaults_to_weight_1(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """Unknown platform falls back to weight 1.0."""
        doc = _make_doc(ref_time, platform="tiktok", authority_score=1.0)
        vol = service.compute_weighted_volume([doc], ref_time)
        assert vol == pytest.approx(1.0)

    def test_naive_timestamp_treated_as_utc(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """Naive timestamps are treated as UTC."""
        naive = datetime(2026, 2, 7, 12, 0, 0)  # No tzinfo
        doc = _make_doc(naive, authority_score=1.0)
        vol = service.compute_weighted_volume([doc], ref_time)
        assert vol > 0.0

    def test_none_timestamp_skipped(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """Docs with None timestamp are silently skipped."""
        doc = SimpleNamespace(
            timestamp=None, platform="twitter", authority_score=0.5
        )
        assert service.compute_weighted_volume([doc], ref_time) == 0.0

    def test_multiple_docs_additive(
        self, service: VolumeMetricsService, ref_time: datetime
    ) -> None:
        """Volume from multiple docs is the sum of individual contributions."""
        d1 = _make_doc(ref_time, platform="twitter", authority_score=1.0)
        d2 = _make_doc(ref_time, platform="twitter", authority_score=1.0)

        vol_single = service.compute_weighted_volume([d1], ref_time)
        vol_double = service.compute_weighted_volume([d1, d2], ref_time)

        assert vol_double == pytest.approx(2 * vol_single)


# ── TestComputeVolumeZscore ──────────────────────────────────


class TestComputeVolumeZscore:
    """Tests for z-score normalization."""

    def test_insufficient_history_returns_none(
        self, service: VolumeMetricsService
    ) -> None:
        """Fewer than min_history_days → None."""
        result = service.compute_volume_zscore(5.0, [1.0, 2.0, 3.0])
        assert result is None

    def test_exactly_min_history(self, service: VolumeMetricsService) -> None:
        """Exactly min_history_days of data → computes z-score."""
        history = [10.0] * 7  # mean=10, std=0
        result = service.compute_volume_zscore(10.0, history)
        assert result is not None
        assert result == 0.0  # current equals mean, std=0 → return 0.0

    def test_positive_zscore(self, service: VolumeMetricsService) -> None:
        """Current above mean → positive z-score."""
        history = [10.0] * 10
        history[0] = 8.0  # Introduce some variance
        result = service.compute_volume_zscore(15.0, history)
        assert result is not None
        assert result > 0

    def test_negative_zscore(self, service: VolumeMetricsService) -> None:
        """Current below mean → negative z-score."""
        history = [10.0] * 10
        history[0] = 12.0
        result = service.compute_volume_zscore(5.0, history)
        assert result is not None
        assert result < 0

    def test_zero_std_returns_zero(self, service: VolumeMetricsService) -> None:
        """All same values (std=0) → z-score is 0.0."""
        history = [5.0] * 10
        result = service.compute_volume_zscore(5.0, history)
        assert result == 0.0

    def test_known_zscore(self, service: VolumeMetricsService) -> None:
        """Verify exact z-score computation."""
        # mean=10, population std = sqrt(((8-10)^2 + (12-10)^2 + 10*0)/12)... simpler:
        history = [8.0, 12.0, 8.0, 12.0, 8.0, 12.0, 8.0]
        mean = sum(history) / len(history)
        variance = sum((v - mean) ** 2 for v in history) / len(history)
        std = math.sqrt(variance)
        current = 14.0
        expected = (current - mean) / std

        result = service.compute_volume_zscore(current, history)
        assert result == pytest.approx(expected)


# ── TestComputeVelocity ──────────────────────────────────────


class TestComputeVelocity:
    """Tests for EMA-based velocity computation."""

    def test_insufficient_data_returns_none(
        self, service: VolumeMetricsService
    ) -> None:
        """Fewer than ema_long_span values → None."""
        result = service.compute_velocity([1.0, 2.0])
        assert result is None

    def test_exactly_long_span(self, service: VolumeMetricsService) -> None:
        """Exactly ema_long_span values → computes velocity."""
        zscores = [1.0] * 7  # Default long_span = 7
        result = service.compute_velocity(zscores)
        assert result is not None
        # Short and long EMA converge on constant input → velocity ≈ 0
        assert result == pytest.approx(0.0, abs=0.01)

    def test_rising_zscores_positive_velocity(
        self, service: VolumeMetricsService
    ) -> None:
        """Increasing z-scores → positive velocity (short EMA > long EMA)."""
        zscores = [float(i) for i in range(10)]
        result = service.compute_velocity(zscores)
        assert result is not None
        assert result > 0

    def test_falling_zscores_negative_velocity(
        self, service: VolumeMetricsService
    ) -> None:
        """Decreasing z-scores → negative velocity."""
        zscores = [float(10 - i) for i in range(10)]
        result = service.compute_velocity(zscores)
        assert result is not None
        assert result < 0


# ── TestComputeAcceleration ──────────────────────────────────


class TestComputeAcceleration:
    """Tests for acceleration (velocity delta)."""

    def test_insufficient_data_returns_none(
        self, service: VolumeMetricsService
    ) -> None:
        assert service.compute_acceleration([1.0]) is None
        assert service.compute_acceleration([]) is None

    def test_positive_acceleration(
        self, service: VolumeMetricsService
    ) -> None:
        result = service.compute_acceleration([1.0, 3.0])
        assert result == pytest.approx(2.0)

    def test_negative_acceleration(
        self, service: VolumeMetricsService
    ) -> None:
        result = service.compute_acceleration([3.0, 1.0])
        assert result == pytest.approx(-2.0)

    def test_zero_acceleration(self, service: VolumeMetricsService) -> None:
        result = service.compute_acceleration([2.0, 2.0])
        assert result == pytest.approx(0.0)

    def test_uses_last_two_only(self, service: VolumeMetricsService) -> None:
        """Only the last two values matter."""
        result = service.compute_acceleration([100.0, 0.0, 5.0, 10.0])
        assert result == pytest.approx(5.0)


# ── TestDetectVolumeAnomaly ──────────────────────────────────


class TestDetectVolumeAnomaly:
    """Tests for surge/collapse anomaly detection."""

    def test_surge(self, service: VolumeMetricsService) -> None:
        assert service.detect_volume_anomaly(3.5) == "surge"

    def test_exact_surge_threshold(
        self, service: VolumeMetricsService
    ) -> None:
        assert service.detect_volume_anomaly(3.0) == "surge"

    def test_collapse(self, service: VolumeMetricsService) -> None:
        assert service.detect_volume_anomaly(-2.5) == "collapse"

    def test_exact_collapse_threshold(
        self, service: VolumeMetricsService
    ) -> None:
        assert service.detect_volume_anomaly(-2.0) == "collapse"

    def test_normal_range(self, service: VolumeMetricsService) -> None:
        assert service.detect_volume_anomaly(1.5) is None
        assert service.detect_volume_anomaly(-1.0) is None
        assert service.detect_volume_anomaly(0.0) is None

    def test_none_zscore(self, service: VolumeMetricsService) -> None:
        assert service.detect_volume_anomaly(None) is None


# ── TestEma ──────────────────────────────────────────────────


class TestEma:
    """Tests for the static _ema helper."""

    def test_empty_returns_zero(self) -> None:
        assert VolumeMetricsService._ema([], 3) == 0.0

    def test_single_value(self) -> None:
        assert VolumeMetricsService._ema([5.0], 3) == 5.0

    def test_constant_values(self) -> None:
        """EMA of constant series equals the constant."""
        result = VolumeMetricsService._ema([7.0, 7.0, 7.0, 7.0], 3)
        assert result == pytest.approx(7.0)

    def test_increasing_values(self) -> None:
        """EMA is pulled toward recent values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = VolumeMetricsService._ema(values, 3)
        # EMA should be between mean and last value
        assert result > sum(values) / len(values)
        assert result < values[-1]


# ── TestComputeForTheme ──────────────────────────────────────


class TestComputeForTheme:
    """Tests for the async orchestrator with mock repositories."""

    @pytest.mark.asyncio
    async def test_basic_orchestration(self) -> None:
        """Orchestrator fetches data, calls computations, returns ThemeMetrics."""
        ref_date = date(2026, 2, 7)

        # Docs assigned to this theme
        docs = [
            _make_doc(
                datetime(2026, 2, 7, 10, 0, tzinfo=timezone.utc),
                platform="news",
                authority_score=0.8,
            ),
            _make_doc(
                datetime(2026, 2, 6, 14, 0, tzinfo=timezone.utc),
                platform="twitter",
                authority_score=0.5,
            ),
        ]

        # Historical metrics with weighted_volume for z-score
        history = [
            ThemeMetrics(
                theme_id="theme_test",
                date=ref_date - timedelta(days=i),
                document_count=10,
                weighted_volume=10.0 + i * 0.5,
                volume_zscore=0.1 * i,
                velocity=0.05 * i,
            )
            for i in range(1, 15)
        ]

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_documents_by_theme = AsyncMock(return_value=docs)

        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_metrics_range = AsyncMock(return_value=history)

        svc = VolumeMetricsService(
            doc_repo=mock_doc_repo, theme_repo=mock_theme_repo
        )

        result = await svc.compute_for_theme("theme_test", ref_date)

        assert isinstance(result, ThemeMetrics)
        assert result.theme_id == "theme_test"
        assert result.date == ref_date
        assert result.document_count == 2
        assert result.weighted_volume is not None
        assert result.weighted_volume > 0

        # Verify repos were called
        mock_doc_repo.get_documents_by_theme.assert_awaited_once_with(
            "theme_test", limit=500
        )
        mock_theme_repo.get_metrics_range.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_history_returns_none_zscore(self) -> None:
        """No historical metrics → z-score/velocity/acceleration are None."""
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_documents_by_theme = AsyncMock(return_value=[])

        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_metrics_range = AsyncMock(return_value=[])

        svc = VolumeMetricsService(
            doc_repo=mock_doc_repo, theme_repo=mock_theme_repo
        )

        result = await svc.compute_for_theme("theme_test", date(2026, 2, 7))

        assert result.weighted_volume == 0.0
        assert result.volume_zscore is None
        assert result.velocity is None
        assert result.acceleration is None

    @pytest.mark.asyncio
    async def test_missing_repos_raises(self) -> None:
        """Calling compute_for_theme without repos raises RuntimeError."""
        svc = VolumeMetricsService()
        with pytest.raises(RuntimeError, match="doc_repo and theme_repo"):
            await svc.compute_for_theme("theme_test", date(2026, 2, 7))


# ── TestVolumeMetricsConfig ──────────────────────────────────


class TestVolumeMetricsConfig:
    """Tests for config defaults and env var overrides."""

    def test_defaults(self) -> None:
        config = VolumeMetricsConfig()
        assert config.decay_factor == 0.3
        assert config.window_days == 7
        assert config.history_window == 30
        assert config.min_history_days == 7
        assert config.surge_threshold == 3.0
        assert config.collapse_threshold == -2.0
        assert config.ema_short_span == 3
        assert config.ema_long_span == 7

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VOLUME_DECAY_FACTOR", "0.5")
        monkeypatch.setenv("VOLUME_SURGE_THRESHOLD", "4.0")
        config = VolumeMetricsConfig()
        assert config.decay_factor == 0.5
        assert config.surge_threshold == 4.0

    def test_custom_config_used_by_service(self) -> None:
        config = VolumeMetricsConfig(surge_threshold=5.0, collapse_threshold=-3.0)
        svc = VolumeMetricsService(config=config)
        assert svc.detect_volume_anomaly(4.0) is None  # Below custom threshold
        assert svc.detect_volume_anomaly(5.0) == "surge"
        assert svc.detect_volume_anomaly(-3.0) == "collapse"
