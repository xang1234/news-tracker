"""Tests for ThemeRankingService — pure computation and async orchestrator."""

import math
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.themes.ranking import (
    LIFECYCLE_MULTIPLIERS,
    STRATEGY_CONFIGS,
    RankedTheme,
    RankingConfig,
    ThemeRankingService,
)
from src.themes.schemas import Theme, ThemeMetrics


# ── Helpers ──────────────────────────────────────────────


def _make_theme(
    theme_id: str = "theme_test",
    lifecycle_stage: str = "emerging",
    document_count: int = 42,
    metadata: dict | None = None,
) -> Theme:
    """Create a Theme with sensible defaults for ranking tests."""
    return Theme(
        theme_id=theme_id,
        name=f"theme_{theme_id}",
        centroid=np.zeros(768, dtype=np.float32),
        lifecycle_stage=lifecycle_stage,
        document_count=document_count,
        metadata=metadata or {},
    )


def _make_metrics(
    theme_id: str = "theme_test",
    volume_zscore: float | None = 1.0,
    target_date: date | None = None,
) -> ThemeMetrics:
    """Create ThemeMetrics with controllable z-score."""
    return ThemeMetrics(
        theme_id=theme_id,
        date=target_date or date(2026, 2, 7),
        document_count=10,
        volume_zscore=volume_zscore,
    )


@pytest.fixture
def service() -> ThemeRankingService:
    """ThemeRankingService with default config."""
    return ThemeRankingService()


# ── TestComputeScore ─────────────────────────────────────


class TestComputeScore:
    """Tests for the core scoring formula."""

    def test_swing_strategy_weights(self, service: ThemeRankingService) -> None:
        """Swing strategy uses alpha=0.6, beta=0.4."""
        theme = _make_theme(metadata={"compellingness": 5.0})
        metrics = _make_metrics(volume_zscore=1.0)

        score, components = service.compute_score(theme, metrics, "swing")

        # volume_component = max(0, 1.0 + 2.0) ** 0.6 = 3.0 ** 0.6
        expected_vol = math.pow(3.0, 0.6)
        assert components["volume_component"] == pytest.approx(expected_vol, rel=1e-4)

        # compellingness = 5.0 ** 0.4
        expected_comp = math.pow(5.0, 0.4)
        assert components["compellingness_component"] == pytest.approx(expected_comp, rel=1e-4)

    def test_position_strategy_weights(self, service: ThemeRankingService) -> None:
        """Position strategy uses alpha=0.4, beta=0.6."""
        theme = _make_theme(metadata={"compellingness": 5.0})
        metrics = _make_metrics(volume_zscore=1.0)

        score, components = service.compute_score(theme, metrics, "position")

        expected_vol = math.pow(3.0, 0.4)
        expected_comp = math.pow(5.0, 0.6)
        assert components["volume_component"] == pytest.approx(expected_vol, rel=1e-4)
        assert components["compellingness_component"] == pytest.approx(expected_comp, rel=1e-4)

    def test_lifecycle_multipliers(self, service: ThemeRankingService) -> None:
        """Each lifecycle stage applies its own multiplier."""
        metrics = _make_metrics(volume_zscore=1.0)

        scores = {}
        for stage in ["emerging", "accelerating", "mature", "fading"]:
            theme = _make_theme(lifecycle_stage=stage)
            score, _ = service.compute_score(theme, metrics, "swing")
            scores[stage] = score

        # emerging (1.5) > accelerating (1.2) > mature (0.8) > fading (0.3)
        assert scores["emerging"] > scores["accelerating"]
        assert scores["accelerating"] > scores["mature"]
        assert scores["mature"] > scores["fading"]

    def test_missing_compellingness_uses_default(self, service: ThemeRankingService) -> None:
        """Theme without compellingness in metadata falls back to 5.0."""
        theme = _make_theme(metadata={})
        metrics = _make_metrics(volume_zscore=1.0)

        score, components = service.compute_score(theme, metrics, "swing")

        expected_comp = math.pow(5.0, 0.4)
        assert components["compellingness_component"] == pytest.approx(expected_comp, rel=1e-4)

    def test_zero_zscore(self, service: ThemeRankingService) -> None:
        """Z-score of 0 → shifted to 2.0, produces positive score."""
        theme = _make_theme()
        metrics = _make_metrics(volume_zscore=0.0)

        score, components = service.compute_score(theme, metrics, "swing")
        assert score > 0
        expected_vol = math.pow(2.0, 0.6)
        assert components["volume_component"] == pytest.approx(expected_vol, rel=1e-4)

    def test_negative_zscore(self, service: ThemeRankingService) -> None:
        """Negative z-score (but > -2) produces small but positive score."""
        theme = _make_theme()
        metrics = _make_metrics(volume_zscore=-1.0)

        score, _ = service.compute_score(theme, metrics, "swing")
        assert score > 0

    def test_very_negative_zscore_zeroes_volume(self, service: ThemeRankingService) -> None:
        """Z-score of -2.0 or below → shifted to 0 or below → volume_component = 0."""
        theme = _make_theme()
        metrics = _make_metrics(volume_zscore=-2.0)

        score, components = service.compute_score(theme, metrics, "swing")
        assert components["volume_component"] == 0.0
        assert score == 0.0

    def test_none_metrics(self, service: ThemeRankingService) -> None:
        """None metrics → z-score defaults to 0.0."""
        theme = _make_theme()
        score, components = service.compute_score(theme, None, "swing")

        assert score > 0
        assert components["volume_zscore"] == 0.0

    def test_high_zscore_produces_high_score(self, service: ThemeRankingService) -> None:
        """High z-score (surge) produces high score."""
        theme = _make_theme(lifecycle_stage="accelerating")
        metrics = _make_metrics(volume_zscore=4.0)

        score, _ = service.compute_score(theme, metrics, "swing")
        # 6.0**0.6 * 5.0**0.4 * 1.2 — should be substantial
        assert score > 5.0

    def test_score_components_match_total(self, service: ThemeRankingService) -> None:
        """Verify that components multiply to the total score."""
        theme = _make_theme(metadata={"compellingness": 7.0})
        metrics = _make_metrics(volume_zscore=2.5)

        score, components = service.compute_score(theme, metrics, "swing")

        reconstructed = (
            components["volume_component"]
            * components["compellingness_component"]
            * components["lifecycle_multiplier"]
        )
        assert score == pytest.approx(reconstructed, rel=1e-4)


# ── TestRankThemes ───────────────────────────────────────


class TestRankThemes:
    """Tests for sorting and filtering by score."""

    def test_empty_input(self, service: ThemeRankingService) -> None:
        assert service.rank_themes([], {}, "swing") == []

    def test_single_theme(self, service: ThemeRankingService) -> None:
        theme = _make_theme()
        metrics = _make_metrics(volume_zscore=1.0)
        result = service.rank_themes([theme], {theme.theme_id: metrics}, "swing")

        assert len(result) == 1
        assert result[0].theme_id == theme.theme_id
        assert result[0].score > 0

    def test_sorted_descending(self, service: ThemeRankingService) -> None:
        """Higher z-score themes rank first."""
        themes = [
            _make_theme("low", lifecycle_stage="mature"),
            _make_theme("mid", lifecycle_stage="mature"),
            _make_theme("high", lifecycle_stage="mature"),
        ]
        metrics_map = {
            "low": _make_metrics("low", volume_zscore=0.0),
            "mid": _make_metrics("mid", volume_zscore=1.0),
            "high": _make_metrics("high", volume_zscore=3.0),
        }

        result = service.rank_themes(themes, metrics_map, "swing")

        assert result[0].theme_id == "high"
        assert result[1].theme_id == "mid"
        assert result[2].theme_id == "low"

    def test_mixed_lifecycles(self, service: ThemeRankingService) -> None:
        """Lifecycle multiplier affects ranking order."""
        themes = [
            _make_theme("fading", lifecycle_stage="fading"),
            _make_theme("emerging", lifecycle_stage="emerging"),
        ]
        # Same z-score, so lifecycle multiplier determines order
        metrics_map = {
            "fading": _make_metrics("fading", volume_zscore=1.0),
            "emerging": _make_metrics("emerging", volume_zscore=1.0),
        }

        result = service.rank_themes(themes, metrics_map, "swing")

        assert result[0].theme_id == "emerging"
        assert result[-1].theme_id == "fading"

    def test_below_threshold_excluded(self) -> None:
        """Themes with score below min_score_threshold are filtered out."""
        config = RankingConfig(min_score_threshold=100.0)  # Impossibly high
        service = ThemeRankingService(config=config)

        theme = _make_theme()
        metrics = _make_metrics(volume_zscore=1.0)

        result = service.rank_themes([theme], {theme.theme_id: metrics}, "swing")
        assert len(result) == 0

    def test_missing_metrics_still_ranked(self, service: ThemeRankingService) -> None:
        """Themes without metrics use default z-score of 0.0."""
        theme = _make_theme()
        result = service.rank_themes([theme], {}, "swing")

        assert len(result) == 1
        assert result[0].components["volume_zscore"] == 0.0


# ── TestAssignTiers ──────────────────────────────────────


class TestAssignTiers:
    """Tests for tier assignment logic."""

    def test_single_theme_tier_1_with_zscore(self, service: ThemeRankingService) -> None:
        """A single theme in top 5% with high z-score gets Tier 1."""
        theme = _make_theme(lifecycle_stage="accelerating")
        metrics = _make_metrics(volume_zscore=3.0)

        result = service.rank_themes([theme], {theme.theme_id: metrics}, "swing")

        assert result[0].tier == 1

    def test_tier_1_requires_zscore_or_accelerating(self, service: ThemeRankingService) -> None:
        """Top percentile with low z-score and non-accelerating → Tier 2."""
        theme = _make_theme(lifecycle_stage="mature")
        metrics = _make_metrics(volume_zscore=0.5)  # Below tier_1_min_zscore (2.0)

        result = service.rank_themes([theme], {theme.theme_id: metrics}, "swing")

        assert result[0].tier == 2

    def test_tier_1_accelerating_bypasses_zscore(self, service: ThemeRankingService) -> None:
        """Accelerating theme in top percentile gets Tier 1 even with low z-score."""
        theme = _make_theme(lifecycle_stage="accelerating")
        metrics = _make_metrics(volume_zscore=0.5)

        result = service.rank_themes([theme], {theme.theme_id: metrics}, "swing")

        assert result[0].tier == 1

    def test_many_themes_tier_distribution(self) -> None:
        """With 100 themes, verify tier distribution matches percentile config."""
        service = ThemeRankingService()

        themes = []
        metrics_map = {}
        for i in range(100):
            tid = f"theme_{i:03d}"
            # Accelerating so tier 1 gate is satisfied
            t = _make_theme(tid, lifecycle_stage="accelerating")
            themes.append(t)
            # Spread z-scores so they all have different scores
            m = _make_metrics(tid, volume_zscore=float(i) * 0.1)
            metrics_map[tid] = m

        result = service.rank_themes(themes, metrics_map, "swing")

        tier_1 = [r for r in result if r.tier == 1]
        tier_2 = [r for r in result if r.tier == 2]
        tier_3 = [r for r in result if r.tier == 3]

        # ceil(100 * 0.05) = 5 tier-1, ceil(100 * 0.20) = 20 positions total for tier 2
        assert len(tier_1) == 5
        assert len(tier_2) == 15  # positions 6-20
        assert len(tier_3) == len(result) - 20

    def test_tier_1_high_zscore_non_accelerating(self, service: ThemeRankingService) -> None:
        """Top percentile with z-score >= 2.0 gets Tier 1 even if not accelerating."""
        theme = _make_theme(lifecycle_stage="emerging")
        metrics = _make_metrics(volume_zscore=2.5)

        result = service.rank_themes([theme], {theme.theme_id: metrics}, "swing")

        assert result[0].tier == 1


# ── TestGetActionable ────────────────────────────────────


class TestGetActionable:
    """Tests for the async orchestrator with mock repos."""

    @pytest.mark.asyncio
    async def test_basic_orchestration(self) -> None:
        """Fetches themes, gets metrics, returns ranked list."""
        themes = [
            _make_theme("t1", lifecycle_stage="accelerating"),
            _make_theme("t2", lifecycle_stage="fading"),
        ]
        metrics_t1 = _make_metrics("t1", volume_zscore=2.0)

        mock_repo = AsyncMock()
        mock_repo.get_all = AsyncMock(return_value=themes)
        mock_repo.get_metrics_range = AsyncMock(
            side_effect=lambda tid, start, end: (
                [metrics_t1] if tid == "t1" else []
            )
        )

        service = ThemeRankingService(theme_repo=mock_repo)
        result = await service.get_actionable(strategy="swing", max_tier=3)

        assert len(result) >= 1
        # t1 (accelerating, z=2.0) should rank above t2 (fading, no metrics)
        assert result[0].theme_id == "t1"

    @pytest.mark.asyncio
    async def test_tier_filtering(self) -> None:
        """max_tier filters out lower tiers."""
        # Single theme that will be tier 1
        theme = _make_theme("t1", lifecycle_stage="accelerating")
        metrics = _make_metrics("t1", volume_zscore=3.0)

        mock_repo = AsyncMock()
        mock_repo.get_all = AsyncMock(return_value=[theme])
        mock_repo.get_metrics_range = AsyncMock(return_value=[metrics])

        service = ThemeRankingService(theme_repo=mock_repo)

        # Tier 1 only
        result = await service.get_actionable(strategy="swing", max_tier=1)
        assert len(result) == 1
        assert result[0].tier == 1

    @pytest.mark.asyncio
    async def test_missing_repo_raises(self) -> None:
        """Calling get_actionable without theme_repo raises RuntimeError."""
        service = ThemeRankingService()
        with pytest.raises(RuntimeError, match="theme_repo is required"):
            await service.get_actionable()

    @pytest.mark.asyncio
    async def test_default_strategy_from_config(self) -> None:
        """Uses config default_strategy when strategy arg is None."""
        config = RankingConfig(default_strategy="position")
        theme = _make_theme()
        metrics = _make_metrics(volume_zscore=1.0)

        mock_repo = AsyncMock()
        mock_repo.get_all = AsyncMock(return_value=[theme])
        mock_repo.get_metrics_range = AsyncMock(return_value=[metrics])

        service = ThemeRankingService(config=config, theme_repo=mock_repo)
        result = await service.get_actionable()

        assert len(result) == 1
        assert result[0].components["strategy"] == "position"

    @pytest.mark.asyncio
    async def test_empty_themes(self) -> None:
        """No themes → empty result."""
        mock_repo = AsyncMock()
        mock_repo.get_all = AsyncMock(return_value=[])

        service = ThemeRankingService(theme_repo=mock_repo)
        result = await service.get_actionable()

        assert result == []


# ── TestRankingConfig ────────────────────────────────────


class TestRankingConfig:
    """Tests for config defaults and env var overrides."""

    def test_defaults(self) -> None:
        config = RankingConfig()
        assert config.default_strategy == "swing"
        assert config.tier_1_percentile == 0.05
        assert config.tier_2_percentile == 0.20
        assert config.tier_1_min_zscore == 2.0
        assert config.min_score_threshold == 0.1
        assert config.default_compellingness == 5.0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANKING_DEFAULT_STRATEGY", "position")
        monkeypatch.setenv("RANKING_TIER_1_PERCENTILE", "0.10")
        config = RankingConfig()
        assert config.default_strategy == "position"
        assert config.tier_1_percentile == 0.10

    def test_custom_config_used_by_service(self) -> None:
        config = RankingConfig(
            default_compellingness=10.0,
            min_score_threshold=0.0,
        )
        service = ThemeRankingService(config=config)

        theme = _make_theme(metadata={})  # No compellingness → uses default
        metrics = _make_metrics(volume_zscore=1.0)

        _, components = service.compute_score(theme, metrics, "swing")
        # Should use 10.0 instead of 5.0
        expected_comp = math.pow(10.0, 0.4)
        assert components["compellingness_component"] == pytest.approx(expected_comp, rel=1e-4)
