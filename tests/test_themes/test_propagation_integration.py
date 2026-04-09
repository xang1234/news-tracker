"""Tests that propagation_impact metadata flows into ranking scores.

Validates that the ranking formula's (1 + propagation_bonus) term
correctly amplifies scores for themes with causal graph impact.
"""

from __future__ import annotations

import pytest

from src.themes.ranking import RankingConfig, ThemeRankingService
from src.themes.schemas import Theme, ThemeMetrics


def _make_theme(
    theme_id: str = "theme_test",
    lifecycle_stage: str = "accelerating",
    metadata: dict | None = None,
) -> Theme:
    return Theme(
        theme_id=theme_id,
        name="Test Theme",
        description="A test theme",
        centroid=[0.0] * 768,
        top_keywords=["test"],
        top_tickers=["NVDA"],
        lifecycle_stage=lifecycle_stage,
        metadata=metadata or {},
    )


def _make_metrics(
    volume_zscore: float = 2.0,
) -> ThemeMetrics:
    from datetime import date

    return ThemeMetrics(
        theme_id="theme_test",
        date=date.today(),
        document_count=10,
        volume_zscore=volume_zscore,
    )


class TestPropagationBonus:
    def test_no_propagation_metadata_gives_zero_bonus(self):
        service = ThemeRankingService()
        theme = _make_theme(metadata={})
        metrics = _make_metrics()

        score, components = service.compute_score(theme, metrics)

        assert components["propagation_bonus"] == 0.0

    def test_propagation_impact_boosts_score(self):
        service = ThemeRankingService()

        theme_without = _make_theme(metadata={})
        theme_with = _make_theme(metadata={"propagation_impact": 0.3})
        metrics = _make_metrics()

        score_without, _ = service.compute_score(theme_without, metrics)
        score_with, comp = service.compute_score(theme_with, metrics)

        assert score_with > score_without
        assert comp["propagation_bonus"] == 0.3

    def test_propagation_bonus_capped_at_half(self):
        service = ThemeRankingService()
        theme = _make_theme(metadata={"propagation_impact": 1.5})
        metrics = _make_metrics()

        _, components = service.compute_score(theme, metrics)

        assert components["propagation_bonus"] == 0.5

    def test_negative_propagation_clamped_to_zero(self):
        service = ThemeRankingService()
        theme = _make_theme(metadata={"propagation_impact": -0.5})
        metrics = _make_metrics()

        _, components = service.compute_score(theme, metrics)

        assert components["propagation_bonus"] == 0.0

    def test_propagation_bonus_multiplier_effect(self):
        """Verify score = base * (1 + bonus) mathematically."""
        service = ThemeRankingService()
        metrics = _make_metrics()

        # Score with no bonus
        theme_0 = _make_theme(metadata={"propagation_impact": 0.0})
        base_score, _ = service.compute_score(theme_0, metrics)

        # Score with 0.2 bonus
        theme_02 = _make_theme(metadata={"propagation_impact": 0.2})
        boosted_score, _ = service.compute_score(theme_02, metrics)

        # The ratio should be 1.2 / 1.0 = 1.2
        assert abs(boosted_score / base_score - 1.2) < 0.001

    def test_ranking_order_reflects_propagation(self):
        """Higher propagation impact should rank higher."""
        service = ThemeRankingService()

        theme_low = _make_theme(theme_id="low", metadata={"propagation_impact": 0.0})
        theme_high = _make_theme(theme_id="high", metadata={"propagation_impact": 0.4})

        metrics_map = {
            "low": _make_metrics(),
            "high": _make_metrics(),
        }
        # Hack: point metrics at the right theme_ids
        metrics_map["low"] = ThemeMetrics(
            theme_id="low", date=metrics_map["low"].date,
            document_count=10, volume_zscore=2.0,
        )
        metrics_map["high"] = ThemeMetrics(
            theme_id="high", date=metrics_map["high"].date,
            document_count=10, volume_zscore=2.0,
        )

        ranked = service.rank_themes(
            [theme_low, theme_high], metrics_map, strategy="swing",
        )

        assert ranked[0].theme_id == "high"
        assert ranked[1].theme_id == "low"
