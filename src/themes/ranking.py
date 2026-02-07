"""Theme ranking engine for actionability-based theme scoring.

Ranks themes by trading actionability using a multiplicative scoring model:
  score = (volume_component ** alpha) * (compellingness ** beta) * lifecycle_multiplier

Strategy-specific weights control the balance between volume momentum
(swing trading) and narrative quality (position trading).  All scoring
methods are stateless and side-effect-free, following the
LifecycleClassifier and VolumeMetricsService patterns.

Components:
- RankingConfig: Pydantic settings (RANKING_* env vars)
- RankedTheme: Scored theme with tier assignment
- ThemeRankingService: Stateless scoring + async orchestrator
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.themes.schemas import Theme, ThemeMetrics

logger = logging.getLogger(__name__)

# ── Types ────────────────────────────────────────────────

RankingStrategy = Literal["swing", "position"]

# ── Constants ────────────────────────────────────────────

STRATEGY_CONFIGS: dict[str, dict[str, float]] = {
    "swing": {"alpha": 0.6, "beta": 0.4},
    "position": {"alpha": 0.4, "beta": 0.6},
}

LIFECYCLE_MULTIPLIERS: dict[str, float] = {
    "emerging": 1.5,
    "accelerating": 1.2,
    "mature": 0.8,
    "fading": 0.3,
}


# ── Schemas ──────────────────────────────────────────────


@dataclass
class RankedTheme:
    """A theme with its computed ranking score and tier assignment.

    Attributes:
        theme_id: Deterministic theme identifier.
        theme: Reference to the underlying Theme record.
        score: Composite ranking score (higher = more actionable).
        tier: 1 (top 5%), 2 (top 20%), or 3 (rest).
        components: Breakdown of score factors for explainability.
    """

    theme_id: str
    theme: Theme
    score: float
    tier: int = 3
    components: dict[str, float] = field(default_factory=dict)


# ── Config ───────────────────────────────────────────────


class RankingConfig(BaseSettings):
    """Configuration for the theme ranking engine.

    All settings can be overridden via ``RANKING_*`` environment variables
    (e.g., ``RANKING_DEFAULT_STRATEGY=position``).
    """

    model_config = SettingsConfigDict(
        env_prefix="RANKING_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    default_strategy: str = "swing"
    """Default ranking strategy when not specified by caller."""

    tier_1_percentile: float = 0.05
    """Top fraction of themes for Tier 1 (e.g., 0.05 = top 5%)."""

    tier_2_percentile: float = 0.20
    """Top fraction of themes for Tier 2 (e.g., 0.20 = top 20%)."""

    tier_1_min_zscore: float = 2.0
    """Minimum volume z-score required for Tier 1 assignment."""

    min_score_threshold: float = 0.1
    """Minimum score to be included in ranked output."""

    default_compellingness: float = 5.0
    """Fallback compellingness when theme metadata is missing."""


# ── Service ──────────────────────────────────────────────


class ThemeRankingService:
    """Ranks themes by trading actionability.

    Pure computation methods (no DB, stateless):
      - ``compute_score`` — multiplicative score from volume, compellingness, lifecycle
      - ``rank_themes`` — score all themes, sort, assign tiers

    Async orchestrator:
      - ``get_actionable`` — fetches themes + metrics from repos, ranks, filters
    """

    def __init__(
        self,
        config: RankingConfig | None = None,
        theme_repo: Any = None,
    ) -> None:
        self._config = config or RankingConfig()
        self._theme_repo = theme_repo

    # ── Pure computation methods ─────────────────────────

    def compute_score(
        self,
        theme: Theme,
        metrics: ThemeMetrics | None,
        strategy: RankingStrategy = "swing",
    ) -> tuple[float, dict[str, float]]:
        """Compute the ranking score for a single theme.

        Formula: volume_component ** alpha * compellingness ** beta * lifecycle_multiplier

        The volume component is ``max(0, volume_zscore + 2) ** alpha``,
        shifting the z-score by +2 so that mildly negative z-scores still
        produce small positive values instead of zero.

        Args:
            theme: Theme to score.
            metrics: Latest ThemeMetrics for volume z-score.  May be None.
            strategy: Ranking strategy (swing or position).

        Returns:
            Tuple of (score, components_dict).
        """
        weights = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["swing"])
        alpha = weights["alpha"]
        beta = weights["beta"]

        # Volume component from z-score
        zscore = metrics.volume_zscore if metrics else None
        if zscore is None:
            zscore = 0.0
        shifted = max(0.0, zscore + 2.0)
        volume_component = math.pow(shifted, alpha) if shifted > 0 else 0.0

        # Compellingness from metadata (future: LLM scorer)
        raw_compellingness = theme.metadata.get(
            "compellingness", self._config.default_compellingness
        )
        compellingness = math.pow(
            max(0.0, float(raw_compellingness)), beta
        ) if raw_compellingness > 0 else 0.0

        # Lifecycle multiplier
        lifecycle_multiplier = LIFECYCLE_MULTIPLIERS.get(
            theme.lifecycle_stage, 0.8
        )

        score = volume_component * compellingness * lifecycle_multiplier

        components = {
            "volume_component": round(volume_component, 6),
            "compellingness_component": round(compellingness, 6),
            "lifecycle_multiplier": lifecycle_multiplier,
            "volume_zscore": zscore,
            "strategy": strategy,
        }

        return score, components

    def rank_themes(
        self,
        themes: list[Theme],
        metrics_map: dict[str, ThemeMetrics],
        strategy: RankingStrategy = "swing",
    ) -> list[RankedTheme]:
        """Score all themes, sort descending, and assign tiers.

        Args:
            themes: Themes to rank.
            metrics_map: theme_id → latest ThemeMetrics.
            strategy: Ranking strategy.

        Returns:
            Sorted list of RankedTheme (highest score first).
        """
        if not themes:
            return []

        ranked: list[RankedTheme] = []
        for theme in themes:
            metrics = metrics_map.get(theme.theme_id)
            score, components = self.compute_score(theme, metrics, strategy)

            if score < self._config.min_score_threshold:
                continue

            ranked.append(
                RankedTheme(
                    theme_id=theme.theme_id,
                    theme=theme,
                    score=score,
                    components=components,
                )
            )

        ranked.sort(key=lambda r: r.score, reverse=True)
        self._assign_tiers(ranked, metrics_map)

        return ranked

    def _assign_tiers(
        self,
        ranked: list[RankedTheme],
        metrics_map: dict[str, ThemeMetrics],
    ) -> None:
        """Assign tiers based on percentile position.

        Tier 1: Top 5% AND (z-score >= tier_1_min_zscore OR lifecycle == accelerating)
        Tier 2: Top 20%
        Tier 3: Everything else

        Mutates the tier field of each RankedTheme in place.
        """
        n = len(ranked)
        if n == 0:
            return

        tier_1_cutoff = max(1, math.ceil(n * self._config.tier_1_percentile))
        tier_2_cutoff = max(1, math.ceil(n * self._config.tier_2_percentile))

        for i, item in enumerate(ranked):
            position = i + 1  # 1-indexed
            metrics = metrics_map.get(item.theme_id)
            zscore = metrics.volume_zscore if metrics else None

            if position <= tier_1_cutoff:
                # Must also pass z-score gate or be accelerating
                zscore_ok = zscore is not None and zscore >= self._config.tier_1_min_zscore
                accelerating = item.theme.lifecycle_stage == "accelerating"
                if zscore_ok or accelerating:
                    item.tier = 1
                else:
                    item.tier = 2
            elif position <= tier_2_cutoff:
                item.tier = 2
            else:
                item.tier = 3

    # ── Async orchestrator ───────────────────────────────

    async def get_actionable(
        self,
        strategy: RankingStrategy | None = None,
        max_tier: int = 3,
    ) -> list[RankedTheme]:
        """Fetch themes + latest metrics, rank, and filter by tier.

        Args:
            strategy: Ranking strategy. Defaults to config default.
            max_tier: Maximum tier to include (1, 2, or 3).

        Returns:
            Sorted list of RankedTheme up to max_tier.

        Raises:
            RuntimeError: If theme_repo is not configured.
        """
        if self._theme_repo is None:
            raise RuntimeError(
                "theme_repo is required for get_actionable"
            )

        effective_strategy: RankingStrategy = (
            strategy or self._config.default_strategy  # type: ignore[assignment]
        )

        # Fetch all themes
        themes = await self._theme_repo.get_all(limit=1000)

        # Fetch latest metrics for each theme (last 1 day)
        today = date.today()
        yesterday = today - timedelta(days=1)
        metrics_map: dict[str, ThemeMetrics] = {}

        for theme in themes:
            metrics_list = await self._theme_repo.get_metrics_range(
                theme.theme_id, yesterday, today
            )
            if metrics_list:
                # Use the most recent entry
                metrics_map[theme.theme_id] = metrics_list[-1]

        ranked = self.rank_themes(themes, metrics_map, effective_strategy)

        # Filter by tier
        return [r for r in ranked if r.tier <= max_tier]
