"""Factor regime classification and point-in-time joins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Protocol

from src.factors.relevance import (
    ThemeWithFactorTags,
    extract_theme_factor_tags,
    normalise_factor_tag,
)
from src.factors.schemas import FactorObservation, FactorSeries

DEFAULT_RELATIVE_THRESHOLD = 0.01
DEFAULT_LOOKBACK_DAYS = 365


class FactorRegimeRepository(Protocol):
    async def list_series(
        self,
        *,
        active_only: bool = False,
        relevance_tag: str | None = None,
        provider: str | None = None,
    ) -> list[FactorSeries]:
        """List registered factor series."""
        ...

    async def get_latest_observations_as_of(
        self,
        factor_ids: list[str],
        *,
        start: date,
        end: date,
        as_of: datetime,
        per_factor: int = 2,
    ) -> dict[str, list[FactorObservation]]:
        """Fetch the latest visible observations for multiple factors."""
        ...


class RankedThemeForFactorContext(Protocol):
    @property
    def theme_id(self) -> str:
        """Theme identifier used to attach context."""
        ...

    @property
    def theme(self) -> ThemeWithFactorTags:
        """Theme payload used for factor relevance matching."""
        ...


@dataclass(frozen=True)
class FactorRegimeContext:
    """Regime classification for one factor at a decision time."""

    factor_id: str
    provider: str
    name: str
    observation_date: date
    value: float | None
    units: str
    regime: str
    available_at: datetime
    relevance_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation for ranking/backtest payloads."""
        return {
            "factor_id": self.factor_id,
            "provider": self.provider,
            "name": self.name,
            "observation_date": self.observation_date.isoformat(),
            "value": self.value,
            "units": self.units,
            "regime": self.regime,
            "available_at": self.available_at.isoformat(),
            "relevance_tags": self.relevance_tags,
            "metadata": self.metadata,
        }


def classify_factor_regime(
    series: FactorSeries,
    current: FactorObservation,
    previous: FactorObservation | None = None,
    *,
    relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD,
) -> FactorRegimeContext:
    """Classify the current factor observation relative to the prior value."""
    metadata: dict[str, Any]
    if current.is_missing:
        regime = "missing"
        metadata = {"trend": "unknown"}
    elif previous is None or previous.value is None or previous.value == 0:
        regime = "observed"
        metadata = {"trend": "unknown"}
    else:
        assert current.value is not None
        delta = current.value - previous.value
        pct_change = delta / abs(previous.value)
        if abs(pct_change) < relative_threshold:
            regime = "stable"
        elif delta > 0:
            regime = "rising"
        else:
            regime = "falling"
        metadata = {
            "trend": regime,
            "delta": delta,
            "pct_change": pct_change,
            "previous_observation_date": previous.observation_date.isoformat(),
            "previous_value": previous.value,
        }

    return FactorRegimeContext(
        factor_id=series.factor_id,
        provider=series.provider,
        name=series.name,
        observation_date=current.observation_date,
        value=current.value,
        units=current.units,
        regime=regime,
        available_at=current.available_at,
        relevance_tags=list(series.relevance_tags),
        metadata=metadata,
    )


class FactorRegimeService:
    """Build factor regime context for theme ranking and backtest replay."""

    def __init__(self, repository: FactorRegimeRepository) -> None:
        self._repository = repository

    async def build_context_map(
        self,
        themes: Sequence[ThemeWithFactorTags],
        *,
        as_of: datetime,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        max_factors_per_theme: int = 5,
    ) -> dict[str, list[FactorRegimeContext]]:
        """Join theme relevance tags to factor observations visible at ``as_of``."""
        if max_factors_per_theme <= 0:
            return {theme.theme_id: [] for theme in themes}

        series_list = await self._repository.list_series(active_only=True)
        start = as_of.date() - timedelta(days=lookback_days)
        end = as_of.date()
        matches_by_theme = {
            theme.theme_id: _matching_series(series_list, extract_theme_factor_tags(theme))
            for theme in themes
        }
        series_by_id = {
            series.factor_id: series
            for matching in matches_by_theme.values()
            for series in matching
        }
        observations_by_factor = (
            await self._repository.get_latest_observations_as_of(
                list(series_by_id),
                start=start,
                end=end,
                as_of=as_of,
                per_factor=2,
            )
            if series_by_id
            else {}
        )

        context_map: dict[str, list[FactorRegimeContext]] = {}
        for theme in themes:
            contexts = []
            for series in matches_by_theme[theme.theme_id]:
                observations = observations_by_factor.get(series.factor_id, [])
                if observations:
                    contexts.append(_context_from_observations(series, observations))
                if len(contexts) >= max_factors_per_theme:
                    break
            context_map[theme.theme_id] = contexts
        return context_map

    async def build_ranked_context_map(
        self,
        ranked_themes: Sequence[RankedThemeForFactorContext],
        *,
        as_of: datetime,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        max_factors_per_theme: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Build serialised factor context for ranked theme payloads."""
        if not ranked_themes:
            return {}

        factor_context_map = await self.build_context_map(
            [ranked_theme.theme for ranked_theme in ranked_themes],
            as_of=as_of,
            lookback_days=lookback_days,
            max_factors_per_theme=max_factors_per_theme,
        )
        return {
            ranked_theme.theme_id: [
                context.to_dict() for context in factor_context_map.get(ranked_theme.theme_id, [])
            ]
            for ranked_theme in ranked_themes
        }


def _context_from_observations(
    series: FactorSeries,
    observations: list[FactorObservation],
) -> FactorRegimeContext:
    sorted_observations = sorted(observations, key=lambda obs: obs.observation_date)
    current = sorted_observations[-1]
    previous = sorted_observations[-2] if len(sorted_observations) > 1 else None
    return classify_factor_regime(series, current, previous)


def _matching_series(
    series_list: list[FactorSeries],
    theme_tags: set[str],
) -> list[FactorSeries]:
    if not theme_tags:
        return []
    return [
        series
        for series in series_list
        if theme_tags.intersection(normalise_factor_tag(tag) for tag in series.relevance_tags)
    ]
