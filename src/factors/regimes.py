"""Factor regime classification and point-in-time joins."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Protocol

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

    async def get_observations_as_of(
        self,
        factor_id: str,
        *,
        start: date,
        end: date,
        as_of: datetime,
    ) -> list[FactorObservation]:
        """Fetch observations visible as of a decision timestamp."""
        ...


class ThemeWithFactorTags(Protocol):
    theme_id: str
    metadata: dict[str, Any]


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
        delta = current.value - previous.value if current.value is not None else 0.0
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
        series_list = await self._repository.list_series(active_only=True)
        start = as_of.date() - timedelta(days=lookback_days)
        end = as_of.date()

        entries = await asyncio.gather(
            *[
                self._build_theme_context(
                    theme,
                    series_list,
                    start=start,
                    end=end,
                    as_of=as_of,
                    max_factors=max_factors_per_theme,
                )
                for theme in themes
            ]
        )
        return dict(entries)

    async def _build_theme_context(
        self,
        theme: ThemeWithFactorTags,
        series_list: list[FactorSeries],
        *,
        start: date,
        end: date,
        as_of: datetime,
        max_factors: int,
    ) -> tuple[str, list[FactorRegimeContext]]:
        matching = _matching_series(series_list, _theme_factor_tags(theme))[:max_factors]
        context_candidates = await asyncio.gather(
            *[
                self._context_for_series(series, start=start, end=end, as_of=as_of)
                for series in matching
            ]
        )
        return theme.theme_id, [context for context in context_candidates if context is not None]

    async def _context_for_series(
        self,
        series: FactorSeries,
        *,
        start: date,
        end: date,
        as_of: datetime,
    ) -> FactorRegimeContext | None:
        observations = await self._repository.get_observations_as_of(
            series.factor_id,
            start=start,
            end=end,
            as_of=as_of,
        )
        return _context_from_observations(series, observations) if observations else None


def _context_from_observations(
    series: FactorSeries,
    observations: list[FactorObservation],
) -> FactorRegimeContext:
    sorted_observations = sorted(observations, key=lambda obs: obs.observation_date)
    current = sorted_observations[-1]
    previous = sorted_observations[-2] if len(sorted_observations) > 1 else None
    return classify_factor_regime(series, current, previous)


def _theme_factor_tags(theme: ThemeWithFactorTags) -> set[str]:
    metadata = theme.metadata or {}
    raw_tags = metadata.get("factor_relevance_tags") or metadata.get("relevance_tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    return {str(tag).lower() for tag in raw_tags if str(tag).strip()}


def _matching_series(
    series_list: list[FactorSeries],
    theme_tags: set[str],
) -> list[FactorSeries]:
    if not theme_tags:
        return []
    return [
        series
        for series in series_list
        if theme_tags.intersection(tag.lower() for tag in series.relevance_tags)
    ]
