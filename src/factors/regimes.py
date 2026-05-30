"""Factor regime classification and point-in-time joins."""

from __future__ import annotations

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
        matches_by_theme = {
            theme.theme_id: _matching_series(series_list, _theme_factor_tags(theme))[
                :max_factors_per_theme
            ]
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
            context_map[theme.theme_id] = contexts
        return context_map


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
    tags = {_normalise_tag(str(tag)) for tag in raw_tags if str(tag).strip()}
    text_parts = [
        getattr(theme, "name", ""),
        getattr(theme, "description", ""),
        " ".join(str(keyword) for keyword in getattr(theme, "top_keywords", []) or []),
        " ".join(str(ticker) for ticker in getattr(theme, "top_tickers", []) or []),
    ]
    for entity in getattr(theme, "top_entities", []) or []:
        if isinstance(entity, dict):
            text_parts.append(str(entity.get("name") or entity.get("text") or ""))
        else:
            text_parts.append(str(entity))
    text = " ".join(text_parts).lower().replace("-", " ").replace("_", " ")
    tags.update(_normalise_tag(keyword) for keyword in text.split() if keyword.strip())
    tags.update(_derived_factor_tags(text))
    return {tag for tag in tags if tag}


def _matching_series(
    series_list: list[FactorSeries],
    theme_tags: set[str],
) -> list[FactorSeries]:
    if not theme_tags:
        return []
    return [
        series
        for series in series_list
        if theme_tags.intersection(_normalise_tag(tag) for tag in series.relevance_tags)
    ]


def _normalise_tag(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _derived_factor_tags(text: str) -> set[str]:
    tags: set[str] = set()
    if _contains_any(
        text,
        "semiconductor",
        "semiconductors",
        "chip",
        "chips",
        "gpu",
        "hbm",
        "nvidia",
        "amd",
        "tsmc",
        "foundry",
        "fab",
    ):
        tags.add("semiconductors")
    if _contains_any(text, "hbm", "dram", "nand", "memory"):
        tags.add("memory")
    if _contains_any(text, "ai", "data center", "datacenter", "power", "electricity"):
        tags.update({"ai_infrastructure", "energy"})
    if _contains_any(text, "energy", "utility", "electricity", "power"):
        tags.add("energy")
    if _contains_any(text, "rate", "rates", "yield", "treasury", "fed"):
        tags.update({"rates", "yield_curve", "macro"})
    if _contains_any(text, "inflation", "cpi", "ppi", "consumer price"):
        tags.update({"inflation", "consumer", "macro"})
    if _contains_any(text, "jobs", "labor", "payroll", "employment"):
        tags.update({"labor", "macro"})
    if _contains_any(text, "industrial", "production", "manufacturing", "pmi"):
        tags.update({"industry", "macro"})
    if _contains_any(text, "gdp", "growth"):
        tags.update({"growth", "macro"})
    if _contains_any(text, "profit", "profits", "earnings", "margin"):
        tags.update({"profits", "earnings", "macro"})
    if _contains_any(text, "trade", "import", "imports", "export", "tariff", "china"):
        tags.update({"trade", "supply_chain"})
    if _contains_any(text, "capex", "equipment", "asml"):
        tags.add("capex")
    return tags


def _contains_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)
