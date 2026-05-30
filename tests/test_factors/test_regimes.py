"""Tests for factor regime classification and point-in-time joins."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from src.factors.regimes import FactorRegimeService, classify_factor_regime
from src.factors.schemas import FactorObservation, FactorSeries


def _series() -> FactorSeries:
    return FactorSeries(
        factor_id="fred:DGS10",
        provider="fred",
        external_id="DGS10",
        name="10-Year Treasury Constant Maturity Rate",
        units="percent",
        cadence="daily",
        relevance_tags=["rates", "macro"],
    )


def _observation(
    value: float | None,
    *,
    observation_date: date = date(2026, 5, 1),
    available_at: datetime = datetime(2026, 5, 2, tzinfo=UTC),
) -> FactorObservation:
    return FactorObservation(
        factor_id="fred:DGS10",
        observation_date=observation_date,
        value=value,
        units="percent",
        available_at=available_at,
        missing_reason="provider_missing_value" if value is None else None,
    )


class FakeFactorRepository:
    def __init__(
        self,
        series: list[FactorSeries],
        observations: list[FactorObservation],
    ) -> None:
        self.series = series
        self.observations = observations
        self.calls: list[dict[str, Any]] = []

    async def list_series(
        self,
        *,
        active_only: bool = False,
        relevance_tag: str | None = None,
        provider: str | None = None,
    ) -> list[FactorSeries]:
        return self.series

    async def get_latest_observations_as_of(
        self,
        factor_ids: list[str],
        *,
        start: date,
        end: date,
        as_of: datetime,
        per_factor: int = 2,
    ) -> dict[str, list[FactorObservation]]:
        self.calls.append(
            {
                "factor_ids": factor_ids,
                "start": start,
                "end": end,
                "as_of": as_of,
                "per_factor": per_factor,
            }
        )
        result: dict[str, list[FactorObservation]] = {}
        for factor_id in factor_ids:
            visible = [
                obs
                for obs in self.observations
                if obs.factor_id == factor_id
                and start <= obs.observation_date <= end
                and obs.available_at <= as_of
            ]
            result[factor_id] = sorted(visible, key=lambda obs: obs.observation_date)[
                -per_factor:
            ]
        return result


def test_classifies_rising_falling_stable_and_missing_regimes() -> None:
    series = _series()

    rising = classify_factor_regime(series, _observation(4.2), _observation(4.0))
    falling = classify_factor_regime(series, _observation(3.8), _observation(4.0))
    stable = classify_factor_regime(series, _observation(4.01), _observation(4.0))
    missing = classify_factor_regime(series, _observation(None), _observation(4.0))

    assert rising.regime == "rising"
    assert rising.metadata["delta"] == pytest.approx(0.2)
    assert falling.regime == "falling"
    assert stable.regime == "stable"
    assert missing.regime == "missing"


@pytest.mark.asyncio
async def test_build_context_map_joins_theme_tags_to_factor_observations_as_of() -> None:
    series = _series()
    as_of = datetime(2026, 5, 3, 23, 59, tzinfo=UTC)
    repo = FakeFactorRepository(
        [series],
        [
            _observation(
                4.0,
                observation_date=date(2026, 4, 30),
                available_at=datetime(2026, 5, 1, tzinfo=UTC),
            ),
            _observation(
                4.2,
                observation_date=date(2026, 5, 1),
                available_at=datetime(2026, 5, 2, tzinfo=UTC),
            ),
            _observation(
                5.0,
                observation_date=date(2026, 5, 2),
                available_at=as_of + timedelta(days=1),
            ),
        ],
    )
    theme = type(
        "ThemeLike",
        (),
        {
            "theme_id": "theme_rates",
            "metadata": {"factor_relevance_tags": ["rates"]},
        },
    )()
    service = FactorRegimeService(repo)

    context_map = await service.build_context_map([theme], as_of=as_of)

    assert repo.calls == [
        {
            "factor_ids": ["fred:DGS10"],
            "start": date(2025, 5, 3),
            "end": date(2026, 5, 3),
            "as_of": as_of,
            "per_factor": 2,
        }
    ]
    assert context_map["theme_rates"][0].factor_id == "fred:DGS10"
    assert context_map["theme_rates"][0].observation_date == date(2026, 5, 1)
    assert context_map["theme_rates"][0].value == 4.2
    assert context_map["theme_rates"][0].regime == "rising"


@pytest.mark.asyncio
async def test_build_context_map_omits_unrelated_theme_tags() -> None:
    service = FactorRegimeService(FakeFactorRepository([_series()], [_observation(4.2)]))
    theme = type(
        "ThemeLike",
        (),
        {
            "theme_id": "theme_memory",
            "metadata": {"factor_relevance_tags": ["memory"]},
        },
    )()

    context_map = await service.build_context_map(
        [theme],
        as_of=datetime(2026, 5, 3, tzinfo=UTC),
    )

    assert context_map["theme_memory"] == []


@pytest.mark.asyncio
async def test_build_context_map_batches_unique_factor_observations_once() -> None:
    series = _series()
    repo = FakeFactorRepository([series], [_observation(4.0), _observation(4.2)])
    theme_a = type(
        "ThemeLike",
        (),
        {"theme_id": "theme_a", "metadata": {"factor_relevance_tags": ["rates"]}},
    )()
    theme_b = type(
        "ThemeLike",
        (),
        {"theme_id": "theme_b", "metadata": {"factor_relevance_tags": ["rates"]}},
    )()

    context_map = await FactorRegimeService(repo).build_context_map(
        [theme_a, theme_b],
        as_of=datetime(2026, 5, 3, tzinfo=UTC),
    )

    assert list(context_map) == ["theme_a", "theme_b"]
    assert len(repo.calls) == 1
    assert repo.calls[0]["factor_ids"] == ["fred:DGS10"]


@pytest.mark.asyncio
async def test_build_context_map_derives_relevance_from_theme_keywords() -> None:
    memory_series = FactorSeries(
        factor_id="census:imports:hs854232:value",
        provider="census",
        external_id="imports",
        name="U.S. Imports of Memory Integrated Circuits",
        units="usd",
        cadence="monthly",
        relevance_tags=["memory", "semiconductors"],
    )
    observation = FactorObservation(
        factor_id=memory_series.factor_id,
        observation_date=date(2026, 4, 1),
        value=100.0,
        units="usd",
        available_at=datetime(2026, 5, 15, tzinfo=UTC),
    )
    repo = FakeFactorRepository([memory_series], [observation])
    theme = type(
        "ThemeLike",
        (),
        {
            "theme_id": "theme_hbm",
            "name": "HBM demand",
            "top_keywords": ["hbm", "gpu"],
            "top_tickers": ["NVDA", "MU"],
            "top_entities": [{"name": "NVIDIA"}],
            "metadata": {"bertopic_topic_id": 7},
        },
    )()

    context_map = await FactorRegimeService(repo).build_context_map(
        [theme],
        as_of=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert context_map["theme_hbm"][0].factor_id == "census:imports:hs854232:value"


@pytest.mark.asyncio
async def test_build_context_map_does_not_match_relevance_inside_unrelated_words() -> None:
    ai_series = FactorSeries(
        factor_id="eia:electricity:retail_sales:industrial_sales_us",
        provider="eia",
        external_id="electricity/retail-sales",
        name="Industrial Electricity Sales",
        units="megawatthours",
        cadence="monthly",
        relevance_tags=["ai_infrastructure", "energy"],
    )
    observation = FactorObservation(
        factor_id=ai_series.factor_id,
        observation_date=date(2026, 4, 1),
        value=100.0,
        units="megawatthours",
        available_at=datetime(2026, 5, 15, tzinfo=UTC),
    )
    repo = FakeFactorRepository([ai_series], [observation])
    theme = type(
        "ThemeLike",
        (),
        {
            "theme_id": "theme_retail_chain",
            "name": "Retail chain paid media optimization",
            "metadata": {},
        },
    )()

    context_map = await FactorRegimeService(repo).build_context_map(
        [theme],
        as_of=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert context_map["theme_retail_chain"] == []
