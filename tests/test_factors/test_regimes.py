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

    async def get_observations_as_of(
        self,
        factor_id: str,
        *,
        start: date,
        end: date,
        as_of: datetime,
    ) -> list[FactorObservation]:
        self.calls.append(
            {
                "factor_id": factor_id,
                "start": start,
                "end": end,
                "as_of": as_of,
            }
        )
        return [
            obs
            for obs in self.observations
            if obs.factor_id == factor_id
            and start <= obs.observation_date <= end
            and obs.available_at <= as_of
        ]


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
            "factor_id": "fred:DGS10",
            "start": date(2025, 5, 3),
            "end": date(2026, 5, 3),
            "as_of": as_of,
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
