"""Shared primitives for factor datasource providers."""

from __future__ import annotations

from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import httpx

from src.factors.schemas import FactorObservation, FactorSeries


class FactorHTTPGetClient(Protocol):
    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Fetch a URL with optional query parameters."""
        ...


class FactorHTTPClient(FactorHTTPGetClient, Protocol):
    async def post(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Post a JSON body to a URL."""
        ...


class FactorProviderError(Exception):
    """Base exception for factor provider failures."""


class MacroProviderError(FactorProviderError):
    """Backward-compatible macro provider error base."""


class MissingProviderCredentialError(MacroProviderError):
    """Raised when a provider cannot run without a configured free key."""


class ProviderResponseError(MacroProviderError):
    """Raised when an upstream provider returns a domain-level error."""


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def date_to_utc(value: str) -> datetime:
    """Convert an ISO date string to UTC midnight."""
    return datetime.combine(date.fromisoformat(value), datetime.min.time(), tzinfo=UTC)


def _period_end_for_cadence(cadence: str, observation_date: date) -> date:
    if cadence == "monthly":
        return date(
            observation_date.year,
            observation_date.month,
            monthrange(observation_date.year, observation_date.month)[1],
        )
    if cadence == "quarterly":
        quarter_end_month = ((observation_date.month - 1) // 3 + 1) * 3
        return date(
            observation_date.year,
            quarter_end_month,
            monthrange(observation_date.year, quarter_end_month)[1],
        )
    if cadence == "annual":
        return date(observation_date.year, 12, 31)
    return observation_date


def lagged_available_at(series: FactorSeries, observation_date: date) -> datetime:
    """Fallback availability for feeds without explicit release timestamps."""
    period_end = _period_end_for_cadence(series.cadence, observation_date)
    release_date = period_end + timedelta(days=series.release_lag_days)
    return datetime.combine(release_date, datetime.min.time(), tzinfo=UTC)


_MISSING_NUMBER_TOKENS = frozenset({"", ".", "-", "--", "na", "n/a", "none", "null", "nan"})


def parse_number(value: Any) -> float | None:
    """Parse provider numeric values while preserving explicit missing markers."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _MISSING_NUMBER_TOKENS:
        return None

    is_parenthesized_negative = text.startswith("(") and text.endswith(")")
    if is_parenthesized_negative:
        text = text[1:-1].strip()

    text = text.replace(",", "").removeprefix("$").strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if text.lower() in _MISSING_NUMBER_TOKENS:
        return None

    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return -parsed if is_parenthesized_negative else parsed


def make_observation(
    series: FactorSeries,
    *,
    observation_date: date,
    value: Any,
    fetched_at: datetime,
    revision: str = "",
    available_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> FactorObservation:
    """Build a point-in-time observation from provider payload fields."""
    parsed_value = parse_number(value)
    estimated_release_at = lagged_available_at(series, observation_date)
    observation_metadata = {
        "estimated_release_at": estimated_release_at.isoformat(),
        **(metadata or {}),
    }
    return FactorObservation(
        factor_id=series.factor_id,
        observation_date=observation_date,
        value=parsed_value,
        units=series.units,
        available_at=available_at or estimated_release_at,
        fetched_at=fetched_at,
        revision=revision,
        missing_reason="provider_missing_value" if parsed_value is None else None,
        metadata=observation_metadata,
    )


def date_in_range(value: date, start: date | None, end: date | None) -> bool:
    """Whether a date falls inside optional inclusive bounds."""
    if start is not None and value < start:
        return False
    return not (end is not None and value > end)


def latest_only(
    observations: list[FactorObservation],
    *,
    latest: bool,
) -> list[FactorObservation]:
    """Select the newest observation explicitly instead of relying on response order."""
    if not latest or not observations:
        return observations
    return [max(observations, key=lambda observation: observation.observation_date)]


def response_json(response: httpx.Response) -> Any:
    """Return decoded JSON with a provider-domain error on invalid JSON."""
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderResponseError("provider returned invalid JSON") from exc
