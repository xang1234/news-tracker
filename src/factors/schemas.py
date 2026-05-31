"""Schema definitions for macro and supply-chain factor series."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

VALID_FACTOR_CADENCES = frozenset(
    {"daily", "weekly", "monthly", "quarterly", "annual", "irregular"}
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_timezone_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass
class FactorSeries:
    """A registered macro or supply-chain time series.

    The registry is provider-neutral: FRED, BLS, EIA, Census, and future
    sources all use the same identity and metadata shape.
    """

    factor_id: str
    provider: str
    external_id: str
    name: str
    units: str
    cadence: str
    description: str = ""
    release_lag_days: int = 0
    relevance_tags: list[str] = field(default_factory=list)
    required_credentials: list[str] = field(default_factory=list)
    source_url: str | None = None
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.factor_id, "factor_id")
        _require_non_empty(self.provider, "provider")
        _require_non_empty(self.external_id, "external_id")
        _require_non_empty(self.name, "name")
        _require_non_empty(self.units, "units")

        if self.cadence not in VALID_FACTOR_CADENCES:
            raise ValueError(
                f"Invalid cadence {self.cadence!r}. Must be one of {sorted(VALID_FACTOR_CADENCES)}"
            )
        if self.release_lag_days < 0:
            raise ValueError("release_lag_days must be non-negative")
        if any(not tag.strip() for tag in self.relevance_tags):
            raise ValueError("relevance_tags must not contain empty values")
        if any(not credential.strip() for credential in self.required_credentials):
            raise ValueError("required_credentials must not contain empty values")


@dataclass
class FactorObservation:
    """One dated factor value with point-in-time availability lineage."""

    factor_id: str
    observation_date: date
    value: float | None
    units: str
    available_at: datetime
    fetched_at: datetime = field(default_factory=_utc_now)
    revision: str = ""
    missing_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.factor_id, "factor_id")
        _require_non_empty(self.units, "units")
        _require_timezone_aware(self.available_at, "available_at")
        _require_timezone_aware(self.fetched_at, "fetched_at")
        if self.value is None and not self.missing_reason:
            raise ValueError("missing_reason is required when value is missing")
        if self.value is not None and self.missing_reason:
            raise ValueError("missing_reason must be empty when value is present")

    @property
    def is_missing(self) -> bool:
        """Whether the provider explicitly lacked a value for this period."""
        return self.value is None


def validate_observation_for_series(
    series: FactorSeries,
    observation: FactorObservation,
) -> None:
    """Validate an observation against its registry entry."""
    if observation.factor_id != series.factor_id:
        raise ValueError(
            f"factor_id mismatch: observation {observation.factor_id!r} "
            f"does not match series {series.factor_id!r}"
        )
    if observation.units != series.units:
        raise ValueError(
            f"units mismatch for {series.factor_id}: observation {observation.units!r} "
            f"does not match series {series.units!r}"
        )
