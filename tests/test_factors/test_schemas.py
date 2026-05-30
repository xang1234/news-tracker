"""Tests for macro/supply-chain factor registry schemas."""

from datetime import UTC, date, datetime

import pytest

from src.factors.schemas import (
    VALID_FACTOR_CADENCES,
    FactorObservation,
    FactorSeries,
    validate_observation_for_series,
)


class TestFactorSeries:
    """FactorSeries registry metadata validation."""

    def test_registry_entry_captures_required_metadata(self) -> None:
        series = FactorSeries(
            factor_id="fred:DGS10",
            provider="fred",
            external_id="DGS10",
            name="10-Year Treasury Constant Maturity Rate",
            units="percent",
            cadence="daily",
            release_lag_days=1,
            relevance_tags=["rates", "macro", "discount-rate"],
            required_credentials=["FRED_API_KEY"],
            source_url="https://fred.stlouisfed.org/series/DGS10",
        )

        assert series.factor_id == "fred:DGS10"
        assert series.provider == "fred"
        assert series.external_id == "DGS10"
        assert series.units == "percent"
        assert series.cadence == "daily"
        assert series.release_lag_days == 1
        assert series.relevance_tags == ["rates", "macro", "discount-rate"]
        assert series.required_credentials == ["FRED_API_KEY"]

    def test_supported_cadences_are_explicit(self) -> None:
        assert frozenset(
            {"daily", "weekly", "monthly", "quarterly", "annual", "irregular"}
        ) == VALID_FACTOR_CADENCES

    def test_invalid_cadence_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid cadence"):
            FactorSeries(
                factor_id="fred:DGS10",
                provider="fred",
                external_id="DGS10",
                name="10-Year Treasury Constant Maturity Rate",
                units="percent",
                cadence="hourly",
            )

    def test_negative_release_lag_rejected(self) -> None:
        with pytest.raises(ValueError, match="release_lag_days"):
            FactorSeries(
                factor_id="fred:DGS10",
                provider="fred",
                external_id="DGS10",
                name="10-Year Treasury Constant Maturity Rate",
                units="percent",
                cadence="daily",
                release_lag_days=-1,
            )

    def test_empty_relevance_tag_rejected(self) -> None:
        with pytest.raises(ValueError, match="relevance_tags"):
            FactorSeries(
                factor_id="fred:DGS10",
                provider="fred",
                external_id="DGS10",
                name="10-Year Treasury Constant Maturity Rate",
                units="percent",
                cadence="daily",
                relevance_tags=["rates", ""],
            )

    def test_empty_required_credential_rejected(self) -> None:
        with pytest.raises(ValueError, match="required_credentials"):
            FactorSeries(
                factor_id="fred:DGS10",
                provider="fred",
                external_id="DGS10",
                name="10-Year Treasury Constant Maturity Rate",
                units="percent",
                cadence="daily",
                required_credentials=["FRED_API_KEY", ""],
            )


class TestFactorObservation:
    """FactorObservation point-in-time validation."""

    def test_observation_captures_availability_and_fetch_lineage(self) -> None:
        available_at = datetime(2026, 5, 1, 13, 30, tzinfo=UTC)
        fetched_at = datetime(2026, 5, 1, 13, 45, tzinfo=UTC)

        observation = FactorObservation(
            factor_id="fred:DGS10",
            observation_date=date(2026, 4, 30),
            value=4.52,
            units="percent",
            available_at=available_at,
            fetched_at=fetched_at,
            revision="initial",
            metadata={"provider_payload_id": "obs-1"},
        )

        assert observation.available_at == available_at
        assert observation.fetched_at == fetched_at
        assert observation.value == 4.52
        assert observation.revision == "initial"
        assert observation.is_missing is False

    def test_missing_observation_requires_reason(self) -> None:
        observation = FactorObservation(
            factor_id="census:semi_exports",
            observation_date=date(2026, 4, 1),
            value=None,
            units="usd",
            available_at=datetime(2026, 5, 15, tzinfo=UTC),
            missing_reason="provider_suppressed",
        )

        assert observation.is_missing is True
        assert observation.missing_reason == "provider_suppressed"

    def test_missing_observation_without_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing_reason"):
            FactorObservation(
                factor_id="census:semi_exports",
                observation_date=date(2026, 4, 1),
                value=None,
                units="usd",
                available_at=datetime(2026, 5, 15, tzinfo=UTC),
            )

    def test_unit_mismatch_rejected_against_registry_entry(self) -> None:
        series = FactorSeries(
            factor_id="fred:DGS10",
            provider="fred",
            external_id="DGS10",
            name="10-Year Treasury Constant Maturity Rate",
            units="percent",
            cadence="daily",
        )
        observation = FactorObservation(
            factor_id="fred:DGS10",
            observation_date=date(2026, 4, 30),
            value=0.0452,
            units="decimal",
            available_at=datetime(2026, 5, 1, tzinfo=UTC),
        )

        with pytest.raises(ValueError, match="units"):
            validate_observation_for_series(series, observation)

    def test_factor_id_mismatch_rejected_against_registry_entry(self) -> None:
        series = FactorSeries(
            factor_id="fred:DGS10",
            provider="fred",
            external_id="DGS10",
            name="10-Year Treasury Constant Maturity Rate",
            units="percent",
            cadence="daily",
        )
        observation = FactorObservation(
            factor_id="fred:DGS2",
            observation_date=date(2026, 4, 30),
            value=4.1,
            units="percent",
            available_at=datetime(2026, 5, 1, tzinfo=UTC),
        )

        with pytest.raises(ValueError, match="factor_id"):
            validate_observation_for_series(series, observation)
