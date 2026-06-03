"""Tests for typed numeric fact parsing and metric/modality inference.

The numeric module turns the free-text ``quantity`` strings captured by
event extraction (``$42 billion``, ``36%``, ``3nm``, ``8 weeks``) into typed,
normalized fields on a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pytest

from src.claims.numeric import (
    compare_numeric_facts,
    infer_metric,
    infer_modality,
    numeric_link_type,
    parse_quantity,
)
from src.claims.schemas import VALID_MODALITIES


@dataclass
class _Fact:
    """Minimal structural stand-in satisfying NumericClaimLike for tests."""

    claim_id: str = "claim_x"
    metric: str | None = "capex"
    numeric_value: float | None = None
    unit: str | None = "USD"
    period: str | None = "2026-Q3"
    confidence: float = 0.7
    source_published_at: datetime | None = None


class TestCompareNumericFacts:
    """Pairwise comparison of two typed numeric facts."""

    def test_same_metric_period_unit_within_tolerance_agrees(self):
        a = _Fact(numeric_value=42e9)
        b = _Fact(numeric_value=43e9)  # ~2.3% apart, under 5% default
        assert compare_numeric_facts(a, b) == "agree"

    def test_same_context_beyond_tolerance_contradicts(self):
        a = _Fact(numeric_value=42e9)
        b = _Fact(numeric_value=36e9)  # ~14% apart
        assert compare_numeric_facts(a, b) == "contradict"

    def test_different_unit_is_incomparable(self):
        a = _Fact(numeric_value=42e9, unit="USD")
        b = _Fact(numeric_value=42e9, unit="count")
        assert compare_numeric_facts(a, b) == "incomparable"

    def test_different_metric_is_incomparable(self):
        a = _Fact(numeric_value=42e9, metric="capex")
        b = _Fact(numeric_value=42e9, metric="capacity")
        assert compare_numeric_facts(a, b) == "incomparable"

    def test_different_period_is_incomparable(self):
        a = _Fact(numeric_value=42e9, period="2026-Q3")
        b = _Fact(numeric_value=42e9, period="2026-Q4")
        assert compare_numeric_facts(a, b) == "incomparable"

    def test_missing_value_is_incomparable(self):
        a = _Fact(numeric_value=None)
        b = _Fact(numeric_value=42e9)
        assert compare_numeric_facts(a, b) == "incomparable"

    def test_both_zero_agree(self):
        a = _Fact(numeric_value=0.0)
        b = _Fact(numeric_value=0.0)
        assert compare_numeric_facts(a, b) == "agree"

    def test_both_none_period_is_comparable(self):
        a = _Fact(numeric_value=42e9, period=None)
        b = _Fact(numeric_value=36e9, period=None)
        assert compare_numeric_facts(a, b) == "contradict"

    def test_custom_tolerance_widens_agreement(self):
        a = _Fact(numeric_value=42e9)
        b = _Fact(numeric_value=36e9)
        assert compare_numeric_facts(a, b, rel_tolerance=0.20) == "agree"


class TestNumericLinkType:
    """Mapping pairwise comparison onto support/contradiction link types."""

    def test_agree_maps_to_support(self):
        a = _Fact(numeric_value=42e9)
        b = _Fact(numeric_value=43e9)
        assert numeric_link_type(a, b) == "support"

    def test_contradict_maps_to_contradiction(self):
        a = _Fact(numeric_value=42e9)
        b = _Fact(numeric_value=36e9)
        assert numeric_link_type(a, b) == "contradiction"

    def test_incomparable_maps_to_none(self):
        a = _Fact(numeric_value=42e9, unit="USD")
        b = _Fact(numeric_value=42e9, unit="count")
        assert numeric_link_type(a, b) is None


class TestParseQuantity:
    def test_dollars_billion_word(self):
        q = parse_quantity("$42 billion")
        assert q is not None
        assert q.value == pytest.approx(42_000_000_000.0)
        assert q.unit == "USD"

    def test_dollars_million_word(self):
        q = parse_quantity("$36 million")
        assert q is not None
        assert q.value == pytest.approx(36_000_000.0)
        assert q.unit == "USD"

    def test_dollars_billion_suffix_no_space(self):
        q = parse_quantity("$1.5B")
        assert q is not None
        assert q.value == pytest.approx(1_500_000_000.0)
        assert q.unit == "USD"

    def test_dollars_million_suffix(self):
        q = parse_quantity("$500M")
        assert q is not None
        assert q.value == pytest.approx(500_000_000.0)
        assert q.unit == "USD"

    def test_percent(self):
        q = parse_quantity("36%")
        assert q is not None
        assert q.value == pytest.approx(36.0)
        assert q.unit == "%"

    def test_nanometers(self):
        q = parse_quantity("3nm")
        assert q is not None
        assert q.value == pytest.approx(3.0)
        assert q.unit == "nm"

    def test_units_plural(self):
        q = parse_quantity("100 units")
        assert q is not None
        assert q.value == pytest.approx(100.0)
        assert q.unit == "count"

    def test_weeks_lead_time(self):
        q = parse_quantity("8 weeks")
        assert q is not None
        assert q.value == pytest.approx(8.0)
        assert q.unit == "weeks"

    def test_comma_grouped_number(self):
        q = parse_quantity("$1,200 million")
        assert q is not None
        assert q.value == pytest.approx(1_200_000_000.0)
        assert q.unit == "USD"

    def test_bare_number_is_count(self):
        q = parse_quantity("250")
        assert q is not None
        assert q.value == pytest.approx(250.0)
        assert q.unit == "count"

    def test_garbage_returns_none(self):
        assert parse_quantity("next quarter") is None

    def test_empty_returns_none(self):
        assert parse_quantity("") is None

    def test_none_returns_none(self):
        assert parse_quantity(None) is None


class TestInferMetric:
    def test_price_change(self):
        assert infer_metric("price_change") == "price"

    def test_guidance_change(self):
        assert infer_metric("guidance_change") == "guidance"

    def test_product_launch_is_timing(self):
        assert infer_metric("product_launch") == "product_timing"

    def test_product_delay_is_timing(self):
        assert infer_metric("product_delay") == "product_timing"

    def test_capacity_expansion_default(self):
        assert infer_metric("capacity_expansion") == "capacity"

    def test_capacity_expansion_with_dollars_is_capex(self):
        assert infer_metric("capacity_expansion", unit="USD") == "capex"

    def test_capacity_expansion_invest_action_is_capex(self):
        assert infer_metric("capacity_expansion", action="invests") == "capex"

    def test_capacity_constraint_default(self):
        assert infer_metric("capacity_constraint") == "capacity"

    def test_capacity_constraint_lead_time_unit(self):
        assert infer_metric("capacity_constraint", unit="weeks") == "lead_time"

    def test_unknown_event_type_returns_none(self):
        assert infer_metric("merger_rumor") is None


class TestInferModality:
    def test_default_is_confirmed(self):
        assert infer_modality("expanded its fab") == "confirmed"

    def test_guidance_event_is_guided(self):
        assert infer_modality("anything", event_type="guidance_change") == "guided"

    def test_forward_looking_cue_is_guided(self):
        assert infer_modality("TSMC expects to expand capacity") == "guided"

    def test_plans_to_is_guided(self):
        assert infer_modality("plans to build a new fab") == "guided"

    def test_reportedly_is_rumored(self):
        assert infer_modality("reportedly will cut production") == "rumored"

    def test_analyst_estimate(self):
        assert infer_modality("analysts estimate revenue of") == "estimate"

    def test_rumored_overrides_guided(self):
        # "reportedly ... will" — rumor about a forward-looking claim → rumored
        assert infer_modality("sources say it will expand") == "rumored"

    def test_all_results_are_valid(self):
        for text in ("expanded", "expects to grow", "reportedly", "analyst estimate"):
            assert infer_modality(text) in VALID_MODALITIES

    def test_none_text_is_confirmed(self):
        assert infer_modality(None) == "confirmed"
