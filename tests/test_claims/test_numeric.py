"""Tests for typed numeric fact parsing and metric/modality inference.

The numeric module turns the free-text ``quantity`` strings captured by
event extraction (``$42 billion``, ``36%``, ``3nm``, ``8 weeks``) into typed,
normalized fields on a claim.
"""

from __future__ import annotations

import pytest

from src.claims.numeric import (
    infer_metric,
    infer_modality,
    parse_quantity,
)
from src.claims.schemas import VALID_MODALITIES


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
