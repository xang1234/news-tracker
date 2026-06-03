"""Tests for numeric contradiction classification.

Turns a set of comparable numeric claims into per-claim link-type *opinions*
(support/contradiction) by picking a per-group anchor and comparing every other
fact against it. The engine turns these opinions into the links that let
``aggregate_assertion`` flip an assertion to ``disputed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from src.assertions.numeric_contradiction import numeric_link_types


@dataclass
class _Fact:
    """Minimal structural stand-in satisfying NumericClaimLike."""

    claim_id: str
    numeric_value: float | None = 42e9
    metric: str | None = "capex"
    unit: str | None = "USD"
    period: str | None = "2026-Q3"
    confidence: float = 0.7
    source_published_at: datetime | None = None


class TestNumericLinkTypes:
    def test_agreeing_facts_all_support(self):
        claims = [
            _Fact("a", numeric_value=42e9),
            _Fact("b", numeric_value=43e9),
        ]
        result = numeric_link_types(claims)
        assert result == {"a": "support", "b": "support"}

    def test_contradicting_fact_flagged_against_anchor(self):
        # Higher-confidence claim is the anchor (support); the divergent one
        # contradicts it.
        claims = [
            _Fact("anchor", numeric_value=42e9, confidence=0.9),
            _Fact("outlier", numeric_value=30e9, confidence=0.6),
        ]
        result = numeric_link_types(claims)
        assert result == {"anchor": "support", "outlier": "contradiction"}

    def test_non_numeric_claim_is_support(self):
        claims = [_Fact("rel", numeric_value=None, metric=None)]
        assert numeric_link_types(claims) == {"rel": "support"}

    def test_different_units_are_separate_groups_all_support(self):
        # Incomparable across units → each is its own group anchor → support.
        claims = [
            _Fact("usd", numeric_value=42e9, unit="USD"),
            _Fact("count", numeric_value=42e9, unit="count"),
        ]
        assert numeric_link_types(claims) == {"usd": "support", "count": "support"}

    def test_different_periods_do_not_contradict(self):
        claims = [
            _Fact("q3", numeric_value=42e9, period="2026-Q3"),
            _Fact("q4", numeric_value=30e9, period="2026-Q4"),
        ]
        assert numeric_link_types(claims) == {"q3": "support", "q4": "support"}

    def test_anchor_tiebreak_prefers_recent_then_id(self):
        older = datetime(2026, 1, 1, tzinfo=UTC)
        newer = datetime(2026, 5, 1, tzinfo=UTC)
        # Same confidence; the more recently published fact anchors.
        claims = [
            _Fact("old", numeric_value=30e9, confidence=0.7, source_published_at=older),
            _Fact("new", numeric_value=42e9, confidence=0.7, source_published_at=newer),
        ]
        result = numeric_link_types(claims)
        assert result == {"new": "support", "old": "contradiction"}
