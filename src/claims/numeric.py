"""Typed numeric facts: parse event quantities into normalized, typed values.

Event extraction captures quantities as free-text strings (``$42 billion``,
``36%``, ``3nm``, ``8 weeks``). This module turns them into typed, normalized
fields (``metric`` / ``numeric_value`` / ``unit`` / ``modality``) so they can be
persisted as first-class, comparable facts on a claim.

Stateless pure functions only — no I/O — matching the project convention for
trivially-testable services (LifecycleClassifier, trigger functions, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

# Default relative tolerance for treating two numeric facts as agreeing.
# 5% absorbs rounding/reporting noise ("$42B" vs "$42.5B") while still
# flagging genuine disagreements ("$42B" vs "$36B").
DEFAULT_REL_TOLERANCE = 0.05

# Duration units imply a lead-time metric rather than a capacity metric.
_DURATION_UNITS = frozenset({"weeks", "months", "days"})

# Cue words for modality classification, checked in priority order:
# a rumor about a forward-looking claim is still a rumor.
_RUMOR_CUES = re.compile(
    r"\b(?:reportedly|rumor(?:s|ed)?|sources?\s+say|speculat\w*|allegedly)\b",
    re.IGNORECASE,
)
_ESTIMATE_CUES = re.compile(
    r"\b(?:analysts?|estimat\w*|consensus|projected\s+by)\b",
    re.IGNORECASE,
)
_GUIDED_CUES = re.compile(
    r"\b(?:expect\w*|plan(?:s|ned|ning)?|will|forecast\w*|project\w*|"
    r"target\w*|guidance|guide[sd]?|aim(?:s|ing)?|intends?|to\s+build)\b",
    re.IGNORECASE,
)

_INVEST_ACTION = re.compile(r"invest|spend|capex|capital", re.IGNORECASE)

# Scale multipliers for magnitude words/suffixes.
_SCALE: dict[str, float] = {
    "billion": 1e9,
    "b": 1e9,
    "million": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}

# Longer alternatives must precede shorter ones so e.g. "billion" wins over "b".
_QUANTITY_RE = re.compile(
    r"^\s*(?P<dollar>\$)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<suffix>billion|million|thousand|nm|units?|weeks?|months?|days?|[bmk]|%)?"
    r"\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedQuantity:
    """A normalized numeric quantity.

    Attributes:
        value: The magnitude in base units (USD dollars, raw percent points,
            nanometers, or a plain count) after applying any scale word.
        unit: Canonical unit — one of ``USD``, ``%``, ``nm``, ``count``,
            ``weeks``, ``months``, ``days``.
        raw: The original text that was parsed (for audit/display).
    """

    value: float
    unit: str
    raw: str


def parse_quantity(raw: str | None) -> ParsedQuantity | None:
    """Parse a free-text quantity string into a typed, normalized quantity.

    Returns None when the string contains no recognizable numeric quantity
    (e.g. a time reference like ``next quarter``), so callers can treat
    numeric and non-numeric claims uniformly.
    """
    if not raw or not raw.strip():
        return None

    match = _QUANTITY_RE.match(raw)
    if match is None:
        return None

    number = float(match.group("num").replace(",", ""))
    suffix = (match.group("suffix") or "").lower()
    has_dollar = match.group("dollar") is not None

    scale = _SCALE.get(suffix, 1.0)
    value = number * scale

    if has_dollar:
        unit = "USD"
    elif suffix == "%":
        unit = "%"
    elif suffix == "nm":
        unit = "nm"
    elif suffix in ("unit", "units"):
        unit = "count"
    elif suffix in ("week", "weeks"):
        unit = "weeks"
    elif suffix in ("month", "months"):
        unit = "months"
    elif suffix in ("day", "days"):
        unit = "days"
    else:
        # Bare number or a magnitude-scaled count (e.g. "5 million").
        unit = "count"

    return ParsedQuantity(value=value, unit=unit, raw=raw)


def infer_metric(
    event_type: str,
    *,
    action: str | None = None,
    unit: str | None = None,
) -> str | None:
    """Infer the metric (value-type) a numeric fact measures.

    Defaults derived from the six event extraction types:
    ``capex``, ``capacity``, ``price``, ``guidance``, ``product_timing``,
    ``lead_time``. Returns None for unrecognized event types so callers can
    skip metric assignment rather than guess.
    """
    if event_type == "price_change":
        return "price"
    if event_type == "guidance_change":
        return "guidance"
    if event_type in ("product_launch", "product_delay"):
        return "product_timing"
    if event_type == "capacity_expansion":
        # A capacity expansion expressed in dollars (or via an investment
        # verb) is capital expenditure, not raw capacity.
        if unit == "USD" or (action is not None and _INVEST_ACTION.search(action)):
            return "capex"
        return "capacity"
    if event_type == "capacity_constraint":
        if unit in _DURATION_UNITS:
            return "lead_time"
        return "capacity"
    return None


@runtime_checkable
class NumericClaimLike(Protocol):
    """Structural protocol for the numeric fields used in comparison.

    ``EvidenceClaim`` satisfies this structurally — the comparison and
    classification functions accept any object exposing these attributes,
    keeping them decoupled from the full claim schema.
    """

    claim_id: str
    metric: str | None
    numeric_value: float | None
    unit: str | None
    period: str | None
    confidence: float
    source_published_at: datetime | None


def compare_numeric_facts(
    a: NumericClaimLike,
    b: NumericClaimLike,
    *,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
) -> str:
    """Compare two numeric facts.

    Returns one of:
        ``"agree"`` — same (metric, unit, period) and values within tolerance.
        ``"contradict"`` — same (metric, unit, period) but values diverge.
        ``"incomparable"`` — missing value, or differing metric/unit/period.

    Facts are only comparable when they describe the same metric, in the
    same unit, for the same period. ``period`` of ``None`` on both sides is
    treated as the same (undated) period.
    """
    if a.numeric_value is None or b.numeric_value is None:
        return "incomparable"
    # A missing metric or unit means the fact isn't fully typed — treat as
    # incomparable rather than letting None == None slip through as a match.
    if a.metric is None or b.metric is None or a.metric != b.metric:
        return "incomparable"
    if a.unit is None or b.unit is None or a.unit != b.unit:
        return "incomparable"
    # period of None on both sides is a legitimately-comparable "undated" period.
    if a.period != b.period:
        return "incomparable"

    denom = max(abs(a.numeric_value), abs(b.numeric_value))
    if denom == 0.0:
        # Both zero (the only way denom is 0 given values are present).
        return "agree"
    relative_diff = abs(a.numeric_value - b.numeric_value) / denom
    return "agree" if relative_diff <= rel_tolerance else "contradict"


def numeric_link_type(
    a: NumericClaimLike,
    b: NumericClaimLike,
    *,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
) -> str | None:
    """Map a pairwise comparison onto an assertion link type.

    ``"agree"`` → ``"support"``, ``"contradict"`` → ``"contradiction"``,
    ``"incomparable"`` → ``None`` (no link implied).
    """
    comparison = compare_numeric_facts(a, b, rel_tolerance=rel_tolerance)
    if comparison == "agree":
        return "support"
    if comparison == "contradict":
        return "contradiction"
    return None


def infer_modality(text: str | None, *, event_type: str | None = None) -> str:
    """Classify the epistemic status of a fact from cue words.

    Priority: rumored > estimate > guided > confirmed — a rumor about a
    forward-looking number is still a rumor. A ``guidance_change`` event is
    forward-looking by definition, so it is at least ``guided``.
    """
    haystack = text or ""
    if _RUMOR_CUES.search(haystack):
        return "rumored"
    if _ESTIMATE_CUES.search(haystack):
        return "estimate"
    if event_type == "guidance_change" or _GUIDED_CUES.search(haystack):
        return "guided"
    return "confirmed"
