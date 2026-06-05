"""Tests for the deterministic templated-briefing fallback.

Used when the LLM is unavailable (breaker open / no key / empty result). Each
claim becomes one clause citing its own id, so the output is always grounded
and never empty-of-citations. Pure function.
"""

from __future__ import annotations

from typing import Any

from src.briefing.template import templated_clauses
from src.claims.schemas import EvidenceClaim, make_claim_key


def _claim(claim_id: str, **overrides: Any) -> EvidenceClaim:
    key = make_claim_key("narrative", claim_id, "TSMC", "supplies_to", "NVIDIA")
    base: dict[str, Any] = {
        "claim_id": claim_id,
        "claim_key": key,
        "lane": "narrative",
        "source_id": claim_id,
        "predicate": "supplies_to",
        "subject_text": "TSMC",
        "object_text": "NVIDIA",
        "contract_version": "v1",
    }
    base.update(overrides)
    return EvidenceClaim(**base)


def test_one_clause_per_claim_each_self_cited() -> None:
    clauses = templated_clauses([_claim("claim_a"), _claim("claim_b")], limit=5)
    assert len(clauses) == 2
    assert clauses[0].claim_ids == ["claim_a"]
    assert clauses[1].claim_ids == ["claim_b"]
    assert "TSMC" in clauses[0].text and "NVIDIA" in clauses[0].text


def test_respects_limit() -> None:
    clauses = templated_clauses([_claim(f"c{i}") for i in range(10)], limit=3)
    assert len(clauses) == 3


def test_empty_claims_yield_no_clauses() -> None:
    assert templated_clauses([], limit=5) == []


def test_includes_numeric_fact_when_present() -> None:
    claim = _claim(
        "claim_n",
        predicate="revises_guidance",
        object_text="capex",
        metric="capex",
        numeric_value=42_000_000_000.0,
        unit="USD",
        period="2026",
    )
    text = templated_clauses([claim], limit=5)[0].text
    assert "capex" in text
    assert "42000000000" in text
