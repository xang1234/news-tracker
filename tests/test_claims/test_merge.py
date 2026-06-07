"""Tests for the rule/LLM claim merge.

Both passes emit the same claim_key for the same triple, so the merge is a
keyed union: no duplicates, corroborated triples flip to 'hybrid', and order is
rule-first then LLM-only.
"""

from __future__ import annotations

from src.claims.merge import merge_claims
from src.claims.schemas import EvidenceClaim, make_claim_id, make_claim_key
from src.contracts.intelligence.lanes import LANE_NARRATIVE


def _claim(subject, predicate, obj=None, *, method="rule", doc="d1"):
    key = make_claim_key(LANE_NARRATIVE, doc, subject, predicate, obj)
    return EvidenceClaim(
        claim_id=make_claim_id(key),
        claim_key=key,
        lane=LANE_NARRATIVE,
        source_id=doc,
        subject_text=subject,
        predicate=predicate,
        object_text=obj,
        extraction_method=method,
    )


def test_union_of_disjoint_claims() -> None:
    rule = [_claim("TSMC", "expands_capacity", "Arizona fab")]
    llm = [_claim("TSMC", "supplies_to", "NVIDIA", method="llm")]
    merged = merge_claims(rule, llm)
    assert len(merged) == 2
    assert [c.extraction_method for c in merged] == ["rule", "llm"]


def test_same_triple_collapses_to_one_hybrid() -> None:
    rule = [_claim("TSMC", "supplies_to", "NVIDIA")]
    llm = [_claim("TSMC", "supplies_to", "NVIDIA", method="llm")]
    merged = merge_claims(rule, llm)
    assert len(merged) == 1
    assert merged[0].extraction_method == "hybrid"
    # The rule claim is the survivor (same key, richer typed fields).
    assert merged[0].claim_key == rule[0].claim_key


def test_rule_claims_come_first_then_llm_only() -> None:
    rule = [_claim("A", "supplies_to", "B")]
    llm = [
        _claim("A", "supplies_to", "B", method="llm"),  # corroborates → hybrid in place
        _claim("C", "competes_with", "D", method="llm"),  # llm-only → appended
    ]
    merged = merge_claims(rule, llm)
    assert [(c.subject_text, c.extraction_method) for c in merged] == [
        ("A", "hybrid"),
        ("C", "llm"),
    ]


def test_does_not_mutate_inputs() -> None:
    rule = [_claim("TSMC", "supplies_to", "NVIDIA")]
    llm = [_claim("TSMC", "supplies_to", "NVIDIA", method="llm")]
    merge_claims(rule, llm)
    assert rule[0].extraction_method == "rule"  # original untouched (replace, not mutate)


def test_empty_llm_returns_rule_claims() -> None:
    rule = [_claim("TSMC", "supplies_to", "NVIDIA")]
    assert merge_claims(rule, []) == rule


def test_empty_rule_returns_llm_claims() -> None:
    llm = [_claim("TSMC", "supplies_to", "NVIDIA", method="llm")]
    merged = merge_claims([], llm)
    assert len(merged) == 1
    assert merged[0].extraction_method == "llm"  # no rule claim to corroborate
