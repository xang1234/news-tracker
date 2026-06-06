"""Merge rule-pass and LLM-pass claims by deterministic claim_key.

Both passes emit the same ``make_claim_key`` for the same
``(subject, predicate, object)`` triple, so the merge is a keyed union — no
double counting. A triple found by BOTH passes is kept once as the rule claim
(it carries the richer typed numeric/period fields) but re-stamped
``extraction_method="hybrid"`` to record that the LLM corroborated it;
LLM-only claims stay ``"llm"`` and rule-only claims stay ``"rule"``.
"""

from __future__ import annotations

import dataclasses

from src.claims.schemas import EvidenceClaim


def merge_claims(
    rule_claims: list[EvidenceClaim], llm_claims: list[EvidenceClaim]
) -> list[EvidenceClaim]:
    """Union rule + LLM claims by claim_key, marking corroborated triples hybrid.

    Rule claims are taken first (and win on collision); an LLM claim whose key
    matches a rule claim flips that claim to ``hybrid`` rather than adding a
    duplicate. Order is rule claims first, then LLM-only claims, in input order.
    """
    merged: dict[str, EvidenceClaim] = {}
    order: list[str] = []

    for claim in rule_claims:
        if claim.claim_key not in merged:
            merged[claim.claim_key] = claim
            order.append(claim.claim_key)

    for claim in llm_claims:
        existing = merged.get(claim.claim_key)
        if existing is None:
            merged[claim.claim_key] = claim
            order.append(claim.claim_key)
        elif existing.extraction_method == "rule":
            # Corroborated by both passes → keep the richer rule claim as hybrid.
            merged[claim.claim_key] = dataclasses.replace(existing, extraction_method="hybrid")

    return [merged[key] for key in order]
