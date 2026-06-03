"""Predicate-polarity contradiction classification.

Some event predicates are natural antonyms: a subject cannot, in the same
period, both ``expands_capacity`` and ``constrains_capacity``. When such a
pair appears on the same subject with overlapping validity windows, the
antonym is evidence *against* the assertion.

This maps claims onto support/contradiction links for a given assertion's
``positive`` predicate: same-predicate claims ``support``, antonym-predicate
claims ``contradiction``, everything else is unrelated (no link). Feeding
those links to ``aggregate_assertion`` flips the assertion to ``disputed``.

Stateless pure functions — no I/O. The caller handles persistence.
"""

from __future__ import annotations

# Antonym predicate pairs (symmetric). Predicates without a clear polarity
# (changes_pricing, revises_guidance, relationship predicates) are absent and
# therefore never trigger polarity contradiction.
ANTONYM_PREDICATES: dict[str, str] = {
    "expands_capacity": "constrains_capacity",
    "constrains_capacity": "expands_capacity",
    "launches_product": "delays_product",
    "delays_product": "launches_product",
}


def antonym_of(predicate: str) -> str | None:
    """Return the antonym predicate, or None if the predicate has no polarity."""
    return ANTONYM_PREDICATES.get(predicate)


def validity_overlaps(a, b) -> bool:
    """Whether two claims' validity windows overlap.

    A ``None`` bound is open (``valid_from=None`` → since forever,
    ``valid_to=None`` → still valid). Two windows overlap unless one ends
    strictly before the other begins.
    """
    a_from, a_to = a.claim_valid_from, a.claim_valid_to
    b_from, b_to = b.claim_valid_from, b.claim_valid_to
    a_ends_before_b = a_to is not None and b_from is not None and a_to < b_from
    b_ends_before_a = b_to is not None and a_from is not None and b_to < a_from
    return not (a_ends_before_b or b_ends_before_a)


def polarity_link_types(positive_predicate: str, claims: list) -> dict[str, str]:
    """Classify claims as ``support``/``contradiction`` for ``positive_predicate``.

    Same-predicate claims ``support``; antonym-predicate claims
    ``contradiction``. Claims with any other predicate are unrelated and get no
    entry. Validity-window filtering is the caller's responsibility. Mirrors
    ``numeric_link_types`` — a pure opinion map, no link objects.
    """
    antonym = antonym_of(positive_predicate)
    opinions: dict[str, str] = {}
    for claim in claims:
        if claim.predicate == positive_predicate:
            opinions[claim.claim_id] = "support"
        elif antonym is not None and claim.predicate == antonym:
            opinions[claim.claim_id] = "contradiction"
    return opinions
