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

from src.assertions.schemas import AssertionClaimLink

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


def classify_polarity_links(
    assertion_id: str,
    positive_predicate: str,
    claims: list,
) -> list[AssertionClaimLink]:
    """Build links for an assertion keyed on ``positive_predicate``.

    Same-predicate claims ``support`` the assertion; antonym-predicate claims
    ``contradiction`` it. Claims with any other predicate are unrelated and
    get no link. Validity-window filtering is the caller's responsibility.
    """
    antonym = antonym_of(positive_predicate)
    links: list[AssertionClaimLink] = []
    for claim in claims:
        if claim.predicate == positive_predicate:
            link_type = "support"
        elif antonym is not None and claim.predicate == antonym:
            link_type = "contradiction"
        else:
            continue
        links.append(
            AssertionClaimLink(
                assertion_id=assertion_id,
                claim_id=claim.claim_id,
                link_type=link_type,
                contribution_weight=1.0,
                metadata={"detector": "predicate_polarity"},
            )
        )
    return links
