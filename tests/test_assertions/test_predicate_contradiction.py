"""Tests for predicate-polarity contradiction classification.

Antonym predicates on the same subject within overlapping validity windows
contradict each other (TSMC ``expands_capacity`` vs ``constrains_capacity``).
This maps same-predicate claims to ``support`` and antonym claims to
``contradiction`` on a single assertion, so ``aggregate_assertion`` flips it
to ``disputed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from src.assertions.predicate_contradiction import (
    ANTONYM_PREDICATES,
    antonym_of,
    classify_polarity_links,
    validity_overlaps,
)
from src.assertions.schemas import AssertionClaimLink


@dataclass
class _Claim:
    claim_id: str
    predicate: str
    claim_valid_from: datetime | None = None
    claim_valid_to: datetime | None = None


class TestAntonymOf:
    def test_known_pairs_are_symmetric(self):
        assert antonym_of("expands_capacity") == "constrains_capacity"
        assert antonym_of("constrains_capacity") == "expands_capacity"
        assert antonym_of("launches_product") == "delays_product"
        assert antonym_of("delays_product") == "launches_product"

    def test_neutral_predicate_has_no_antonym(self):
        assert antonym_of("changes_pricing") is None
        assert antonym_of("supplies_to") is None

    def test_map_is_symmetric(self):
        for predicate, opposite in ANTONYM_PREDICATES.items():
            assert ANTONYM_PREDICATES[opposite] == predicate


class TestValidityOverlaps:
    def _d(self, month: int) -> datetime:
        return datetime(2026, month, 1, tzinfo=UTC)

    def test_open_windows_always_overlap(self):
        a = _Claim("a", "expands_capacity")
        b = _Claim("b", "constrains_capacity")
        assert validity_overlaps(a, b) is True

    def test_disjoint_windows_do_not_overlap(self):
        a = _Claim("a", "expands_capacity", self._d(1), self._d(3))
        b = _Claim("b", "constrains_capacity", self._d(5), self._d(7))
        assert validity_overlaps(a, b) is False

    def test_touching_windows_overlap(self):
        a = _Claim("a", "expands_capacity", self._d(1), self._d(5))
        b = _Claim("b", "constrains_capacity", self._d(3), self._d(7))
        assert validity_overlaps(a, b) is True

    def test_one_open_end_overlaps(self):
        a = _Claim("a", "expands_capacity", self._d(1), None)  # ongoing
        b = _Claim("b", "constrains_capacity", self._d(5), self._d(7))
        assert validity_overlaps(a, b) is True


class TestClassifyPolarityLinks:
    def test_same_predicate_supports_antonym_contradicts(self):
        claims = [
            _Claim("c1", "expands_capacity"),
            _Claim("c2", "constrains_capacity"),
        ]
        links = classify_polarity_links("asrt_x", "expands_capacity", claims)
        by_id = {link.claim_id: link for link in links}
        assert by_id["c1"].link_type == "support"
        assert by_id["c2"].link_type == "contradiction"
        assert all(isinstance(link, AssertionClaimLink) for link in links)
        assert all(link.assertion_id == "asrt_x" for link in links)

    def test_unrelated_predicate_is_not_linked(self):
        claims = [
            _Claim("c1", "expands_capacity"),
            _Claim("c2", "changes_pricing"),  # neither positive nor antonym
        ]
        links = classify_polarity_links("asrt_x", "expands_capacity", claims)
        assert {link.claim_id for link in links} == {"c1"}

    def test_only_support_when_no_antonym_present(self):
        claims = [_Claim("c1", "expands_capacity"), _Claim("c2", "expands_capacity")]
        links = classify_polarity_links("asrt_x", "expands_capacity", claims)
        assert {link.link_type for link in links} == {"support"}
