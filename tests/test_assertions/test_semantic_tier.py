"""Tests for the semantic contradiction tier.

The tier applies only to *residual* claims (resolved subject, non-numeric, no
antonym predicate) — the space the deterministic tiers don't cover. It asks an
injected judge whether same-subject/same-predicate claims contradict, emitting
a ``contradiction`` opinion only for high-confidence ``contradicts`` verdicts.
A fake judge is injected so no live API is touched.
"""

from __future__ import annotations

import pytest

from src.assertions.reconciliation_engine import SemanticTier
from src.assertions.semantic_judge import ContradictionVerdict
from src.claims.schemas import EvidenceClaim, make_claim_key


def _claim(
    claim_id: str,
    *,
    predicate: str = "changes_pricing",
    subject_concept_id: str | None = "concept_tsmc",
    source_text: str | None = None,
    numeric_value: float | None = None,
    metric: str | None = None,
) -> EvidenceClaim:
    key = make_claim_key("narrative", claim_id, "TSMC", predicate, "")
    return EvidenceClaim(
        claim_id=claim_id,
        claim_key=key,
        lane="narrative",
        source_id=claim_id,
        predicate=predicate,
        subject_text="TSMC",
        subject_concept_id=subject_concept_id,
        source_text=source_text,
        numeric_value=numeric_value,
        metric=metric,
    )


class FakeJudge:
    """Returns a canned verdict keyed by the second (candidate) text."""

    def __init__(self, by_candidate: dict[str, ContradictionVerdict]) -> None:
        self._by_candidate = by_candidate
        self.calls: list[tuple[str, str]] = []

    async def judge(self, text_a: str, text_b: str) -> ContradictionVerdict | None:
        self.calls.append((text_a, text_b))
        return self._by_candidate.get(text_b)


class TestAppliesTo:
    def test_residual_relationship_claim_applies(self):
        tier = SemanticTier(FakeJudge({}))
        assert tier.applies_to(_claim("c1", predicate="changes_pricing")) is True

    def test_numeric_claim_does_not_apply(self):
        tier = SemanticTier(FakeJudge({}))
        assert tier.applies_to(_claim("c1", numeric_value=42.0, metric="price")) is False

    def test_antonym_predicate_does_not_apply(self):
        tier = SemanticTier(FakeJudge({}))
        assert tier.applies_to(_claim("c1", predicate="expands_capacity")) is False

    def test_unresolved_subject_does_not_apply(self):
        tier = SemanticTier(FakeJudge({}))
        assert tier.applies_to(_claim("c1", subject_concept_id=None)) is False


@pytest.mark.asyncio
async def test_high_confidence_contradiction_is_flagged():
    incoming = _claim("c1", source_text="TSMC will raise wafer prices")
    other = _claim("c2", source_text="TSMC will cut wafer prices")
    judge = FakeJudge({"TSMC will cut wafer prices": ContradictionVerdict("contradicts", 0.9)})
    tier = SemanticTier(judge, confidence_threshold=0.7)

    opinions = await tier.classify(incoming, [incoming, other])

    assert opinions == {"c2": "contradiction"}


@pytest.mark.asyncio
async def test_low_confidence_is_dropped():
    incoming = _claim("c1", source_text="A")
    other = _claim("c2", source_text="B")
    judge = FakeJudge({"B": ContradictionVerdict("contradicts", 0.5)})
    tier = SemanticTier(judge, confidence_threshold=0.7)

    opinions = await tier.classify(incoming, [incoming, other])

    assert opinions == {}


@pytest.mark.asyncio
async def test_agrees_and_unrelated_emit_no_opinion():
    incoming = _claim("c1", source_text="A")
    agree = _claim("c2", source_text="B")
    unrel = _claim("c3", source_text="C")
    judge = FakeJudge(
        {
            "B": ContradictionVerdict("agrees", 0.95),
            "C": ContradictionVerdict("unrelated", 0.95),
        }
    )
    tier = SemanticTier(judge, confidence_threshold=0.7)

    opinions = await tier.classify(incoming, [incoming, agree, unrel])

    assert opinions == {}


@pytest.mark.asyncio
async def test_incoming_claim_is_not_compared_to_itself():
    incoming = _claim("c1", source_text="A")
    judge = FakeJudge({})
    tier = SemanticTier(judge)

    await tier.classify(incoming, [incoming])

    assert judge.calls == []


@pytest.mark.asyncio
async def test_candidate_count_is_bounded():
    incoming = _claim("c1", source_text="A")
    others = [_claim(f"c{i}", source_text=f"t{i}") for i in range(2, 12)]  # 10 candidates
    judge = FakeJudge({})
    tier = SemanticTier(judge, max_candidates=3)

    await tier.classify(incoming, [incoming, *others])

    assert len(judge.calls) == 3
