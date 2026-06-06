"""CI-runnable extraction eval against the checked-in golden set.

This is the gate (news-tracker-7th.1): it pins the current regex extractor's
recall/precision so a regression fails CI, and it's the same harness the LLM
pass (7th.2) must beat on recall without dropping precision below the floor.
"""

from __future__ import annotations

from src.eval.extraction import evaluate, rule_extractor
from src.eval.golden import load_golden_set

# Floors below the current rule-extractor scores (recall 0.70 / precision 1.00),
# with margin for benign label edits. The LLM pass must clear the recall floor
# by a wide margin and must NOT drop precision below this.
MIN_RECALL = 0.6
MIN_PRECISION = 0.9


def test_golden_set_loads() -> None:
    docs = load_golden_set()
    assert len(docs) >= 5
    assert sum(len(d.expected_claims) for d in docs) >= 8
    # Every document carries inputs and at least one label.
    for d in docs:
        assert d.content
        assert d.expected_claims


def test_rule_extractor_meets_floor() -> None:
    docs = load_golden_set()
    report = evaluate(docs, rule_extractor)
    assert report.recall >= MIN_RECALL, f"recall {report.recall:.3f} below floor {MIN_RECALL}"
    assert report.precision >= MIN_PRECISION, (
        f"precision {report.precision:.3f} below floor {MIN_PRECISION}"
    )


def test_eval_is_reproducible() -> None:
    docs = load_golden_set()
    a = evaluate(docs, rule_extractor)
    b = evaluate(docs, rule_extractor)
    assert (a.recall, a.precision, a.true_positives) == (b.recall, b.precision, b.true_positives)


def test_golden_set_has_recall_headroom_for_llm() -> None:
    # The golden set must contain claims the regex extractor misses, otherwise
    # it can't measure the LLM pass's recall lift (the whole point of the epic).
    docs = load_golden_set()
    report = evaluate(docs, rule_extractor)
    assert report.false_negatives > 0
