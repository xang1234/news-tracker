"""Schemas for the extraction eval harness.

A golden document carries the extractor's *inputs* (content + the
``events_extracted`` / ``entities_mentioned`` shapes the pipeline would have
populated) alongside the *labels* (the claims a correct extractor should
produce). The harness runs any extractor over the inputs and scores its output
against the labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoldenClaim:
    """An expected claim, identified by its (subject, predicate, object) triple."""

    subject: str
    predicate: str
    object: str | None = None


@dataclass(frozen=True)
class GoldenDocument:
    """A labelled document: extractor inputs + the claims it should yield."""

    doc_id: str
    content: str
    events: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    expected_claims: list[GoldenClaim] = field(default_factory=list)


@dataclass(frozen=True)
class DocEval:
    """Per-document scoring detail (which triples matched / missed / spurious)."""

    doc_id: str
    true_positives: list[tuple[str, str, str]]
    false_negatives: list[tuple[str, str, str]]  # expected but not extracted (missed)
    false_positives: list[tuple[str, str, str]]  # extracted but not expected (spurious)


@dataclass(frozen=True)
class ExtractionEval:
    """Aggregate recall/precision of an extractor over a golden set.

    The counts are the single source of truth: ``expected = matched ∪ missed``
    and ``extracted = matched ∪ spurious`` (disjoint), so the totals are
    derived rather than stored — they can never disagree with tp/fn/fp.
    """

    true_positives: int
    false_negatives: int
    false_positives: int
    per_doc: list[DocEval] = field(default_factory=list)

    @property
    def total_expected(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def total_extracted(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return 1.0 if denom == 0 else self.true_positives / denom

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return 1.0 if denom == 0 else self.true_positives / denom

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
