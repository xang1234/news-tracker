"""Extraction eval harness.

Scores any extractor against a labelled golden set by matching normalized
``(subject, predicate, object)`` triples:

- a *true positive* is an expected claim the extractor produced,
- a *false negative* is an expected claim it missed (lowers recall),
- a *false positive* is a produced claim not in the golden labels (lowers precision).

The extractor is any callable ``GoldenDocument -> list[EvidenceClaim]``, so the
same harness scores the rule extractor today and the LLM/hybrid pass later
(epic ``7th``) — that's the gate: the LLM pass must beat regex on recall
without a precision regression beyond the agreed bound.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from src.claims.schemas import EvidenceClaim
from src.eval.schemas import DocEval, ExtractionEval, GoldenClaim, GoldenDocument

Extractor = Callable[[GoldenDocument], list[EvidenceClaim]]

Triple = tuple[str, str, str]


def normalize_triple(subject: str, predicate: str, object_text: str | None) -> Triple:
    """Case/space-insensitive triple key; a missing object becomes ``""``."""
    return (
        subject.strip().lower(),
        predicate.strip().lower(),
        (object_text or "").strip().lower(),
    )


def _golden_triple(claim: GoldenClaim) -> Triple:
    return normalize_triple(claim.subject, claim.predicate, claim.object)


def _claim_triple(claim: EvidenceClaim) -> Triple:
    return normalize_triple(claim.subject_text, claim.predicate, claim.object_text)


def evaluate(golden_docs: Iterable[GoldenDocument], extractor: Extractor) -> ExtractionEval:
    """Run ``extractor`` over each golden doc and aggregate recall/precision."""
    per_doc: list[DocEval] = []
    tp = fn = fp = 0

    for doc in golden_docs:
        expected = {_golden_triple(c) for c in doc.expected_claims}
        extracted = {_claim_triple(c) for c in extractor(doc)}

        matched = expected & extracted
        missed = expected - extracted
        spurious = extracted - expected
        tp += len(matched)
        fn += len(missed)
        fp += len(spurious)

        per_doc.append(
            DocEval(
                doc_id=doc.doc_id,
                true_positives=sorted(matched),
                false_negatives=sorted(missed),
                false_positives=sorted(spurious),
            )
        )

    return ExtractionEval(
        true_positives=tp,
        false_negatives=fn,
        false_positives=fp,
        per_doc=per_doc,
    )


def rule_extractor(doc: GoldenDocument) -> list[EvidenceClaim]:
    """Adapter running the current regex/co-occurrence extractor on a golden doc."""
    # Imported lazily so the harness/schemas stay importable without the full
    # extraction stack (e.g. for pure metric tests).
    from src.claims.narrative_extractor import extract_claims_from_document

    return extract_claims_from_document(doc.doc_id, doc.events, doc.entities, doc.content)
