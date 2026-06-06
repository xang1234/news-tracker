"""Tests for the extraction eval harness.

Pure scoring logic: given a golden set and an extractor callable, compute
recall/precision by matching normalized (subject, predicate, object) triples.
Uses synthetic extractors so the metric math is verified independently of any
real extractor.
"""

from __future__ import annotations

from src.claims.schemas import EvidenceClaim, make_claim_id, make_claim_key
from src.eval.extraction import evaluate, normalize_triple
from src.eval.schemas import GoldenClaim, GoldenDocument


def _claim(subject: str, predicate: str, object_text: str | None = None) -> EvidenceClaim:
    key = make_claim_key("narrative", "d", subject, predicate, object_text)
    return EvidenceClaim(
        claim_id=make_claim_id(key),
        claim_key=key,
        lane="narrative",
        source_id="d",
        subject_text=subject,
        predicate=predicate,
        object_text=object_text,
        contract_version="v1",
    )


def _doc(expected: list[GoldenClaim], doc_id: str = "d1") -> GoldenDocument:
    return GoldenDocument(doc_id=doc_id, content="", expected_claims=expected)


def _extractor(claims_by_doc: dict[str, list[EvidenceClaim]]):
    return lambda doc: claims_by_doc.get(doc.doc_id, [])


def test_normalize_triple_is_case_and_space_insensitive() -> None:
    assert normalize_triple(" TSMC ", "Supplies_To", "NVIDIA") == normalize_triple(
        "tsmc", "supplies_to", " nvidia "
    )
    # Missing object normalizes to empty string on both sides.
    assert normalize_triple("TSMC", "expands_capacity", None) == ("tsmc", "expands_capacity", "")


def test_perfect_extraction() -> None:
    doc = _doc([GoldenClaim("TSMC", "supplies_to", "NVIDIA")])
    report = evaluate([doc], _extractor({"d1": [_claim("TSMC", "supplies_to", "NVIDIA")]}))
    assert report.recall == 1.0
    assert report.precision == 1.0
    assert report.f1 == 1.0
    assert report.true_positives == 1


def test_missed_claim_lowers_recall() -> None:
    doc = _doc(
        [
            GoldenClaim("TSMC", "supplies_to", "NVIDIA"),
            GoldenClaim("Samsung", "expands_capacity"),
        ]
    )
    # Extractor finds only the first.
    report = evaluate([doc], _extractor({"d1": [_claim("TSMC", "supplies_to", "NVIDIA")]}))
    assert report.recall == 0.5
    assert report.precision == 1.0
    assert report.false_negatives == 1


def test_spurious_claim_lowers_precision() -> None:
    doc = _doc([GoldenClaim("TSMC", "supplies_to", "NVIDIA")])
    report = evaluate(
        [doc],
        _extractor(
            {
                "d1": [
                    _claim("TSMC", "supplies_to", "NVIDIA"),
                    _claim("Intel", "competes_with", "AMD"),  # not in golden
                ]
            }
        ),
    )
    assert report.recall == 1.0
    assert report.precision == 0.5
    assert report.false_positives == 1


def test_case_insensitive_matching() -> None:
    doc = _doc([GoldenClaim("TSMC", "supplies_to", "NVIDIA")])
    report = evaluate([doc], _extractor({"d1": [_claim("tsmc", "supplies_to", "nvidia")]}))
    assert report.true_positives == 1
    assert report.recall == 1.0


def test_object_none_matches_object_none() -> None:
    doc = _doc([GoldenClaim("Samsung", "expands_capacity", None)])
    report = evaluate([doc], _extractor({"d1": [_claim("Samsung", "expands_capacity", None)]}))
    assert report.true_positives == 1


def test_aggregates_across_documents() -> None:
    docs = [
        _doc([GoldenClaim("TSMC", "supplies_to", "NVIDIA")], "d1"),
        _doc([GoldenClaim("Samsung", "expands_capacity")], "d2"),
    ]
    report = evaluate(
        docs,
        _extractor(
            {
                "d1": [_claim("TSMC", "supplies_to", "NVIDIA")],
                "d2": [],  # missed
            }
        ),
    )
    assert report.total_expected == 2
    assert report.true_positives == 1
    assert report.false_negatives == 1
    assert report.recall == 0.5
    assert len(report.per_doc) == 2


def test_empty_golden_set_is_perfect_score() -> None:
    report = evaluate([], _extractor({}))
    assert report.recall == 1.0
    assert report.precision == 1.0


def test_per_doc_detail_records_matches() -> None:
    doc = _doc(
        [
            GoldenClaim("TSMC", "supplies_to", "NVIDIA"),
            GoldenClaim("Intel", "delays_product", "GPU"),
        ]
    )
    report = evaluate(
        [doc],
        _extractor(
            {
                "d1": [
                    _claim("TSMC", "supplies_to", "NVIDIA"),
                    _claim("AMD", "launches_product", "MI400"),
                ]
            }
        ),
    )
    detail = report.per_doc[0]
    assert ("tsmc", "supplies_to", "nvidia") in detail.true_positives
    assert ("intel", "delays_product", "gpu") in detail.false_negatives
    assert ("amd", "launches_product", "mi400") in detail.false_positives
