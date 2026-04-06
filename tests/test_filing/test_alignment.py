"""Tests for section alignment, diff logic, and regression corpus.

Validates:
1. Section name normalization handles real-world variation.
2. Alignment matches sections by fuzzy name similarity.
3. Diffs capture content changes at the line level.
4. Full comparison produces correct summary statistics.
5. Regression corpus is loadable and structurally valid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.filing.alignment import (
    FilingComparison,
    SectionAlignment,
    align_sections,
    compare_filings,
    diff_aligned_sections,
    normalize_section_name,
)
from src.filing.persistence import FilingSectionRecord

CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "filing" / "data" / "regression_corpus.json"
)


# -- Helpers ---------------------------------------------------------------


def _section(
    name: str,
    content: str = "",
    index: int = 0,
    accession: str = "acc-001",
) -> FilingSectionRecord:
    return FilingSectionRecord(
        section_id=f"sec_{index}_{name[:8]}",
        accession_number=accession,
        section_index=index,
        section_name=name,
        content=content,
        word_count=len(content.split()) if content else 0,
        content_hash=f"hash_{name[:8]}",
    )


# -- Normalization tests ---------------------------------------------------


class TestNormalizeSectionName:
    """normalize_section_name handles real-world filing section names."""

    def test_simple_lowercase(self) -> None:
        assert normalize_section_name("Risk Factors") == "risk factors"

    def test_strips_item_number(self) -> None:
        assert normalize_section_name("Item 1A. Risk Factors") == "risk factors"

    def test_strips_item_number_no_dot(self) -> None:
        assert normalize_section_name("Item 1A Risk Factors") == "risk factors"

    def test_strips_part_number(self) -> None:
        assert normalize_section_name("Part I") == ""  # just a part header

    def test_collapses_whitespace(self) -> None:
        assert normalize_section_name("Risk   Factors") == "risk factors"

    def test_uppercase(self) -> None:
        assert normalize_section_name("RISK FACTORS") == "risk factors"

    def test_strips_leading_trailing(self) -> None:
        assert normalize_section_name("  Risk Factors  ") == "risk factors"

    def test_complex_real_world(self) -> None:
        result = normalize_section_name("Item 7. Management's Discussion and Analysis")
        assert "management" in result
        assert "discussion" in result


# -- Alignment tests -------------------------------------------------------


class TestAlignSections:
    """align_sections fuzzy-matches sections between filings."""

    def test_exact_name_match(self) -> None:
        base = [_section("Risk Factors", index=0)]
        target = [_section("Risk Factors", index=0, accession="acc-002")]
        alignments = align_sections(base, target)
        assert len(alignments) == 1
        assert alignments[0].is_matched
        assert alignments[0].similarity == 1.0

    def test_fuzzy_name_match(self) -> None:
        base = [_section("Item 1A. Risk Factors")]
        target = [_section("RISK FACTORS", accession="acc-002")]
        alignments = align_sections(base, target)
        matched = [a for a in alignments if a.is_matched]
        assert len(matched) == 1
        assert matched[0].similarity > 0.6

    def test_removed_section(self) -> None:
        base = [_section("Risk Factors"), _section("Properties", index=1)]
        target = [_section("Risk Factors", accession="acc-002")]
        alignments = align_sections(base, target)
        removed = [a for a in alignments if a.is_removed]
        assert len(removed) == 1
        assert "properties" in removed[0].normalized_name

    def test_added_section(self) -> None:
        base = [_section("Risk Factors")]
        target = [
            _section("Risk Factors", accession="acc-002"),
            _section("New Section", index=1, accession="acc-002"),
        ]
        alignments = align_sections(base, target)
        added = [a for a in alignments if a.is_added]
        assert len(added) == 1

    def test_multiple_sections_aligned(self) -> None:
        base = [
            _section("Business", index=0),
            _section("Risk Factors", index=1),
            _section("MD&A", index=2),
        ]
        target = [
            _section("Business", index=0, accession="acc-002"),
            _section("Risk Factors", index=1, accession="acc-002"),
            _section("MD&A", index=2, accession="acc-002"),
        ]
        alignments = align_sections(base, target)
        matched = [a for a in alignments if a.is_matched]
        assert len(matched) == 3

    def test_empty_sections(self) -> None:
        alignments = align_sections([], [])
        assert alignments == []

    def test_no_base_all_added(self) -> None:
        target = [_section("New", accession="acc-002")]
        alignments = align_sections([], target)
        assert len(alignments) == 1
        assert alignments[0].is_added

    def test_no_target_all_removed(self) -> None:
        base = [_section("Old")]
        alignments = align_sections(base, [])
        assert len(alignments) == 1
        assert alignments[0].is_removed

    def test_low_similarity_not_matched(self) -> None:
        base = [_section("Risk Factors")]
        target = [_section("Financial Statements", accession="acc-002")]
        alignments = align_sections(base, target, min_similarity=0.8)
        matched = [a for a in alignments if a.is_matched]
        assert len(matched) == 0


# -- Diff tests ------------------------------------------------------------


class TestDiffAlignedSections:
    """diff_aligned_sections computes content-level diffs."""

    def test_identical_content(self) -> None:
        base = _section("Risk Factors", "We face significant risks.")
        target = _section("Risk Factors", "We face significant risks.", accession="acc-002")
        alignment = SectionAlignment(base_section=base, target_section=target, similarity=1.0)
        diffs = diff_aligned_sections([alignment])
        assert len(diffs) == 1
        assert diffs[0].content_changed is False
        assert diffs[0].diff_summary == "No changes"

    def test_content_changed(self) -> None:
        base = _section("Risk Factors", "We face risks.")
        target = _section("Risk Factors", "We face significant new risks.", accession="acc-002")
        alignment = SectionAlignment(base_section=base, target_section=target, similarity=1.0)
        diffs = diff_aligned_sections([alignment])
        assert len(diffs) == 1
        assert diffs[0].content_changed is True
        assert diffs[0].word_count_delta > 0

    def test_content_removed(self) -> None:
        base = _section("Risk Factors", "Line one.\nLine two.\nLine three.")
        target = _section("Risk Factors", "Line one.", accession="acc-002")
        alignment = SectionAlignment(base_section=base, target_section=target, similarity=1.0)
        diffs = diff_aligned_sections([alignment])
        assert diffs[0].removed_lines > 0
        assert diffs[0].word_count_delta < 0

    def test_skips_non_matched(self) -> None:
        added = SectionAlignment(base_section=None, target_section=_section("New"))
        removed = SectionAlignment(base_section=_section("Old"), target_section=None)
        diffs = diff_aligned_sections([added, removed])
        assert len(diffs) == 0

    def test_diff_ratio_range(self) -> None:
        base = _section("S", "Hello world.")
        target = _section("S", "Hello universe.", accession="acc-002")
        alignment = SectionAlignment(base_section=base, target_section=target, similarity=1.0)
        diffs = diff_aligned_sections([alignment])
        assert 0 <= diffs[0].diff_ratio <= 1


# -- Full comparison tests -------------------------------------------------


class TestCompareFilings:
    """compare_filings produces complete comparison results."""

    def test_identical_filings(self) -> None:
        sections = [
            _section("Business", "Our business.", index=0),
            _section("Risk Factors", "We face risks.", index=1),
        ]
        target_sections = [
            _section("Business", "Our business.", index=0, accession="acc-002"),
            _section("Risk Factors", "We face risks.", index=1, accession="acc-002"),
        ]
        result = compare_filings("acc-001", "acc-002", sections, target_sections)
        assert isinstance(result, FilingComparison)
        assert result.sections_changed == 0
        assert result.sections_unchanged == 2
        assert result.sections_added == 0
        assert result.sections_removed == 0

    def test_mixed_changes(self) -> None:
        base = [
            _section("Business", "Old business.", index=0),
            _section("Risk Factors", "Old risks.", index=1),
            _section("Legal Proceedings", "Gone.", index=2),
        ]
        target = [
            _section("Business", "New business description.", index=0, accession="acc-002"),
            _section("Risk Factors", "Old risks.", index=1, accession="acc-002"),
            _section("Cybersecurity", "Added.", index=2, accession="acc-002"),
        ]
        result = compare_filings("acc-001", "acc-002", base, target)
        assert result.sections_changed == 1  # Business changed
        assert result.sections_unchanged == 1  # Risk Factors same
        assert result.sections_added == 1  # Cybersecurity
        assert result.sections_removed == 1  # Legal Proceedings

    def test_empty_filings(self) -> None:
        result = compare_filings("acc-001", "acc-002", [], [])
        assert result.sections_changed == 0
        assert len(result.alignments) == 0

    def test_accession_numbers_preserved(self) -> None:
        result = compare_filings("base-acc", "target-acc", [], [])
        assert result.base_accession == "base-acc"
        assert result.target_accession == "target-acc"


# -- Regression corpus tests -----------------------------------------------


class TestRegressionCorpus:
    """Regression corpus is structurally valid and loadable."""

    @pytest.fixture()
    def corpus(self) -> dict:
        with open(CORPUS_PATH) as f:
            return json.load(f)

    def test_corpus_file_exists(self) -> None:
        assert CORPUS_PATH.exists()

    def test_corpus_has_cases(self, corpus: dict) -> None:
        assert len(corpus["cases"]) >= 5

    def test_cases_have_required_fields(self, corpus: dict) -> None:
        required = {"id", "issuer", "filing_type", "expected_sections", "known_issues"}
        for case in corpus["cases"]:
            missing = required - set(case.keys())
            assert not missing, f"Case {case.get('id', '?')} missing: {missing}"

    def test_cases_have_valid_filing_types(self, corpus: dict) -> None:
        from src.filing.schemas import VALID_FILING_TYPES

        for case in corpus["cases"]:
            assert case["filing_type"] in VALID_FILING_TYPES, (
                f"Case {case['id']} has invalid type: {case['filing_type']}"
            )

    def test_cases_cover_key_issuers(self, corpus: dict) -> None:
        issuers = {c["issuer"] for c in corpus["cases"]}
        assert "NVIDIA Corporation" in issuers
        assert "Taiwan Semiconductor Manufacturing" in issuers
        assert "Intel Corporation" in issuers

    def test_cases_cover_multiple_filing_types(self, corpus: dict) -> None:
        types = {c["filing_type"] for c in corpus["cases"]}
        assert "10-K" in types
        assert "20-F" in types
        assert "8-K" in types

    def test_cases_have_unique_ids(self, corpus: dict) -> None:
        ids = [c["id"] for c in corpus["cases"]]
        assert len(ids) == len(set(ids))

    def test_foreign_filer_cases_present(self, corpus: dict) -> None:
        foreign = [c for c in corpus["cases"] if c["filing_type"] in ("20-F", "6-K")]
        assert len(foreign) >= 2

    def test_non_narrative_cases_present(self, corpus: dict) -> None:
        non_narrative = [
            c for c in corpus["cases"] if c["filing_type"] in ("8-K", "SC 13D", "SC 13G", "4")
        ]
        assert len(non_narrative) >= 1
