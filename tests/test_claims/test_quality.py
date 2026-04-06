"""Tests for claim quality checks, dead-letter handling, and quarantine.

Verifies that the claim pipeline can fail safely: structural issues
cause dead-letter, semantic/safety issues cause quarantine, and
every failure path preserves enough context for replay.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.claims.quality import (
    VALID_DL_REASONS,
    CheckCode,
    DeadLetterRecord,
    Disposition,
    capture_dead_letter,
    make_dead_letter_id,
    run_quality_checks,
    verdict_to_dead_letter,
)
from src.claims.schemas import EvidenceClaim

MIGRATION_PATH = Path("migrations/025_claim_dead_letters.sql")


# -- Helpers ---------------------------------------------------------------


def _make_claim(**overrides) -> EvidenceClaim:
    """Build a valid EvidenceClaim with sensible defaults."""
    defaults = {
        "claim_id": "claim_test_001",
        "claim_key": "clk_test_001",
        "lane": "narrative",
        "source_id": "doc_1",
        "predicate": "supplies_to",
        "subject_text": "TSMC",
        "confidence": 0.7,
        "contract_version": "0.1.0",
        "extraction_method": "rule",
    }
    defaults.update(overrides)
    return EvidenceClaim(**defaults)


def _make_raw_claim(**overrides) -> EvidenceClaim:
    """Build a claim then patch fields, bypassing __post_init__ validation.

    Use this for quality-check tests that need structurally invalid
    claims (empty source_id, blank predicate, etc.) which the
    EvidenceClaim constructor would reject.
    """
    # Build a valid claim first, then override fields
    claim = _make_claim()
    for key, value in overrides.items():
        object.__setattr__(claim, key, value)
    return claim


# -- Quality check: accept -------------------------------------------------


class TestQualityAccept:
    """Claims that pass all quality checks."""

    def test_valid_claim_accepted(self) -> None:
        claim = _make_claim()
        verdict = run_quality_checks(claim)
        assert verdict.passed
        assert verdict.disposition == Disposition.ACCEPT
        assert verdict.checks_failed == []

    def test_claim_with_all_fields(self) -> None:
        claim = _make_claim(
            object_text="NVIDIA",
            subject_concept_id="concept_tsmc",
            object_concept_id="concept_nvda",
            source_text="TSMC supplies to NVIDIA",
            source_span_start=0,
            source_span_end=23,
        )
        verdict = run_quality_checks(claim)
        assert verdict.passed

    def test_verdict_has_claim_id(self) -> None:
        claim = _make_claim(claim_id="claim_xyz")
        verdict = run_quality_checks(claim)
        assert verdict.claim_id == "claim_xyz"


# -- Quality check: dead-letter (structural) --------------------------------


class TestQualityDeadLetter:
    """Structural failures that produce dead-letter verdicts."""

    def test_empty_subject_text(self) -> None:
        claim = _make_claim(subject_text="")
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.DEAD_LETTER
        assert CheckCode.EMPTY_SUBJECT in verdict.checks_failed

    def test_whitespace_subject_text(self) -> None:
        claim = _make_claim(subject_text="   ")
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.DEAD_LETTER
        assert CheckCode.EMPTY_SUBJECT in verdict.checks_failed

    def test_empty_predicate(self) -> None:
        claim = _make_raw_claim(predicate="   ")
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.DEAD_LETTER
        assert CheckCode.EMPTY_PREDICATE in verdict.checks_failed

    def test_missing_source_id(self) -> None:
        claim = _make_raw_claim(source_id="")
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.DEAD_LETTER
        assert CheckCode.MISSING_SOURCE_ID in verdict.checks_failed

    def test_missing_contract_version(self) -> None:
        claim = _make_raw_claim(contract_version="")
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.DEAD_LETTER
        assert CheckCode.MISSING_CONTRACT_VERSION in verdict.checks_failed

    def test_inverted_span(self) -> None:
        claim = _make_claim(
            source_span_start=100, source_span_end=50
        )
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.DEAD_LETTER
        assert CheckCode.INVALID_SPAN in verdict.checks_failed

    def test_multiple_structural_failures(self) -> None:
        """Multiple structural issues are all reported."""
        claim = _make_raw_claim(
            source_id="",
            source_span_start=100,
            source_span_end=50,
        )
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.DEAD_LETTER
        assert len(verdict.checks_failed) >= 2

    def test_dead_letter_takes_priority_over_quarantine(self) -> None:
        """Structural failures override semantic concerns."""
        claim = _make_raw_claim(
            source_id="",  # Dead letter
            confidence=0.001,  # Would quarantine
        )
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.DEAD_LETTER


# -- Quality check: quarantine (semantic/safety) ----------------------------


class TestQualityQuarantine:
    """Semantic/safety issues that produce quarantine verdicts."""

    def test_self_referencing(self) -> None:
        claim = _make_claim(
            subject_concept_id="concept_same",
            object_concept_id="concept_same",
        )
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.QUARANTINE
        assert CheckCode.SELF_REFERENCING in verdict.checks_failed

    def test_confidence_below_floor(self) -> None:
        claim = _make_claim(confidence=0.01)
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.QUARANTINE
        assert CheckCode.CONFIDENCE_BELOW_FLOOR in verdict.checks_failed

    def test_confidence_at_floor_passes(self) -> None:
        claim = _make_claim(confidence=0.05)
        verdict = run_quality_checks(claim)
        assert verdict.passed

    def test_custom_confidence_floor(self) -> None:
        claim = _make_claim(confidence=0.3)
        verdict = run_quality_checks(claim, confidence_floor=0.5)
        assert verdict.disposition == Disposition.QUARANTINE

    def test_source_text_too_long(self) -> None:
        claim = _make_claim(source_text="x" * 60_000)
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.QUARANTINE
        assert CheckCode.SOURCE_TEXT_TOO_LONG in verdict.checks_failed

    def test_custom_source_text_limit(self) -> None:
        claim = _make_claim(source_text="x" * 200)
        verdict = run_quality_checks(
            claim, max_source_text_length=100
        )
        assert verdict.disposition == Disposition.QUARANTINE

    def test_metadata_too_large(self) -> None:
        big_meta = {"data": "x" * 200_000}
        claim = _make_claim(metadata=big_meta)
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.QUARANTINE
        assert CheckCode.METADATA_TOO_LARGE in verdict.checks_failed

    def test_no_self_reference_when_concepts_differ(self) -> None:
        claim = _make_claim(
            subject_concept_id="concept_a",
            object_concept_id="concept_b",
        )
        verdict = run_quality_checks(claim)
        assert verdict.passed

    def test_no_self_reference_when_concepts_unresolved(self) -> None:
        claim = _make_claim(
            subject_concept_id=None,
            object_concept_id=None,
        )
        verdict = run_quality_checks(claim)
        assert verdict.passed


# -- Dead-letter record tests -----------------------------------------------


class TestDeadLetterRecord:
    """DeadLetterRecord dataclass and builder."""

    def test_valid_construction(self) -> None:
        dl = DeadLetterRecord(
            record_id="dl_test",
            lane="narrative",
            run_id="run_1",
            source_id="doc_1",
            reason="extraction_error",
            error_message="Parser failed",
        )
        assert dl.reason == "extraction_error"

    def test_invalid_reason(self) -> None:
        with pytest.raises(ValueError, match="Invalid dead-letter reason"):
            DeadLetterRecord(
                record_id="dl_test",
                lane="narrative",
                run_id="run_1",
                source_id="doc_1",
                reason="unknown_reason",
                error_message="Something",
            )

    def test_all_reasons_accepted(self) -> None:
        for reason in VALID_DL_REASONS:
            dl = DeadLetterRecord(
                record_id=f"dl_{reason}",
                lane="narrative",
                run_id="run_1",
                source_id="doc_1",
                reason=reason,
                error_message="Test",
            )
            assert dl.reason == reason


class TestCaptureDeadLetter:
    """capture_dead_letter builder function."""

    def test_without_claim(self) -> None:
        dl = capture_dead_letter(
            lane="narrative",
            run_id="run_1",
            source_id="doc_1",
            reason="extraction_error",
            error_message="Timeout during extraction",
        )
        assert dl.record_id.startswith("dl_")
        assert dl.claim_snapshot is None
        assert dl.reason == "extraction_error"

    def test_with_claim(self) -> None:
        claim = _make_claim()
        dl = capture_dead_letter(
            lane="narrative",
            run_id="run_1",
            source_id="doc_1",
            reason="quality_check_failed",
            error_message="Empty subject",
            claim=claim,
        )
        assert dl.claim_snapshot is not None
        assert dl.claim_snapshot["claim_id"] == "claim_test_001"
        assert dl.claim_snapshot["predicate"] == "supplies_to"

    def test_deterministic_id(self) -> None:
        dl1 = capture_dead_letter(
            lane="narrative",
            run_id="run_1",
            source_id="doc_1",
            reason="extraction_error",
            error_message="Same error",
        )
        dl2 = capture_dead_letter(
            lane="narrative",
            run_id="run_1",
            source_id="doc_1",
            reason="extraction_error",
            error_message="Same error",
        )
        assert dl1.record_id == dl2.record_id

    def test_different_errors_different_ids(self) -> None:
        dl1 = capture_dead_letter(
            lane="narrative",
            run_id="run_1",
            source_id="doc_1",
            reason="extraction_error",
            error_message="Error A",
        )
        dl2 = capture_dead_letter(
            lane="narrative",
            run_id="run_1",
            source_id="doc_1",
            reason="extraction_error",
            error_message="Error B",
        )
        assert dl1.record_id != dl2.record_id

    def test_preserves_source_text(self) -> None:
        dl = capture_dead_letter(
            lane="filing",
            run_id="run_2",
            source_id="filing_1",
            reason="parse_error",
            error_message="Malformed section",
            source_text="Some raw filing text...",
        )
        assert dl.source_text == "Some raw filing text..."


class TestVerdictToDeadLetter:
    """Converting failed verdicts to dead-letter records."""

    def test_converts_dead_letter_verdict(self) -> None:
        claim = _make_raw_claim(source_id="")
        verdict = run_quality_checks(claim)
        assert verdict.disposition == Disposition.DEAD_LETTER

        dl = verdict_to_dead_letter(
            verdict, claim=claim, run_id="run_1"
        )
        assert dl.reason == "quality_check_failed"
        assert dl.claim_snapshot is not None
        assert "checks_failed" in dl.error_detail
        assert dl.error_detail["disposition"] == "dead_letter"

    def test_preserves_source_text(self) -> None:
        claim = _make_raw_claim(source_id="")
        verdict = run_quality_checks(claim)
        dl = verdict_to_dead_letter(
            verdict,
            claim=claim,
            run_id="run_1",
            source_text="Original text for replay",
        )
        assert dl.source_text == "Original text for replay"


# -- Deterministic ID tests ------------------------------------------------


class TestMakeDeadLetterId:
    """Dead-letter ID generation."""

    def test_deterministic(self) -> None:
        id1 = make_dead_letter_id("narrative", "run_1", "doc_1", "abc")
        id2 = make_dead_letter_id("narrative", "run_1", "doc_1", "abc")
        assert id1 == id2

    def test_prefix(self) -> None:
        dl_id = make_dead_letter_id("narrative", "run_1", "doc_1", "abc")
        assert dl_id.startswith("dl_")

    def test_different_inputs(self) -> None:
        id1 = make_dead_letter_id("narrative", "run_1", "doc_1", "abc")
        id2 = make_dead_letter_id("narrative", "run_1", "doc_2", "abc")
        assert id1 != id2


# -- Migration structural tests -------------------------------------------


class TestMigration025:
    """Structural validation of migration 025."""

    @pytest.fixture(autouse=True)
    def _load_sql(self) -> None:
        self.sql = MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_dead_letters_table(self) -> None:
        assert "CREATE TABLE IF NOT EXISTS news_intel.claim_dead_letters" in self.sql

    def test_lane_check(self) -> None:
        for lane in ("narrative", "filing", "structural", "backtest"):
            assert lane in self.sql

    def test_reason_check(self) -> None:
        for reason in VALID_DL_REASONS:
            assert reason in self.sql, f"Missing reason {reason!r}"

    def test_run_index(self) -> None:
        assert "idx_claim_dl_run" in self.sql

    def test_lane_reason_index(self) -> None:
        assert "idx_claim_dl_lane_reason" in self.sql

    def test_source_index(self) -> None:
        assert "idx_claim_dl_source" in self.sql

    def test_jsonb_columns(self) -> None:
        for col in ("error_detail", "claim_snapshot", "metadata"):
            assert col in self.sql

    def test_source_text_column(self) -> None:
        assert "source_text" in self.sql
