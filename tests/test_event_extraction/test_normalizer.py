"""Tests for TimeNormalizer."""

from datetime import date

from src.event_extraction.normalizer import TimeNormalizer


class TestQuarterNormalization:
    """Tests for quarter references."""

    def test_q1_with_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 15))
        assert norm.normalize("Q1 2026") == "2026-Q1"

    def test_q4_with_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 15))
        assert norm.normalize("Q4 2025") == "2025-Q4"

    def test_quarter_without_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 15))
        assert norm.normalize("Q3") == "2026-Q3"

    def test_quarter_case_insensitive(self):
        norm = TimeNormalizer(reference_date=date(2026, 1, 1))
        assert norm.normalize("q2 2026") == "2026-Q2"

    def test_all_quarters(self):
        norm = TimeNormalizer(reference_date=date(2026, 3, 1))
        for q in range(1, 5):
            assert norm.normalize(f"Q{q}") == f"2026-Q{q}"


class TestHalfNormalization:
    """Tests for half-year references."""

    def test_h1_with_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 3, 1))
        assert norm.normalize("H1 2026") == "2026-H1"

    def test_h2_without_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 3, 1))
        assert norm.normalize("H2") == "2026-H2"

    def test_h1_case_insensitive(self):
        norm = TimeNormalizer(reference_date=date(2026, 3, 1))
        assert norm.normalize("h1 2025") == "2025-H1"


class TestRelativeQuarter:
    """Tests for relative quarter references."""

    def test_next_quarter_from_q1(self):
        norm = TimeNormalizer(reference_date=date(2026, 2, 15))
        assert norm.normalize("next quarter") == "2026-Q2"

    def test_next_quarter_from_q4(self):
        norm = TimeNormalizer(reference_date=date(2026, 11, 1))
        assert norm.normalize("next quarter") == "2027-Q1"

    def test_last_quarter_from_q2(self):
        norm = TimeNormalizer(reference_date=date(2026, 5, 1))
        assert norm.normalize("last quarter") == "2026-Q1"

    def test_last_quarter_from_q1(self):
        norm = TimeNormalizer(reference_date=date(2026, 1, 15))
        assert norm.normalize("last quarter") == "2025-Q4"


class TestRelativeYear:
    """Tests for relative year references."""

    def test_next_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 1))
        assert norm.normalize("next year") == "2027"

    def test_last_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 1))
        assert norm.normalize("last year") == "2025"


class TestEndOfYear:
    """Tests for end-of-year references."""

    def test_by_end_of_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 1))
        assert norm.normalize("by end of year") == "2026-Q4"

    def test_end_of_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 1))
        assert norm.normalize("end of year") == "2026-Q4"

    def test_year_end(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 1))
        assert norm.normalize("year-end") == "2026-Q4"

    def test_by_year_end(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 1))
        assert norm.normalize("by year end") == "2026-Q4"

    def test_end_of_the_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 1))
        assert norm.normalize("end of the year") == "2026-Q4"


class TestThisYear:
    """Tests for 'this year' reference."""

    def test_this_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 1))
        assert norm.normalize("this year") == "2026"


class TestMonthYear:
    """Tests for month name references."""

    def test_full_month_with_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 1, 1))
        assert norm.normalize("January 2026") == "2026-01"
        assert norm.normalize("December 2025") == "2025-12"

    def test_abbreviated_month_with_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 1, 1))
        assert norm.normalize("Feb 2026") == "2026-02"
        assert norm.normalize("Sept 2026") == "2026-09"

    def test_month_without_year(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 1))
        assert norm.normalize("March") == "2026-03"


class TestRelativeMonth:
    """Tests for relative month references."""

    def test_next_month(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 15))
        assert norm.normalize("next month") == "2026-07"

    def test_next_month_december(self):
        norm = TimeNormalizer(reference_date=date(2026, 12, 15))
        assert norm.normalize("next month") == "2027-01"

    def test_last_month(self):
        norm = TimeNormalizer(reference_date=date(2026, 6, 15))
        assert norm.normalize("last month") == "2026-05"

    def test_last_month_january(self):
        norm = TimeNormalizer(reference_date=date(2026, 1, 15))
        assert norm.normalize("last month") == "2025-12"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_string(self):
        norm = TimeNormalizer()
        assert norm.normalize("") == ""

    def test_whitespace_only(self):
        norm = TimeNormalizer()
        assert norm.normalize("   ") == "   "

    def test_unknown_format_passthrough(self):
        norm = TimeNormalizer()
        assert norm.normalize("sometime soon") == "sometime soon"
        assert norm.normalize("TBD") == "TBD"

    def test_whitespace_stripped(self):
        norm = TimeNormalizer(reference_date=date(2026, 1, 1))
        assert norm.normalize("  Q1 2026  ") == "2026-Q1"
