"""Time reference normalizer for event extraction.

Converts relative and informal time references from financial text
into normalized ISO-style date strings for downstream processing.
"""

from __future__ import annotations

import re
from datetime import date


# Month name → number mapping
_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Quarter → month ranges
_QUARTER_START: dict[int, int] = {1: 1, 2: 4, 3: 7, 4: 10}
_QUARTER_END: dict[int, int] = {1: 3, 2: 6, 3: 9, 4: 12}

# Half → month ranges
_HALF_START: dict[int, int] = {1: 1, 2: 7}
_HALF_END: dict[int, int] = {1: 6, 2: 12}


class TimeNormalizer:
    """
    Stateless normalizer for time references in financial text.

    Converts informal references like "Q3 2026", "next quarter", "H1",
    "by end of year" into ISO-style strings.

    Args:
        reference_date: Base date for resolving relative references.
            Defaults to today.
    """

    def __init__(self, reference_date: date | None = None):
        self._ref = reference_date or date.today()

    def normalize(self, time_ref: str) -> str:
        """
        Normalize a time reference string.

        Args:
            time_ref: Raw time reference from text.

        Returns:
            Normalized ISO-style string, or the original string if
            no pattern matches.
        """
        if not time_ref or not time_ref.strip():
            return time_ref

        text = time_ref.strip()

        # Try each normalizer in priority order
        for fn in (
            self._try_quarter,
            self._try_half,
            self._try_relative_quarter,
            self._try_relative_year,
            self._try_end_of_year,
            self._try_this_year,
            self._try_month_year,
            self._try_relative_month,
        ):
            result = fn(text)
            if result is not None:
                return result

        # Passthrough for unknown formats
        return text

    def _try_quarter(self, text: str) -> str | None:
        """Match Q1-Q4 with optional year: 'Q3 2026', 'Q1', 'q2 2025'."""
        m = re.match(r"(?i)^Q([1-4])[\s,]*(\d{4})?$", text)
        if not m:
            return None
        q = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else self._ref.year
        return f"{year}-Q{q}"

    def _try_half(self, text: str) -> str | None:
        """Match H1/H2 with optional year: 'H1 2026', 'H2'."""
        m = re.match(r"(?i)^H([12])[\s,]*(\d{4})?$", text)
        if not m:
            return None
        h = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else self._ref.year
        return f"{year}-H{h}"

    def _try_relative_quarter(self, text: str) -> str | None:
        """Match 'next quarter', 'last quarter'."""
        lower = text.lower()
        if lower == "next quarter":
            q = (self._ref.month - 1) // 3 + 1
            year = self._ref.year
            q += 1
            if q > 4:
                q = 1
                year += 1
            return f"{year}-Q{q}"
        if lower == "last quarter":
            q = (self._ref.month - 1) // 3 + 1
            year = self._ref.year
            q -= 1
            if q < 1:
                q = 4
                year -= 1
            return f"{year}-Q{q}"
        return None

    def _try_relative_year(self, text: str) -> str | None:
        """Match 'next year', 'last year'."""
        lower = text.lower()
        if lower == "next year":
            return str(self._ref.year + 1)
        if lower == "last year":
            return str(self._ref.year - 1)
        return None

    def _try_end_of_year(self, text: str) -> str | None:
        """Match 'by end of year', 'end of year', 'year-end', 'by year end'."""
        lower = text.lower().strip()
        if re.match(
            r"^(by\s+)?(end\s+of\s+(the\s+)?year|year[\s-]?end)$", lower
        ):
            return f"{self._ref.year}-Q4"
        return None

    def _try_this_year(self, text: str) -> str | None:
        """Match 'this year'."""
        if text.lower().strip() == "this year":
            return str(self._ref.year)
        return None

    def _try_month_year(self, text: str) -> str | None:
        """Match 'January 2026', 'Feb 2025', 'March'."""
        m = re.match(r"(?i)^([a-z]+)[\s,]*(\d{4})?$", text)
        if not m:
            return None
        month_name = m.group(1).lower()
        if month_name not in _MONTH_MAP:
            return None
        month = _MONTH_MAP[month_name]
        year = int(m.group(2)) if m.group(2) else self._ref.year
        return f"{year}-{month:02d}"

    def _try_relative_month(self, text: str) -> str | None:
        """Match 'next month', 'last month'."""
        lower = text.lower()
        if lower == "next month":
            month = self._ref.month + 1
            year = self._ref.year
            if month > 12:
                month = 1
                year += 1
            return f"{year}-{month:02d}"
        if lower == "last month":
            month = self._ref.month - 1
            year = self._ref.year
            if month < 1:
                month = 12
                year -= 1
            return f"{year}-{month:02d}"
        return None
