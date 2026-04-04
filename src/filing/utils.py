"""Shared utilities for filing providers.

Extracted from EdgarToolsProvider and SecApiProvider to avoid
duplication of deterministic ID generation, date parsing, and
filing type normalization.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime
from typing import Any

from src.filing.schemas import VALID_FILING_TYPES

logger = logging.getLogger(__name__)

# Filing type aliases for normalization
_FILING_TYPE_ALIASES: dict[str, str] = {
    "10K": "10-K",
    "10Q": "10-Q",
    "8K": "8-K",
    "DEF14A": "DEF 14A",
    "SC13D": "SC 13D",
    "SC13G": "SC 13G",
    "13F": "13F-HR",
    "FORM 4": "4",
}


def make_section_id(accession: str, index: int, name: str) -> str:
    """Generate a deterministic section ID from filing + section info."""
    key = f"{accession}:{index}:{name}"
    return f"sec_{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def parse_filing_date(value: Any) -> date:
    """Parse a date from various SEC/edgartools formats.

    Handles datetime objects, date objects, ISO strings, and None.
    Falls back to date.today() for unparseable values.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return date.today()


def normalize_filing_type(raw_type: str) -> str:
    """Normalize a raw SEC form type to our canonical set.

    Handles variations like '10-K/A' (amended), case differences,
    and common aliases. Unrecognized types default to '8-K' with
    a warning logged.
    """
    cleaned = raw_type.strip().upper()
    if cleaned.endswith("/A"):
        cleaned = cleaned[:-2]
    if cleaned in VALID_FILING_TYPES:
        return cleaned
    if cleaned in _FILING_TYPE_ALIASES:
        return _FILING_TYPE_ALIASES[cleaned]
    logger.warning("Unrecognized filing type %r, defaulting to 8-K", raw_type)
    return "8-K"
