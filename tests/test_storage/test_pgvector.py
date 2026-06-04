"""Tests for the canonical pgvector wire-format helper."""

from __future__ import annotations

from src.storage.pgvector import to_pgvector_literal


def test_formats_vector_as_bracketed_csv() -> None:
    assert to_pgvector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def test_single_element() -> None:
    assert to_pgvector_literal([1.0]) == "[1.0]"


def test_empty_vector() -> None:
    assert to_pgvector_literal([]) == "[]"
