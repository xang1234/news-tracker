"""Tests for the shared cited-entry grounding gate.

``parse_cited_entries`` is the single grounding gate behind both the briefing
and Q&A parsers: it keeps only entries with non-empty text that cite at least
one retrieved claim, strips invented ids, dedupes, and drops malformed input.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.citation_gate import parse_cited_entries


@dataclass(frozen=True)
class _Entry:
    text: str
    claim_ids: list[str]


def _parse(payload, valid):
    return parse_cited_entries(
        payload, valid, key="items", factory=lambda t, ids: _Entry(text=t, claim_ids=ids)
    )


def test_keeps_grounded_entry() -> None:
    out = _parse({"items": [{"text": "TSMC supplies NVIDIA.", "claim_ids": ["a"]}]}, {"a"})
    assert out == [_Entry("TSMC supplies NVIDIA.", ["a"])]


def test_strips_invented_ids() -> None:
    out = _parse({"items": [{"text": "x", "claim_ids": ["a", "ghost"]}]}, {"a"})
    assert out == [_Entry("x", ["a"])]


def test_drops_entry_left_uncited() -> None:
    out = _parse({"items": [{"text": "x", "claim_ids": ["ghost"]}]}, {"a"})
    assert out == []


def test_dedupes_preserving_order() -> None:
    out = _parse({"items": [{"text": "x", "claim_ids": ["b", "a", "b", "a"]}]}, {"a", "b"})
    assert out == [_Entry("x", ["b", "a"])]


def test_blank_text_dropped() -> None:
    assert _parse({"items": [{"text": "   ", "claim_ids": ["a"]}]}, {"a"}) == []


def test_text_is_stripped() -> None:
    out = _parse({"items": [{"text": "  hi  ", "claim_ids": ["a"]}]}, {"a"})
    assert out == [_Entry("hi", ["a"])]


def test_accepts_json_string_payload() -> None:
    out = _parse('{"items": [{"text": "x", "claim_ids": ["a"]}]}', {"a"})
    assert out == [_Entry("x", ["a"])]


def test_malformed_inputs_yield_empty() -> None:
    assert _parse("not json", {"a"}) == []
    assert _parse(None, {"a"}) == []
    assert _parse({"items": "nope"}, {"a"}) == []
    assert _parse({"other_key": []}, {"a"}) == []
    assert _parse({"items": ["scalar", 7, None]}, {"a"}) == []
    assert _parse({"items": [{"text": "x", "claim_ids": "a"}]}, {"a"}) == []
