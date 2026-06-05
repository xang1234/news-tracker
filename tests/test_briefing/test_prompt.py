"""Tests for the pure briefing prompt builder + citation-validating parser.

The parser is where the "every clause is grounded, no uncited assertion"
invariant is enforced: clauses citing ids outside the retrieved set have those
ids stripped, and clauses left with no valid citation are dropped. Pure
functions — no model, no DB.
"""

from __future__ import annotations

from src.briefing.prompt import build_briefing_prompt, parse_briefing_response


def test_prompt_includes_theme_and_claim_ids() -> None:
    prompt = build_briefing_prompt(
        theme_name="AI accelerators",
        claims=[
            ("claim_a", "TSMC supplies to NVIDIA"),
            ("claim_b", "Samsung expands capacity"),
        ],
    )
    assert "AI accelerators" in prompt
    assert "claim_a" in prompt and "TSMC supplies to NVIDIA" in prompt
    assert "claim_b" in prompt and "Samsung expands capacity" in prompt
    # Must instruct JSON output with per-clause citations.
    assert "claim_ids" in prompt
    assert "JSON" in prompt or "json" in prompt


VALID = {"claim_a", "claim_b", "claim_c"}


def test_parses_valid_clauses() -> None:
    payload = {
        "clauses": [
            {"text": "TSMC supplies NVIDIA.", "claim_ids": ["claim_a"]},
            {"text": "Samsung is expanding.", "claim_ids": ["claim_b", "claim_c"]},
        ]
    }
    clauses = parse_briefing_response(payload, VALID)
    assert [c.text for c in clauses] == ["TSMC supplies NVIDIA.", "Samsung is expanding."]
    assert clauses[1].claim_ids == ["claim_b", "claim_c"]


def test_strips_hallucinated_claim_ids() -> None:
    payload = {"clauses": [{"text": "Mixed.", "claim_ids": ["claim_a", "claim_ghost"]}]}
    clauses = parse_briefing_response(payload, VALID)
    assert len(clauses) == 1
    assert clauses[0].claim_ids == ["claim_a"]  # ghost id removed


def test_drops_clause_left_uncited() -> None:
    # A clause whose every id is hallucinated has no grounding → dropped.
    payload = {
        "clauses": [
            {"text": "Grounded.", "claim_ids": ["claim_a"]},
            {"text": "Hallucinated.", "claim_ids": ["claim_ghost"]},
        ]
    }
    clauses = parse_briefing_response(payload, VALID)
    assert [c.text for c in clauses] == ["Grounded."]


def test_drops_clause_with_no_citations() -> None:
    payload = {"clauses": [{"text": "Floating claim.", "claim_ids": []}]}
    assert parse_briefing_response(payload, VALID) == []


def test_drops_empty_text() -> None:
    payload = {"clauses": [{"text": "   ", "claim_ids": ["claim_a"]}]}
    assert parse_briefing_response(payload, VALID) == []


def test_accepts_json_string_payload() -> None:
    clauses = parse_briefing_response(
        '{"clauses": [{"text": "X.", "claim_ids": ["claim_a"]}]}', VALID
    )
    assert len(clauses) == 1


def test_malformed_payload_returns_empty() -> None:
    assert parse_briefing_response("not json", VALID) == []
    assert parse_briefing_response(None, VALID) == []
    assert parse_briefing_response({"wrong": "shape"}, VALID) == []
    assert parse_briefing_response({"clauses": "notalist"}, VALID) == []


def test_deduplicates_claim_ids_within_clause() -> None:
    payload = {"clauses": [{"text": "Dup.", "claim_ids": ["claim_a", "claim_a"]}]}
    clauses = parse_briefing_response(payload, VALID)
    assert clauses[0].claim_ids == ["claim_a"]
