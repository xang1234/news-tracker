"""Tests for the pure Q&A prompt builder + citation-validating parser.

Same grounding gate as briefings: a segment citing ids outside the retrieved
set has those ids stripped, and a segment left uncited is dropped — so an
answer can never contain an uncited/hallucinated-cited assertion.
"""

from __future__ import annotations

from src.qa.prompt import build_qa_prompt, parse_qa_response


def test_prompt_includes_question_and_claim_ids() -> None:
    prompt = build_qa_prompt(
        question="What is the latest on HBM supply for NVIDIA?",
        claims=[
            ("claim_a", "SK Hynix supplies HBM to NVIDIA"),
            ("claim_b", "Samsung expands HBM capacity"),
        ],
    )
    assert "HBM supply for NVIDIA" in prompt
    assert "claim_a" in prompt and "SK Hynix supplies HBM to NVIDIA" in prompt
    assert "claim_b" in prompt
    assert "claim_ids" in prompt
    assert "JSON" in prompt or "json" in prompt


VALID = {"claim_a", "claim_b"}


def test_parses_valid_segments() -> None:
    payload = {
        "segments": [
            {"text": "SK Hynix supplies NVIDIA.", "claim_ids": ["claim_a"]},
            {"text": "Samsung is expanding.", "claim_ids": ["claim_b"]},
        ]
    }
    segs = parse_qa_response(payload, VALID)
    assert [s.text for s in segs] == ["SK Hynix supplies NVIDIA.", "Samsung is expanding."]


def test_strips_hallucinated_ids_and_drops_uncited() -> None:
    payload = {
        "segments": [
            {"text": "Grounded.", "claim_ids": ["claim_a", "ghost"]},
            {"text": "Floating.", "claim_ids": ["ghost"]},
        ]
    }
    segs = parse_qa_response(payload, VALID)
    assert len(segs) == 1
    assert segs[0].claim_ids == ["claim_a"]


def test_drops_empty_text_and_no_citations() -> None:
    assert parse_qa_response({"segments": [{"text": "  ", "claim_ids": ["claim_a"]}]}, VALID) == []
    assert parse_qa_response({"segments": [{"text": "x", "claim_ids": []}]}, VALID) == []


def test_accepts_json_string_and_rejects_malformed() -> None:
    assert len(parse_qa_response('{"segments":[{"text":"x","claim_ids":["claim_a"]}]}', VALID)) == 1
    assert parse_qa_response("nope", VALID) == []
    assert parse_qa_response(None, VALID) == []
    assert parse_qa_response({"wrong": 1}, VALID) == []
    assert parse_qa_response({"segments": "notalist"}, VALID) == []
