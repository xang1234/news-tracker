"""Tests for the LLM claim extractor's pure prompt + parser.

The parser must emit schema-valid EvidenceClaims with the SAME deterministic
claim_key the rule pass produces (so the two passes dedup-merge), constrain
predicates to the narrative vocabulary, and degrade malformed input to [].
"""

from __future__ import annotations

from src.claims.llm_prompt import build_extraction_prompt, parse_extraction_response
from src.claims.schemas import make_claim_key
from src.contracts.intelligence.lanes import LANE_NARRATIVE


def _c(subject, predicate, obj=None, confidence=None):
    """Compact claim-entry builder; confidence omitted unless given."""
    entry = {"subject": subject, "predicate": predicate, "object": obj}
    if confidence is not None:
        entry["confidence"] = confidence
    return entry


class TestBuildPrompt:
    def test_includes_content_vocabulary_and_cap(self) -> None:
        prompt = build_extraction_prompt("TSMC supplies NVIDIA.", max_claims=7)
        assert "TSMC supplies NVIDIA." in prompt
        assert "supplies_to" in prompt  # vocabulary listed
        assert "expands_capacity" in prompt
        assert "at most 7" in prompt


class TestParse:
    def test_builds_schema_valid_llm_claim(self) -> None:
        payload = {"claims": [_c("TSMC", "supplies_to", "NVIDIA", 0.9)]}
        claims = parse_extraction_response(payload, doc_id="d1")
        assert len(claims) == 1
        c = claims[0]
        assert c.extraction_method == "llm"
        assert c.lane == LANE_NARRATIVE
        assert (c.subject_text, c.predicate, c.object_text) == ("TSMC", "supplies_to", "NVIDIA")
        assert c.confidence == 0.9
        assert c.source_id == "d1"

    def test_claim_key_matches_rule_key_for_same_triple(self) -> None:
        # The whole point: an LLM triple and a rule triple collapse to one row.
        payload = {"claims": [_c("TSMC", "supplies_to", "NVIDIA")]}
        c = parse_extraction_response(payload, doc_id="d1")[0]
        assert c.claim_key == make_claim_key(LANE_NARRATIVE, "d1", "TSMC", "supplies_to", "NVIDIA")

    def test_normalizes_predicate_to_snake_case(self) -> None:
        payload = {"claims": [_c("TSMC", "Supplies To", "NVIDIA")]}
        c = parse_extraction_response(payload, doc_id="d1")[0]
        assert c.predicate == "supplies_to"

    def test_out_of_vocabulary_predicate_dropped(self) -> None:
        payload = {"claims": [_c("TSMC", "acquires", "Intel")]}
        assert parse_extraction_response(payload, doc_id="d1") == []

    def test_missing_subject_or_predicate_dropped(self) -> None:
        payload = {
            "claims": [
                {"predicate": "supplies_to", "object": "NVIDIA"},
                {"subject": "TSMC", "object": "NVIDIA"},
                _c("  ", "supplies_to"),
            ]
        }
        assert parse_extraction_response(payload, doc_id="d1") == []

    def test_object_null_yields_none(self) -> None:
        payload = {"claims": [_c("Samsung", "expands_capacity", None)]}
        c = parse_extraction_response(payload, doc_id="d1")[0]
        assert c.object_text is None

    def test_confidence_clamped_and_defaulted(self) -> None:
        payload = {
            "claims": [
                _c("TSMC", "supplies_to", "A", 5),  # over-range → clamp to 1.0
                _c("TSMC", "competes_with", "B"),  # no confidence → default 0.5
            ]
        }
        claims = parse_extraction_response(payload, doc_id="d1")
        by_obj = {c.object_text: c.confidence for c in claims}
        assert by_obj["A"] == 1.0
        assert by_obj["B"] == 0.5

    def test_min_confidence_filter(self) -> None:
        payload = {
            "claims": [
                _c("TSMC", "supplies_to", "A", 0.3),
                _c("TSMC", "supplies_to", "B", 0.8),
            ]
        }
        claims = parse_extraction_response(payload, doc_id="d1", min_confidence=0.5)
        assert [c.object_text for c in claims] == ["B"]

    def test_dedupes_within_response(self) -> None:
        # Same triple after normalization → one claim (first wins).
        payload = {
            "claims": [
                _c("TSMC", "supplies_to", "NVIDIA", 0.9),
                _c("tsmc", "SUPPLIES_TO", "nvidia", 0.7),
            ]
        }
        claims = parse_extraction_response(payload, doc_id="d1")
        assert len(claims) == 1

    def test_malformed_inputs_yield_empty(self) -> None:
        assert parse_extraction_response("not json", doc_id="d1") == []
        assert parse_extraction_response(None, doc_id="d1") == []
        assert parse_extraction_response({"claims": "nope"}, doc_id="d1") == []
        assert parse_extraction_response({"other": []}, doc_id="d1") == []
        assert parse_extraction_response({"claims": [7, "x", None]}, doc_id="d1") == []

    def test_recovers_implicit_golden_claim(self) -> None:
        # gold_001's headroom: the regex misses "TSMC supplies_to NVIDIA" (implicit).
        # Given the JSON an ideal LLM returns, the parser yields both golden triples.
        payload = {
            "claims": [
                _c("TSMC", "expands_capacity", "Arizona fab"),
                _c("TSMC", "supplies_to", "NVIDIA"),
            ]
        }
        triples = {
            (c.subject_text, c.predicate, c.object_text)
            for c in parse_extraction_response(payload, doc_id="gold_001")
        }
        assert ("TSMC", "supplies_to", "NVIDIA") in triples
        assert ("TSMC", "expands_capacity", "Arizona fab") in triples
