"""Tests for the narrative claim extractor.

Validates event→claim conversion, co-occurrence extraction, and
deduplication via claim_key.
"""

from __future__ import annotations

import pytest

from src.claims.narrative_extractor import (
    EVENT_TYPE_TO_PREDICATE,
    extract_claims_from_cooccurrence,
    extract_claims_from_document,
    extract_claims_from_events,
)
from src.contracts.intelligence.lanes import LANE_NARRATIVE


# -- Event → Claim tests -----------------------------------------------------


class TestExtractClaimsFromEvents:
    def test_basic_event_extraction(self):
        events = [
            {
                "event_type": "capacity_expansion",
                "actor": "TSMC",
                "action": "is expanding",
                "object": "3nm fab capacity",
                "confidence": 0.85,
                "span_start": 10,
                "span_end": 50,
                "tickers": ["TSM"],
            },
        ]

        claims = extract_claims_from_events("doc_123", events)

        assert len(claims) == 1
        claim = claims[0]
        assert claim.lane == LANE_NARRATIVE
        assert claim.source_id == "doc_123"
        assert claim.subject_text == "TSMC"
        assert claim.predicate == "expands_capacity"
        assert claim.object_text == "3nm fab capacity"
        assert claim.confidence == 0.85
        assert claim.extraction_method == "rule"
        assert claim.metadata["event_type"] == "capacity_expansion"
        assert claim.metadata["tickers"] == ["TSM"]

    def test_all_event_types_mapped(self):
        for event_type, predicate in EVENT_TYPE_TO_PREDICATE.items():
            events = [
                {
                    "event_type": event_type,
                    "actor": "Intel",
                    "action": "does something",
                    "object": "thing",
                    "confidence": 0.7,
                    "span_start": 0,
                    "span_end": 10,
                },
            ]
            claims = extract_claims_from_events("doc_1", events)
            assert len(claims) == 1
            assert claims[0].predicate == predicate

    def test_skips_event_without_actor(self):
        events = [
            {
                "event_type": "capacity_expansion",
                "actor": "",
                "action": "expanding",
                "object": "capacity",
                "confidence": 0.8,
                "span_start": 0,
                "span_end": 10,
            },
        ]
        claims = extract_claims_from_events("doc_1", events)
        assert len(claims) == 0

    def test_skips_unknown_event_type(self):
        events = [
            {
                "event_type": "unknown_type",
                "actor": "TSMC",
                "action": "does",
                "object": "thing",
                "confidence": 0.7,
                "span_start": 0,
                "span_end": 10,
            },
        ]
        claims = extract_claims_from_events("doc_1", events)
        assert len(claims) == 0

    def test_deterministic_claim_key(self):
        events = [
            {
                "event_type": "product_launch",
                "actor": "NVIDIA",
                "action": "launches",
                "object": "H200",
                "confidence": 0.9,
                "span_start": 0,
                "span_end": 10,
            },
        ]
        claims_1 = extract_claims_from_events("doc_1", events)
        claims_2 = extract_claims_from_events("doc_1", events)
        assert claims_1[0].claim_key == claims_2[0].claim_key

    def test_multiple_events(self):
        events = [
            {
                "event_type": "capacity_expansion",
                "actor": "TSMC",
                "action": "expands",
                "object": "3nm",
                "confidence": 0.8,
                "span_start": 0,
                "span_end": 10,
            },
            {
                "event_type": "product_launch",
                "actor": "NVIDIA",
                "action": "launches",
                "object": "H200",
                "confidence": 0.9,
                "span_start": 50,
                "span_end": 80,
            },
        ]
        claims = extract_claims_from_events("doc_1", events)
        assert len(claims) == 2
        assert claims[0].subject_text == "TSMC"
        assert claims[1].subject_text == "NVIDIA"


# -- Co-occurrence → Claim tests ----------------------------------------------


class TestExtractClaimsFromCooccurrence:
    def test_supply_verb_detection(self):
        entities = [
            {"type": "COMPANY", "text": "TSMC", "start": 0},
            {"type": "COMPANY", "text": "NVIDIA", "start": 30},
        ]
        content = "TSMC supplies advanced chips to NVIDIA for AI training."

        claims = extract_claims_from_cooccurrence("doc_1", entities, content)

        assert len(claims) == 1
        assert claims[0].subject_text == "TSMC"
        assert claims[0].predicate == "supplies_to"
        assert claims[0].object_text == "NVIDIA"
        assert claims[0].confidence == 0.45
        assert claims[0].metadata["detection"] == "cooccurrence"

    def test_compete_verb_detection(self):
        entities = [
            {"type": "COMPANY", "text": "AMD", "start": 0},
            {"type": "COMPANY", "text": "Intel", "start": 30},
        ]
        content = "AMD competes aggressively with Intel in the server market."

        claims = extract_claims_from_cooccurrence("doc_1", entities, content)

        assert len(claims) == 1
        assert claims[0].predicate == "competes_with"

    def test_no_verb_no_claim(self):
        entities = [
            {"type": "COMPANY", "text": "Apple", "start": 0},
            {"type": "COMPANY", "text": "Google", "start": 20},
        ]
        content = "Apple and Google were mentioned."

        claims = extract_claims_from_cooccurrence("doc_1", entities, content)
        assert len(claims) == 0

    def test_too_far_apart(self):
        entities = [
            {"type": "COMPANY", "text": "TSMC", "start": 0},
            {"type": "COMPANY", "text": "NVIDIA", "start": 500},
        ]
        content = "TSMC " + "x" * 480 + " supplies NVIDIA"

        claims = extract_claims_from_cooccurrence("doc_1", entities, content)
        assert len(claims) == 0

    def test_same_entity_skipped(self):
        entities = [
            {"type": "COMPANY", "text": "TSMC", "start": 0},
            {"type": "COMPANY", "text": "TSMC", "start": 30},
        ]
        content = "TSMC supplies chips, and TSMC expands capacity."

        claims = extract_claims_from_cooccurrence("doc_1", entities, content)
        assert len(claims) == 0

    def test_single_entity_returns_empty(self):
        entities = [
            {"type": "COMPANY", "text": "TSMC", "start": 0},
        ]
        claims = extract_claims_from_cooccurrence("doc_1", entities, "TSMC supplies chips.")
        assert len(claims) == 0

    def test_non_company_entities_skipped(self):
        entities = [
            {"type": "TECHNOLOGY", "text": "EUV", "start": 0},
            {"type": "PRODUCT", "text": "H200", "start": 30},
        ]
        content = "EUV supplies advanced H200 chips."
        claims = extract_claims_from_cooccurrence("doc_1", entities, content)
        assert len(claims) == 0


# -- Combined extraction tests ------------------------------------------------


class TestExtractClaimsFromDocument:
    def test_combines_event_and_cooccurrence(self):
        events = [
            {
                "event_type": "capacity_expansion",
                "actor": "TSMC",
                "action": "expands",
                "object": "capacity",
                "confidence": 0.8,
                "span_start": 0,
                "span_end": 30,
            },
        ]
        content = "TSMC expands capacity. AMD competes with Intel in CPUs."
        # Entity positions match their actual offset in content
        amd_pos = content.index("AMD")
        intel_pos = content.index("Intel")
        entities = [
            {"type": "COMPANY", "text": "AMD", "start": amd_pos},
            {"type": "COMPANY", "text": "Intel", "start": intel_pos},
        ]

        claims = extract_claims_from_document(
            "doc_1", events, entities, content,
        )

        # 1 event claim + 1 co-occurrence claim
        assert len(claims) == 2
        predicates = {c.predicate for c in claims}
        assert "expands_capacity" in predicates
        assert "competes_with" in predicates

    def test_deduplication_by_claim_key(self):
        # Same actor/predicate from both event and co-occurrence
        events = [
            {
                "event_type": "capacity_expansion",
                "actor": "TSMC",
                "action": "expands",
                "object": "capacity",
                "confidence": 0.8,
                "span_start": 0,
                "span_end": 30,
            },
        ]
        claims = extract_claims_from_document("doc_1", events, [], "text")
        keys = [c.claim_key for c in claims]
        assert len(keys) == len(set(keys)), "Duplicate claim keys"

    def test_empty_inputs(self):
        claims = extract_claims_from_document("doc_1", [], [], "some text")
        assert len(claims) == 0
