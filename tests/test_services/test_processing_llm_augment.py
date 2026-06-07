"""Tests for the LLM-augmentation wiring in ProcessingService Stage 6.

Exercises ``_augment_with_llm_claims`` in isolation (built via __new__ to skip
queue/DB construction): the high-value gate decides whether the paid LLM pass
runs, and merge_claims dedups its output into the rule claims by claim_key.
"""

from __future__ import annotations

import pytest

from src.claims.llm_extractor import LLMExtractionConfig
from src.claims.schemas import EvidenceClaim, make_claim_id, make_claim_key
from src.contracts.intelligence.lanes import LANE_NARRATIVE
from src.services.processing_service import ProcessingService


def _claim(subject, predicate, obj=None, *, method="rule", doc="d1"):
    key = make_claim_key(LANE_NARRATIVE, doc, subject, predicate, obj)
    return EvidenceClaim(
        claim_id=make_claim_id(key),
        claim_key=key,
        lane=LANE_NARRATIVE,
        source_id=doc,
        subject_text=subject,
        predicate=predicate,
        object_text=obj,
        extraction_method=method,
    )


class _Doc:
    def __init__(self, authority=None, engagement_score=0.0) -> None:
        self.id = "d1"
        self.content = "TSMC supplies NVIDIA with advanced packaging."
        self.timestamp = None
        self.authority_score = authority
        self.engagement = type("E", (), {"engagement_score": engagement_score})()


class _FakeExtractor:
    def __init__(self, llm_claims, *, config=None) -> None:
        self._llm_claims = llm_claims
        self.config = config or LLMExtractionConfig()
        self.calls = 0

    async def extract(self, doc_id, content, *, run_id=None, published_at=None):
        self.calls += 1
        return self._llm_claims


def _service(extractor) -> ProcessingService:
    # Bypass __init__ (queue/DB/preprocessor); only _llm_extractor is needed here.
    svc = ProcessingService.__new__(ProcessingService)
    svc._llm_extractor = extractor
    return svc


@pytest.mark.asyncio
async def test_low_value_doc_skips_llm_pass() -> None:
    extractor = _FakeExtractor([_claim("TSMC", "supplies_to", "NVIDIA", method="llm")])
    svc = _service(extractor)
    rule = [_claim("TSMC", "expands_capacity", "Arizona fab")]

    result = await svc._augment_with_llm_claims(_Doc(authority=0.1), rule)

    assert result == rule
    assert extractor.calls == 0  # gated out before the paid call


@pytest.mark.asyncio
async def test_high_value_doc_merges_llm_claims() -> None:
    extractor = _FakeExtractor([_claim("TSMC", "supplies_to", "NVIDIA", method="llm")])
    svc = _service(extractor)
    rule = [_claim("TSMC", "expands_capacity", "Arizona fab")]

    result = await svc._augment_with_llm_claims(_Doc(authority=0.9), rule)

    assert extractor.calls == 1
    triples = {(c.subject_text, c.predicate, c.extraction_method) for c in result}
    assert ("TSMC", "expands_capacity", "rule") in triples
    assert ("TSMC", "supplies_to", "llm") in triples


@pytest.mark.asyncio
async def test_corroborating_llm_claim_marks_rule_hybrid() -> None:
    extractor = _FakeExtractor([_claim("TSMC", "supplies_to", "NVIDIA", method="llm")])
    svc = _service(extractor)
    rule = [_claim("TSMC", "supplies_to", "NVIDIA")]  # same triple

    result = await svc._augment_with_llm_claims(_Doc(authority=0.9), rule)

    assert len(result) == 1
    assert result[0].extraction_method == "hybrid"
