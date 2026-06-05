"""Tests for the pure claim→citation mapper.

A citation carries the lineage the UI needs to make a cited claim_id clickable:
the resolved triple plus the source (type/id/span) so the frontend can jump to
the evidence document and span. Pure function — no model, no DB.
"""

from __future__ import annotations

from typing import Any

from src.briefing.citations import citation_from_claim
from src.briefing.schemas import ClaimCitation
from src.claims.schemas import EvidenceClaim, make_claim_id, make_claim_key


def _claim(**overrides: Any) -> EvidenceClaim:
    key = make_claim_key("narrative", "doc_42", "TSMC", "supplies_to", "NVIDIA")
    base: dict[str, Any] = {
        "claim_id": make_claim_id(key),
        "claim_key": key,
        "lane": "narrative",
        "source_id": "doc_42",
        "source_type": "document",
        "subject_text": "TSMC",
        "predicate": "supplies_to",
        "object_text": "NVIDIA",
        "source_span_start": 100,
        "source_span_end": 140,
        "source_text": "TSMC supplies advanced packaging to NVIDIA",
        "contract_version": "v1",
    }
    base.update(overrides)
    return EvidenceClaim(**base)


def test_maps_lineage_fields() -> None:
    cit = citation_from_claim(_claim())
    assert isinstance(cit, ClaimCitation)
    assert cit.claim_id == _claim().claim_id
    assert cit.subject_text == "TSMC"
    assert cit.predicate == "supplies_to"
    assert cit.object_text == "NVIDIA"
    assert cit.source_type == "document"
    assert cit.source_id == "doc_42"
    assert cit.source_span_start == 100
    assert cit.source_span_end == 140
    assert cit.snippet == "TSMC supplies advanced packaging to NVIDIA"


def test_handles_missing_optional_fields() -> None:
    cit = citation_from_claim(
        _claim(object_text=None, source_span_start=None, source_span_end=None, source_text=None)
    )
    assert cit.object_text is None
    assert cit.source_span_start is None
    assert cit.snippet is None
