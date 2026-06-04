"""Tests for the pure claim-embedding-text composer.

The retrieval substrate embeds a synthesized sentence per claim (not raw
source spans), so the text is deterministic and captures the structured
triple plus any typed numeric fact. These are pure-function tests — no
model, no DB.
"""

from __future__ import annotations

from typing import Any

from src.claims.schemas import EvidenceClaim, make_claim_id, make_claim_key
from src.retrieval.text import claim_embedding_text


def _claim(**overrides: Any) -> EvidenceClaim:
    key = make_claim_key("narrative", "doc_1", "TSMC", "supplies_to", "NVIDIA")
    base: dict[str, Any] = {
        "claim_id": make_claim_id(key),
        "claim_key": key,
        "lane": "narrative",
        "source_id": "doc_1",
        "predicate": "supplies_to",
        "subject_text": "TSMC",
        "object_text": "NVIDIA",
        "contract_version": "v1",
    }
    base.update(overrides)
    return EvidenceClaim(**base)


def test_humanizes_predicate_and_joins_triple() -> None:
    assert claim_embedding_text(_claim()) == "TSMC supplies to NVIDIA"


def test_omits_missing_object() -> None:
    claim = _claim(predicate="expands_capacity", object_text=None)
    assert claim_embedding_text(claim) == "TSMC expands capacity"


def test_appends_numeric_fact_when_present() -> None:
    claim = _claim(
        predicate="revises_guidance",
        object_text="capex",
        metric="capex",
        numeric_value=42_000_000_000.0,
        unit="USD",
        period="2026",
        modality="guided",
    )
    text = claim_embedding_text(claim)
    assert text.startswith("TSMC revises guidance capex")
    # Integer-valued magnitudes drop the trailing .0 noise.
    assert "42000000000" in text
    assert ".0" not in text
    assert "USD" in text
    assert "2026" in text
    assert "guided" in text


def test_keeps_fractional_numeric_value() -> None:
    claim = _claim(metric="margin", numeric_value=53.5, unit="%", object_text=None)
    text = claim_embedding_text(claim)
    assert "53.5" in text
    assert "%" in text


def test_numeric_fact_without_unit_or_period() -> None:
    claim = _claim(metric="headcount", numeric_value=1000.0, object_text=None)
    text = claim_embedding_text(claim)
    assert "headcount" in text
    assert "1000" in text


def test_result_is_stripped_and_nonempty() -> None:
    text = claim_embedding_text(_claim())
    assert text == text.strip()
    assert text
