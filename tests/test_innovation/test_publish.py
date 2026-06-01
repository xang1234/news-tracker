"""Tests for publishing patent/research innovation evidence."""

from __future__ import annotations

from datetime import UTC, date, datetime

from src.innovation.patent_schemas import PatentSignal
from src.innovation.publish import (
    build_innovation_evidence_payload,
    group_innovation_evidence_by_concept,
    group_innovation_evidence_by_theme,
)
from src.innovation.research_schemas import ResearchSignal

NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _patent_signal(confidence: float = 0.92) -> PatentSignal:
    return PatentSignal(
        patent_id="US123",
        patent_family_id="FAM1",
        event_type="grant",
        event_date=date(2026, 5, 20),
        title="HBM package thermal bridge",
        issuer_concept_id="concept_issuer_nvda",
        security_concept_id="concept_security_nvda",
        theme_id="theme_hbm",
        confidence=confidence,
        confidence_reasons=["assignee_alias:0.95", "class:H01L"],
        source_lineage={"source": "uspto_patentsview_bulk", "cpc_classes": ["H01L"]},
        metadata={"requires_review": False},
        source_url="https://patents.example/US123",
        fetched_at=NOW,
    )


def _research_signal(confidence: float = 0.48) -> ResearchSignal:
    return ResearchSignal(
        source="openalex",
        record_id="https://openalex.org/W1",
        published_date=date(2026, 5, 21),
        title="EDA placement for chiplet accelerators",
        issuer_concept_id="concept_issuer_nvda",
        security_concept_id="concept_security_nvda",
        theme_id="theme_eda",
        confidence=confidence,
        confidence_reasons=["institution_alias:0.60", "topic:theme_eda"],
        source_lineage={"source": "openalex", "openalex_id": "https://openalex.org/W1"},
        metadata={"requires_review": True},
        url="https://example.org/work",
        fetched_at=NOW,
    )


def test_patent_payload_is_conservative_slow_moving_evidence() -> None:
    payload = build_innovation_evidence_payload(_patent_signal())

    assert payload["evidence_family"] == "innovation"
    assert payload["source_type"] == "patent"
    assert payload["source_id"] == "US123"
    assert payload["event_date"] == "2026-05-20"
    assert payload["confidence"] == 0.92
    assert payload["confidence_label"] == "medium"
    assert payload["confidence_weight"] == 0.65
    assert payload["evidence_horizon"] == "slow_moving_innovation"
    assert payload["catalyst_role"] == "structural_not_near_term_price"
    assert payload["lineage"]["cpc_classes"] == ["H01L"]


def test_research_payload_keeps_low_confidence_research_lineage() -> None:
    payload = build_innovation_evidence_payload(_research_signal())

    assert payload["source_type"] == "research"
    assert payload["source"] == "openalex"
    assert payload["source_id"] == "https://openalex.org/W1"
    assert payload["event_date"] == "2026-05-21"
    assert payload["confidence_label"] == "low"
    assert payload["confidence_weight"] == 0.36
    assert payload["lineage"]["openalex_id"] == "https://openalex.org/W1"


def test_groups_evidence_by_issuer_and_security_concepts_with_detached_payloads() -> None:
    grouped = group_innovation_evidence_by_concept([_patent_signal(), _research_signal()])

    assert set(grouped) == {"concept_issuer_nvda", "concept_security_nvda"}
    assert [item["source_type"] for item in grouped["concept_security_nvda"]] == [
        "patent",
        "research",
    ]
    assert grouped["concept_issuer_nvda"][0] is not grouped["concept_security_nvda"][0]


def test_groups_evidence_by_theme_for_narrative_theme_rollups() -> None:
    grouped = group_innovation_evidence_by_theme([_patent_signal(), _research_signal()])

    assert set(grouped) == {"theme_hbm", "theme_eda"}
    assert grouped["theme_hbm"][0]["source_type"] == "patent"
    assert grouped["theme_eda"][0]["source_type"] == "research"
