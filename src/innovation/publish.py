"""Publish payload helpers for patent and research innovation evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.innovation.patent_schemas import PatentSignal
from src.innovation.research_schemas import ResearchSignal

InnovationSignal = PatentSignal | ResearchSignal
InnovationEvidencePayload = dict[str, Any]

MAX_INNOVATION_CONFIDENCE_WEIGHT = 0.65
INNOVATION_CONFIDENCE_MULTIPLIER = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.65


def build_innovation_evidence_payload(signal: InnovationSignal) -> InnovationEvidencePayload:
    """Build a conservative publish payload for patent/research evidence."""
    common = {
        "evidence_family": "innovation",
        "issuer_concept_id": signal.issuer_concept_id,
        "security_concept_id": signal.security_concept_id,
        "theme_id": signal.theme_id,
        "confidence": signal.confidence,
        "confidence_label": _confidence_label(signal.confidence),
        "confidence_weight": _confidence_weight(signal.confidence),
        "confidence_reasons": list(signal.confidence_reasons),
        "evidence_horizon": "slow_moving_innovation",
        "catalyst_role": "structural_not_near_term_price",
        "lineage": dict(signal.source_lineage),
        "metadata": dict(signal.metadata),
        "fetched_at": signal.fetched_at.isoformat(),
    }
    if isinstance(signal, PatentSignal):
        return {
            **common,
            "source_type": "patent",
            "source": signal.source_lineage.get("source", "patent"),
            "source_id": signal.patent_id,
            "family_id": signal.patent_family_id,
            "event_type": signal.event_type,
            "event_date": signal.event_date.isoformat(),
            "title": signal.title,
            "url": signal.source_url,
        }
    return {
        **common,
        "source_type": "research",
        "source": signal.source,
        "source_id": signal.record_id,
        "event_type": "publication",
        "event_date": signal.published_date.isoformat(),
        "title": signal.title,
        "url": signal.url,
    }


def group_innovation_evidence_by_concept(
    signals: Iterable[InnovationSignal],
) -> dict[str, list[InnovationEvidencePayload]]:
    """Group innovation evidence by issuer and security concept ids."""
    grouped: dict[str, list[InnovationEvidencePayload]] = {}
    for signal in signals:
        payload = build_innovation_evidence_payload(signal)
        for concept_id in dict.fromkeys([signal.issuer_concept_id, signal.security_concept_id]):
            grouped.setdefault(concept_id, []).append(dict(payload))
    return grouped


def group_innovation_evidence_by_theme(
    signals: Iterable[InnovationSignal],
) -> dict[str, list[InnovationEvidencePayload]]:
    """Group innovation evidence by theme id for narrative rollups."""
    grouped: dict[str, list[InnovationEvidencePayload]] = {}
    for signal in signals:
        grouped.setdefault(signal.theme_id, []).append(build_innovation_evidence_payload(signal))
    return grouped


def _confidence_label(confidence: float) -> str:
    return "low" if confidence < LOW_CONFIDENCE_THRESHOLD else "medium"


def _confidence_weight(confidence: float) -> float:
    return round(
        min(
            MAX_INNOVATION_CONFIDENCE_WEIGHT,
            confidence * INNOVATION_CONFIDENCE_MULTIPLIER,
        ),
        4,
    )
