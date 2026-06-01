"""Alias policy for patent assignees and research affiliations."""

from __future__ import annotations

import re

from src.security_master.concept_schemas import ConceptAlias

INNOVATION_ALIAS_TYPES = frozenset(
    {
        "name",
        "abbreviation",
        "former_name",
        "subsidiary",
        "acquired_entity",
        "lab",
        "research_institution",
    }
)

_DEFAULT_CONFIDENCE = {
    "name": 0.95,
    "former_name": 0.82,
    "subsidiary": 0.86,
    "acquired_entity": 0.78,
    "lab": 0.72,
    "research_institution": 0.64,
    "abbreviation": 0.70,
}

_GENERIC_ALIASES = frozenset(
    {
        "research",
        "research lab",
        "research labs",
        "laboratory",
        "laboratories",
        "advanced research",
        "technology",
        "technologies",
        "semiconductor",
        "semiconductors",
        "systems",
        "microelectronics",
    }
)

_ACADEMIC_TERMS = (
    "university",
    "college",
    "school of",
    "institute of technology",
    "polytechnic",
)

_ACADEMIC_ABBREVIATIONS = frozenset(
    {
        "mit",
        "cmu",
        "caltech",
        "ucsd",
        "ucla",
        "uiuc",
        "gatech",
        "eth",
        "ntu",
        "nus",
    }
)


def normalize_innovation_alias(alias: str) -> str:
    """Normalize an assignee or affiliation alias without deleting meaning."""
    normalized = alias.strip().lower()
    normalized = re.sub(
        r"\b(inc|incorporated|corp|corporation|co|company|ltd|llc)\b\.?",
        "",
        normalized,
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def build_innovation_alias(
    *,
    concept_id: str,
    alias: str,
    alias_type: str,
    source_attribution: str,
    source_contexts: list[str] | None = None,
    confidence: float | None = None,
    review_note: str = "",
) -> ConceptAlias:
    """Build a concept alias with patent/research false-positive guardrails."""
    if alias_type not in INNOVATION_ALIAS_TYPES:
        raise ValueError(
            f"alias_type must be one of {sorted(INNOVATION_ALIAS_TYPES)} for innovation aliases"
        )

    normalized = normalize_innovation_alias(alias)
    risks = _false_positive_risks(normalized, alias_type)
    base_confidence = confidence if confidence is not None else _DEFAULT_CONFIDENCE[alias_type]
    if risks:
        base_confidence = min(base_confidence, 0.35 if "generic_alias" in risks else 0.50)

    metadata = {
        "normalized_alias": normalized,
        "source_contexts": list(source_contexts or []),
        "false_positive_risks": risks,
    }
    return ConceptAlias(
        alias=alias.strip(),
        concept_id=concept_id,
        alias_type=alias_type,
        confidence=base_confidence,
        source_attribution=source_attribution,
        review_status="needs_review" if risks else "accepted",
        review_note=review_note or _review_note_for_risks(risks),
        metadata=metadata,
    )


def _false_positive_risks(normalized_alias: str, alias_type: str) -> list[str]:
    risks: list[str] = []
    if normalized_alias in _GENERIC_ALIASES:
        risks.append("generic_alias")
    if _has_academic_collision(normalized_alias, alias_type):
        risks.append("academic_institution_collision")
    return risks


def _has_academic_collision(normalized_alias: str, alias_type: str) -> bool:
    if alias_type == "research_institution":
        return True
    if normalized_alias in _ACADEMIC_ABBREVIATIONS:
        return True
    return any(term in normalized_alias for term in _ACADEMIC_TERMS)


def _review_note_for_risks(risks: list[str]) -> str:
    if not risks:
        return ""
    return "Manual review required for innovation alias false-positive risks: " + ", ".join(
        risks
    )
