"""Tests for patent/research alias mapping policy."""

from src.security_master.innovation_aliases import (
    INNOVATION_ALIAS_TYPES,
    build_innovation_alias,
    normalize_innovation_alias,
)


def test_builds_company_lab_alias_with_auditable_metadata() -> None:
    alias = build_innovation_alias(
        concept_id="concept_issuer_nvda",
        alias="NVIDIA Research",
        alias_type="lab",
        source_attribution="curated_innovation_aliases",
        source_contexts=["patents", "research"],
    )

    assert alias.alias == "NVIDIA Research"
    assert alias.alias_type == "lab"
    assert alias.confidence == 0.72
    assert alias.review_status == "accepted"
    assert alias.source_attribution == "curated_innovation_aliases"
    assert alias.metadata["source_contexts"] == ["patents", "research"]


def test_supports_required_innovation_alias_types() -> None:
    assert {
        "name",
        "abbreviation",
        "subsidiary",
        "acquired_entity",
        "lab",
        "research_institution",
    }.issubset(INNOVATION_ALIAS_TYPES)


def test_generic_aliases_are_retained_but_flagged_for_review() -> None:
    alias = build_innovation_alias(
        concept_id="concept_issuer_test",
        alias="Research",
        alias_type="lab",
        source_attribution="curated_innovation_aliases",
    )

    assert alias.review_status == "needs_review"
    assert alias.confidence == 0.35
    assert "generic_alias" in alias.metadata["false_positive_risks"]


def test_university_company_collisions_are_low_confidence_review_candidates() -> None:
    alias = build_innovation_alias(
        concept_id="concept_issuer_micron",
        alias="MIT",
        alias_type="abbreviation",
        source_attribution="curated_innovation_aliases",
    )

    assert alias.review_status == "needs_review"
    assert alias.confidence <= 0.5
    assert "academic_institution_collision" in alias.metadata["false_positive_risks"]


def test_research_institution_names_need_review_before_company_resolution() -> None:
    alias = build_innovation_alias(
        concept_id="concept_issuer_test",
        alias="University of Toronto",
        alias_type="research_institution",
        source_attribution="curated_innovation_aliases",
    )

    assert alias.review_status == "needs_review"
    assert "academic_institution_collision" in alias.metadata["false_positive_risks"]


def test_normalization_preserves_distinguishing_words() -> None:
    assert normalize_innovation_alias("  NVIDIA, Inc. Research Lab  ") == ("nvidia research lab")
