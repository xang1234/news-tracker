"""Tests for research evidence linking and duplicate suppression."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.innovation.research import (
    ResearchRecord,
    ResearchSignalLinker,
    deduplicate_research_records,
)
from src.security_master.concept_schemas import (
    Concept,
    ConceptAlias,
    ConceptAliasCandidate,
    IssuerSecurityLink,
)


def _concept(concept_id: str) -> Concept:
    return Concept(
        concept_id=concept_id,
        concept_type="issuer",
        canonical_name=concept_id.upper(),
        description="",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _candidate(
    concept_id: str,
    *,
    confidence: float,
    review_status: str = "accepted",
) -> ConceptAliasCandidate:
    return ConceptAliasCandidate(
        concept=_concept(concept_id),
        alias=ConceptAlias(
            alias="NVIDIA Research",
            concept_id=concept_id,
            alias_type="lab",
            confidence=confidence,
            source_attribution="innovation_aliases",
            review_status=review_status,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


class FakeConceptRepository:
    def __init__(self, *, candidate_confidence: float = 0.88) -> None:
        self.candidate_confidence = candidate_confidence
        self.aliases: list[str] = []

    async def resolve_alias_candidates(
        self,
        alias: str,
        *,
        limit: int = 10,
        include_rejected: bool = False,
    ) -> list[ConceptAliasCandidate]:
        self.aliases.append(alias)
        return [_candidate("issuer_nvda", confidence=self.candidate_confidence)]

    async def get_securities_for_issuer(self, issuer_concept_id: str) -> list[IssuerSecurityLink]:
        return [
            IssuerSecurityLink(
                issuer_concept_id=issuer_concept_id,
                security_concept_id="security_nvda",
                relationship_type="primary",
                metadata={"ticker": "NVDA", "exchange": "US"},
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]


def _record(
    source: str,
    record_id: str,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
) -> ResearchRecord:
    return ResearchRecord(
        source=source,
        record_id=record_id,
        title="AI accelerator interconnect research",
        abstract="",
        authors=["Ada Researcher"],
        institutions=["NVIDIA Research"],
        topics=["AI accelerators"],
        categories=["cs.AR"],
        published_date=date(2026, 5, 20),
        url="https://example.org/research",
        doi=doi,
        arxiv_id=arxiv_id,
        source_lineage={"source": source},
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_research_linker_maps_curated_topics_to_security_theme_signals() -> None:
    repository = FakeConceptRepository()
    linker = ResearchSignalLinker(
        concept_repository=repository,
        theme_by_topic={"ai accelerators": "theme_ai_accelerators"},
    )

    signals = await linker.link_records([_record("openalex", "W1", doi="10.1000/chiplet")])

    assert repository.aliases == ["NVIDIA Research"]
    assert len(signals) == 1
    assert signals[0].source == "openalex"
    assert signals[0].record_id == "W1"
    assert signals[0].issuer_concept_id == "issuer_nvda"
    assert signals[0].security_concept_id == "security_nvda"
    assert signals[0].theme_id == "theme_ai_accelerators"
    assert signals[0].confidence > 0.70
    assert signals[0].source_lineage["institution_alias"] == "NVIDIA Research"
    assert signals[0].metadata["alias_review_status"] == "accepted"


@pytest.mark.asyncio
async def test_research_linker_skips_weak_issuer_matches() -> None:
    linker = ResearchSignalLinker(
        concept_repository=FakeConceptRepository(candidate_confidence=0.31),
        theme_by_topic={"ai accelerators": "theme_ai_accelerators"},
        min_alias_confidence=0.55,
    )

    assert await linker.link_records([_record("openalex", "W1")]) == []


def test_deduplicate_research_records_prefers_openalex_metadata_for_duplicate_doi() -> None:
    records = [
        _record("arxiv", "2605.12345", doi="10.1000/chiplet", arxiv_id="2605.12345"),
        _record("openalex", "https://openalex.org/W1", doi="10.1000/chiplet"),
        _record("arxiv", "2605.99999", arxiv_id="2605.99999"),
    ]

    deduped = deduplicate_research_records(records)

    assert [(record.source, record.record_id) for record in deduped] == [
        ("openalex", "https://openalex.org/W1"),
        ("arxiv", "2605.99999"),
    ]
