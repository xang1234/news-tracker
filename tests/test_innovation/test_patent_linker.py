"""Tests for linking patent evidence to concepts, securities, and themes."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.innovation.patents import PatentRecord, PatentSignalLinker
from src.security_master.concept_schemas import (
    Concept,
    ConceptAlias,
    ConceptAliasCandidate,
    IssuerSecurityLink,
)


def _concept(concept_id: str, name: str) -> Concept:
    return Concept(
        concept_id=concept_id,
        concept_type="issuer",
        canonical_name=name,
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
        concept=_concept(concept_id, concept_id.upper()),
        alias=ConceptAlias(
            alias="Advanced Micro Devices",
            concept_id=concept_id,
            alias_type="name",
            confidence=confidence,
            source_attribution="issuer_registry",
            review_status=review_status,
            review_note="needs human review" if review_status != "accepted" else "",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


class FakeConceptRepository:
    def __init__(self) -> None:
        self.aliases: list[str] = []

    async def resolve_alias_candidates(
        self,
        alias: str,
        *,
        limit: int = 10,
        include_rejected: bool = False,
    ) -> list[ConceptAliasCandidate]:
        self.aliases.append(alias)
        return [
            _candidate("issuer_amd", confidence=0.92),
            _candidate(
                "issuer_amd_research_foundation",
                confidence=0.54,
                review_status="needs_review",
            ),
        ]

    async def get_securities_for_issuer(self, issuer_concept_id: str) -> list[IssuerSecurityLink]:
        return [
            IssuerSecurityLink(
                issuer_concept_id=issuer_concept_id,
                security_concept_id=f"security_{issuer_concept_id}",
                relationship_type="primary",
                metadata={"ticker": "AMD", "exchange": "US"},
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ]


class NoSecurityConceptRepository(FakeConceptRepository):
    async def get_securities_for_issuer(self, issuer_concept_id: str) -> list[IssuerSecurityLink]:
        return []


@pytest.mark.asyncio
async def test_linker_retains_ambiguous_assignee_matches_with_confidence_metadata() -> None:
    repository = FakeConceptRepository()
    linker = PatentSignalLinker(
        concept_repository=repository,
        theme_by_class={"G06N": "theme_ai_accelerators"},
    )
    record = PatentRecord(
        patent_id="US999",
        application_id="17123456",
        patent_family_id="fam-999",
        title="Neural network chiplet package",
        abstract="",
        assignees=["Advanced Micro Devices"],
        cpc_classes=["G06N 3/04", "H01L 23/00"],
        ipc_classes=[],
        application_date=date(2024, 2, 1),
        grant_date=date(2025, 8, 1),
        source_url="https://api.uspto.gov/api/v1/patent/applications/search",
        source_attribution="uspto_odp_patentsview_transition",
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    signals = await linker.link_records([record])

    assert repository.aliases == ["Advanced Micro Devices"]
    assert [signal.issuer_concept_id for signal in signals] == [
        "issuer_amd",
        "issuer_amd_research_foundation",
    ]
    assert {signal.theme_id for signal in signals} == {"theme_ai_accelerators"}
    assert signals[0].security_concept_id == "security_issuer_amd"
    assert signals[0].confidence > signals[1].confidence
    assert signals[1].metadata["requires_review"] is True
    assert signals[1].metadata["alias_review_status"] == "needs_review"
    assert signals[0].source_lineage["assignee_alias"] == "Advanced Micro Devices"
    assert signals[0].source_lineage["patent_family_id"] == "fam-999"
    assert signals[0].source_lineage["classes"] == ["G06N 3/04", "H01L 23/00"]


@pytest.mark.asyncio
async def test_linker_does_not_emit_theme_signals_for_unmapped_classifications() -> None:
    linker = PatentSignalLinker(
        concept_repository=FakeConceptRepository(),
        theme_by_class={"G06N": "theme_ai_accelerators"},
    )
    record = PatentRecord(
        patent_id="US1000",
        application_id="17123457",
        patent_family_id="fam-1000",
        title="Unmapped mechanical device",
        abstract="",
        assignees=["Advanced Micro Devices"],
        cpc_classes=["A01B 33/00"],
        ipc_classes=[],
        application_date=date(2024, 2, 1),
        grant_date=date(2025, 8, 1),
        source_url="https://api.uspto.gov/api/v1/patent/applications/search",
        source_attribution="uspto_odp_patentsview_transition",
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert await linker.link_records([record]) == []


@pytest.mark.asyncio
async def test_linker_requires_a_security_link_before_emitting_signals() -> None:
    linker = PatentSignalLinker(
        concept_repository=NoSecurityConceptRepository(),
        theme_by_class={"G06N": "theme_ai_accelerators"},
    )
    record = PatentRecord(
        patent_id="US1001",
        application_id="17123458",
        patent_family_id="fam-1001",
        title="Securityless issuer evidence",
        abstract="",
        assignees=["Advanced Micro Devices"],
        cpc_classes=["G06N 3/04"],
        ipc_classes=[],
        application_date=date(2024, 2, 1),
        grant_date=date(2025, 8, 1),
        source_url="https://api.uspto.gov/api/v1/patent/applications/search",
        source_attribution="uspto_odp_patentsview_transition",
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert await linker.link_records([record]) == []
