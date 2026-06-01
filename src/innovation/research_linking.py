"""Research-record to company/theme linking logic."""

from __future__ import annotations

import re
from typing import Protocol

from src.innovation.research_schemas import ResearchRecord, ResearchSignal
from src.security_master.concept_schemas import ConceptAliasCandidate, IssuerSecurityLink


class ResearchConceptRepository(Protocol):
    async def resolve_alias_candidates(
        self,
        alias: str,
        *,
        limit: int = 10,
        include_rejected: bool = False,
    ) -> list[ConceptAliasCandidate]:
        """Return issuer candidates for an institution or lab alias."""
        ...

    async def get_securities_for_issuer(
        self,
        issuer_concept_id: str,
    ) -> list[IssuerSecurityLink]:
        """Return securities linked to an issuer concept."""
        ...


class ResearchSignalLinker:
    """Link research records to tracked securities and curated themes."""

    def __init__(
        self,
        *,
        concept_repository: ResearchConceptRepository,
        theme_by_topic: dict[str, str],
        min_alias_confidence: float = 0.50,
    ) -> None:
        self._concept_repository = concept_repository
        self._theme_by_topic = {
            _normalize_text(key): value for key, value in theme_by_topic.items()
        }
        self._min_alias_confidence = min_alias_confidence

    async def link_records(self, records: list[ResearchRecord]) -> list[ResearchSignal]:
        signals: list[ResearchSignal] = []
        for record in records:
            theme_id = self._theme_for_record(record)
            if theme_id is None:
                continue
            for institution in record.institutions:
                candidates = await self._concept_repository.resolve_alias_candidates(
                    institution,
                    limit=10,
                )
                signals.extend(
                    await self._signals_for_candidates(
                        record,
                        institution=institution,
                        candidates=candidates,
                        theme_id=theme_id,
                    )
                )
        return signals

    async def _signals_for_candidates(
        self,
        record: ResearchRecord,
        *,
        institution: str,
        candidates: list[ConceptAliasCandidate],
        theme_id: str,
    ) -> list[ResearchSignal]:
        signals: list[ResearchSignal] = []
        for candidate in candidates:
            if candidate.alias.confidence < self._min_alias_confidence:
                continue
            securities = await self._concept_repository.get_securities_for_issuer(
                candidate.concept.concept_id
            )
            signals.extend(
                self._make_signal(
                    record,
                    candidate,
                    security,
                    institution=institution,
                    theme_id=theme_id,
                )
                for security in securities
            )
        return signals

    def _theme_for_record(self, record: ResearchRecord) -> str | None:
        text_values = [_normalize_text(value) for value in [*record.topics, *record.categories]]
        for query, theme_id in self._theme_by_topic.items():
            if any(query in value or value in query for value in text_values):
                return theme_id
        return None

    def _make_signal(
        self,
        record: ResearchRecord,
        candidate: ConceptAliasCandidate,
        security: IssuerSecurityLink,
        *,
        institution: str,
        theme_id: str,
    ) -> ResearchSignal:
        requires_review = candidate.requires_review
        confidence = _research_confidence(
            candidate.alias.confidence,
            record.source,
            requires_review,
        )
        return ResearchSignal(
            source=record.source,
            record_id=record.record_id,
            published_date=record.published_date,
            title=record.title,
            issuer_concept_id=candidate.concept.concept_id,
            security_concept_id=security.security_concept_id,
            theme_id=theme_id,
            confidence=confidence,
            confidence_reasons=[
                f"institution_alias:{candidate.alias.confidence:.2f}",
                f"source:{record.source}",
                f"topic:{theme_id}",
            ],
            source_lineage={
                **record.source_lineage,
                "institution_alias": institution,
                "alias_source_attribution": candidate.alias.source_attribution,
                "topics": record.topics,
                "categories": record.categories,
            },
            metadata={
                **record.metadata,
                "requires_review": requires_review,
                "alias_type": candidate.alias.alias_type,
                "alias_review_status": candidate.alias.review_status,
                "security_relationship": security.relationship_type,
            },
            url=record.url,
            fetched_at=record.fetched_at,
        )


def deduplicate_research_records(records: list[ResearchRecord]) -> list[ResearchRecord]:
    """Collapse duplicate DOI/arXiv records before signal linking."""
    chosen: dict[str, ResearchRecord] = {}
    for record in records:
        key = _dedupe_key(record)
        current = chosen.get(key)
        if current is None or _dedupe_rank(record) > _dedupe_rank(current):
            chosen[key] = record
    return list(chosen.values())


def _dedupe_key(record: ResearchRecord) -> str:
    if record.doi:
        return f"doi:{_normalize_doi(record.doi)}"
    if record.arxiv_id:
        return f"arxiv:{record.arxiv_id.lower()}"
    return f"{record.source}:{record.record_id}"


def _dedupe_rank(record: ResearchRecord) -> tuple[int, int, str]:
    source_rank = 2 if record.source == "openalex" else 1
    return (source_rank, len(record.institutions), record.record_id)


def _research_confidence(alias_confidence: float, source: str, requires_review: bool) -> float:
    source_weight = 0.90 if source == "openalex" else 0.80
    review_weight = 0.75 if requires_review else 1.0
    return round(max(0.0, min(1.0, alias_confidence * source_weight * review_weight)), 4)


def _normalize_doi(value: str) -> str:
    return value.strip().lower().removeprefix("https://doi.org/")


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
