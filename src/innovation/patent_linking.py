"""Patent-to-company/theme linking logic."""

from __future__ import annotations

import re
from datetime import date
from typing import Protocol

from src.innovation.patent_schemas import PatentRecord, PatentSignal
from src.security_master.concept_schemas import ConceptAliasCandidate, IssuerSecurityLink


class PatentConceptRepository(Protocol):
    async def resolve_alias_candidates(
        self,
        alias: str,
        *,
        limit: int = 10,
        include_rejected: bool = False,
    ) -> list[ConceptAliasCandidate]:
        """Return candidate issuer concepts for an assignee alias."""
        ...

    async def get_securities_for_issuer(
        self,
        issuer_concept_id: str,
    ) -> list[IssuerSecurityLink]:
        """Return tracked securities linked to an issuer concept."""
        ...


class PatentSignalLinker:
    """Link normalized patents to concepts, securities, and configured themes."""

    def __init__(
        self,
        *,
        concept_repository: PatentConceptRepository,
        theme_by_class: dict[str, str],
    ) -> None:
        self._concept_repository = concept_repository
        self._theme_by_class = {
            _normalize_class(key): value for key, value in theme_by_class.items()
        }

    async def link_records(self, records: list[PatentRecord]) -> list[PatentSignal]:
        """Create conservative theme-linked signals for mapped patent records."""
        signals: list[PatentSignal] = []
        for record in records:
            event_date = record.event_date
            theme_id = self._theme_for_record(record)
            if event_date is None or theme_id is None:
                continue
            signals.extend(
                await self._signals_for_record(
                    record,
                    theme_id=theme_id,
                    event_date=event_date,
                )
            )
        return signals

    async def _signals_for_record(
        self,
        record: PatentRecord,
        *,
        theme_id: str,
        event_date: date,
    ) -> list[PatentSignal]:
        signals: list[PatentSignal] = []
        for assignee in record.assignees:
            candidates = await self._concept_repository.resolve_alias_candidates(
                assignee,
                limit=10,
            )
            for candidate in candidates:
                securities = await self._concept_repository.get_securities_for_issuer(
                    candidate.concept.concept_id
                )
                signals.extend(
                    self._make_signal(
                        record,
                        candidate,
                        security,
                        assignee=assignee,
                        theme_id=theme_id,
                        event_date=event_date,
                    )
                    for security in securities
                )
        return signals

    def _theme_for_record(self, record: PatentRecord) -> str | None:
        for classification in [*record.cpc_classes, *record.ipc_classes]:
            normalized = _normalize_class(classification)
            for prefix, theme_id in self._theme_by_class.items():
                if normalized.startswith(prefix):
                    return theme_id
        return None

    def _make_signal(
        self,
        record: PatentRecord,
        candidate: ConceptAliasCandidate,
        security: IssuerSecurityLink,
        *,
        assignee: str,
        theme_id: str,
        event_date: date,
    ) -> PatentSignal:
        requires_review = candidate.requires_review
        confidence = _link_confidence(
            candidate.alias.confidence,
            record.event_type,
            requires_review,
        )
        classes = [*record.cpc_classes, *record.ipc_classes]
        return PatentSignal(
            patent_id=record.patent_id or record.application_id,
            patent_family_id=record.patent_family_id,
            event_type=record.event_type,
            event_date=event_date,
            title=record.title,
            issuer_concept_id=candidate.concept.concept_id,
            security_concept_id=security.security_concept_id,
            theme_id=theme_id,
            confidence=confidence,
            confidence_reasons=[
                f"assignee_alias:{candidate.alias.confidence:.2f}",
                f"classification:{classes[0]}" if classes else "classification:missing",
                f"event_type:{record.event_type}",
            ],
            source_lineage={
                "source": record.source_attribution,
                "patent_id": record.patent_id,
                "application_id": record.application_id,
                "patent_family_id": record.patent_family_id,
                "assignee_alias": assignee,
                "alias_source_attribution": candidate.alias.source_attribution,
                "classes": classes,
                "source_url": record.source_url,
            },
            metadata={
                "requires_review": requires_review,
                "alias_type": candidate.alias.alias_type,
                "alias_review_status": candidate.alias.review_status,
                "security_relationship": security.relationship_type,
                **record.metadata,
            },
            source_url=record.source_url,
            fetched_at=record.fetched_at,
        )


def _normalize_class(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _link_confidence(alias_confidence: float, event_type: str, requires_review: bool) -> float:
    event_weight = 0.95 if event_type == "grant" else 0.85
    review_weight = 0.75 if requires_review else 1.0
    return round(max(0.0, min(1.0, alias_confidence * event_weight * review_weight)), 4)
