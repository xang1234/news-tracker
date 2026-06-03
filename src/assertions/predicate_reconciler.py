"""Predicate-polarity reconciliation orchestrator.

Sibling of the numeric reconciler: given a resolved claim whose predicate has
an antonym, find same-subject claims on either polarity side, keep those whose
validity window overlaps the incoming claim, classify support/contradiction,
recompute the assertion, and persist — flipping it to ``disputed`` when an
antonym claim is present.

Only claims with a resolved subject and a polarity-bearing predicate are
reconciled; everything else is a no-op (returns ``None``).
"""

from __future__ import annotations

import structlog

from src.assertions.predicate_contradiction import (
    antonym_of,
    classify_polarity_links,
    validity_overlaps,
)
from src.assertions.recompute import recompute_assertion
from src.assertions.schemas import ResolvedAssertion, make_assertion_id
from src.claims.schemas import EvidenceClaim

logger = structlog.get_logger(__name__)


class PredicateContradictionReconciler:
    """Reconciles a polarity-bearing claim against its antonym counterparts."""

    def __init__(self, claim_repo, assertion_repo) -> None:
        self._claim_repo = claim_repo
        self._assertion_repo = assertion_repo

    @staticmethod
    def _is_reconcilable(claim: EvidenceClaim) -> bool:
        """Reconcilable only if resolved and the predicate has a polarity."""
        return claim.subject_concept_id is not None and antonym_of(claim.predicate) is not None

    async def reconcile_claim(self, claim: EvidenceClaim) -> ResolvedAssertion | None:
        """Reconcile one polarity-bearing claim; returns the assertion or None."""
        if not self._is_reconcilable(claim):
            return None

        antonym = antonym_of(claim.predicate)
        comparable = await self._claim_repo.list_claims_by_subject_predicates(
            subject_concept_id=claim.subject_concept_id,
            predicates=[claim.predicate, antonym],
        )

        # Fold the incoming claim in (it may not be visible to the lookup yet)
        # and keep only claims whose validity overlaps it.
        claims_by_id: dict[str, EvidenceClaim] = {c.claim_id: c for c in comparable}
        claims_by_id[claim.claim_id] = claim
        overlapping = [
            c
            for c in claims_by_id.values()
            if c.claim_id == claim.claim_id or validity_overlaps(claim, c)
        ]

        assertion_id = make_assertion_id(
            claim.subject_concept_id,
            claim.predicate,
            claim.object_concept_id,
        )
        links = classify_polarity_links(assertion_id, claim.predicate, overlapping)

        existing = await self._assertion_repo.get_assertion(assertion_id)
        assertion, _delta = recompute_assertion(
            existing,
            overlapping,
            links,
            subject_concept_id=claim.subject_concept_id,
            predicate=claim.predicate,
            object_concept_id=claim.object_concept_id,
        )

        await self._assertion_repo.upsert_assertion(assertion)
        for link in links:
            await self._assertion_repo.upsert_link(link)

        if assertion.status == "disputed":
            logger.info(
                "Predicate-polarity contradiction detected",
                assertion_id=assertion_id,
                subject_concept_id=claim.subject_concept_id,
                predicate=claim.predicate,
                antonym=antonym,
                contradiction_count=assertion.contradiction_count,
            )

        return assertion
