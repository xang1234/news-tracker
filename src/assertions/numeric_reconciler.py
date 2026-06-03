"""Numeric reconciliation orchestrator.

The live-pipeline glue that turns a resolved numeric claim into a persisted
(possibly ``disputed``) assertion:

    1. Find comparable numeric facts (same subject_concept_id + metric +
       period) via the claim repository.
    2. Fold the incoming claim into that set (it may not be persisted/visible
       to the lookup yet).
    3. Classify each fact as support/contradiction (``classify_numeric_links``).
    4. Recompute the assertion from all facts + links (``recompute_assertion``,
       which flips status to ``disputed`` on contradiction).
    5. Persist the assertion and every claim link.

Only numeric claims with a resolved subject are reconciled; everything else is
a no-op (returns ``None``). The repositories are injected so the orchestrator
is unit-testable against in-memory fakes.
"""

from __future__ import annotations

import structlog

from src.assertions.numeric_contradiction import classify_numeric_links
from src.assertions.recompute import recompute_assertion
from src.assertions.schemas import ResolvedAssertion, make_assertion_id
from src.claims.numeric import DEFAULT_REL_TOLERANCE
from src.claims.schemas import EvidenceClaim

logger = structlog.get_logger(__name__)


class NumericReconciler:
    """Reconciles a numeric claim against comparable facts into an assertion."""

    def __init__(
        self,
        claim_repo,
        assertion_repo,
        *,
        rel_tolerance: float = DEFAULT_REL_TOLERANCE,
    ) -> None:
        self._claim_repo = claim_repo
        self._assertion_repo = assertion_repo
        self._rel_tolerance = rel_tolerance

    @staticmethod
    def _is_reconcilable(claim: EvidenceClaim) -> bool:
        """A claim is reconcilable only if it is a resolved numeric fact."""
        return (
            claim.numeric_value is not None
            and claim.metric is not None
            and claim.subject_concept_id is not None
        )

    async def reconcile_claim(self, claim: EvidenceClaim) -> ResolvedAssertion | None:
        """Reconcile one numeric claim; returns the updated assertion or None.

        Returns ``None`` (a no-op) for non-numeric or unresolved claims.
        """
        if not self._is_reconcilable(claim):
            return None

        comparable = await self._claim_repo.list_comparable_numeric_claims(
            subject_concept_id=claim.subject_concept_id,
            metric=claim.metric,
            period=claim.period,
        )

        # The incoming claim may not yet be visible to the lookup (just
        # written, or written in the same transaction) — fold it in by id.
        claims_by_id: dict[str, EvidenceClaim] = {c.claim_id: c for c in comparable}
        claims_by_id[claim.claim_id] = claim
        claims = list(claims_by_id.values())

        assertion_id = make_assertion_id(
            claim.subject_concept_id,
            claim.predicate,
            claim.object_concept_id,
        )
        links = classify_numeric_links(assertion_id, claims, rel_tolerance=self._rel_tolerance)

        existing = await self._assertion_repo.get_assertion(assertion_id)
        assertion, _delta = recompute_assertion(
            existing,
            claims,
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
                "Numeric contradiction detected",
                assertion_id=assertion_id,
                subject_concept_id=claim.subject_concept_id,
                metric=claim.metric,
                period=claim.period,
                contradiction_count=assertion.contradiction_count,
            )

        return assertion
