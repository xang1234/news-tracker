"""Claim reconciliation engine: compose contradiction/corroboration tiers.

A single engine owns each assertion. Per incoming claim it:

    1. determines the assertion triple (subject, predicate, object),
    2. gathers candidate claims from every applicable tier,
    3. asks each tier for a per-claim ``support``/``contradiction`` *opinion*,
    4. merges opinions **contradiction-dominant** (any contradiction wins),
    5. recomputes and persists the assertion + links exactly once.

Tiers are pure classifiers — they never persist — so they cannot clobber one
another. Adding a tier (e.g. a semantic NLI tier) means appending one object
to the ``tiers`` list; the engine and persist path are untouched.
"""

from __future__ import annotations

from typing import Protocol

import structlog

from src.assertions.numeric_contradiction import numeric_link_types
from src.assertions.predicate_contradiction import (
    antonym_of,
    polarity_link_types,
    validity_overlaps,
)
from src.assertions.recompute import recompute_assertion
from src.assertions.schemas import AssertionClaimLink, ResolvedAssertion, make_assertion_id
from src.claims.schemas import EvidenceClaim

logger = structlog.get_logger(__name__)


class ContradictionTier(Protocol):
    """A reconciliation tier: a pure classifier the engine fans in.

    Tiers never persist. ``classify`` returns a ``{claim_id: link_type}``
    opinion (``support``/``contradiction``); the engine merges opinions across
    tiers and writes the assertion once.
    """

    def applies_to(self, claim: EvidenceClaim) -> bool: ...

    async def fetch_candidates(self, claim: EvidenceClaim, claim_repo) -> list[EvidenceClaim]: ...

    async def classify(
        self, claim: EvidenceClaim, claims: list[EvidenceClaim]
    ) -> dict[str, str]: ...


def merge_link_types(opinions: list[dict[str, str]]) -> dict[str, str]:
    """Merge per-claim link opinions, contradiction-dominant.

    A claim is ``contradiction`` if *any* tier judges it so, else ``support``
    if any tier supports it. Claims no tier judged are absent from the result.
    """
    merged: dict[str, str] = {}
    for opinion in opinions:
        for claim_id, link_type in opinion.items():
            if merged.get(claim_id) == "contradiction":
                continue  # contradiction is sticky
            merged[claim_id] = link_type
    return merged


# -- Tiers (pure classifiers) ----------------------------------------------


class NumericTier:
    """Numeric-fact disagreement on the same metric/period contradicts."""

    def applies_to(self, claim: EvidenceClaim) -> bool:
        return claim.numeric_value is not None and claim.metric is not None

    async def fetch_candidates(self, claim: EvidenceClaim, claim_repo) -> list[EvidenceClaim]:
        return await claim_repo.list_comparable_numeric_claims(
            subject_concept_id=claim.subject_concept_id,
            metric=claim.metric,
            period=claim.period,
        )

    async def classify(self, claim: EvidenceClaim, claims: list[EvidenceClaim]) -> dict[str, str]:
        numeric = [c for c in claims if c.numeric_value is not None and c.metric is not None]
        return numeric_link_types(numeric)


class PredicateContradictionTier:
    """Antonym predicates on the same subject with overlapping validity contradict."""

    def applies_to(self, claim: EvidenceClaim) -> bool:
        return antonym_of(claim.predicate) is not None

    async def fetch_candidates(self, claim: EvidenceClaim, claim_repo) -> list[EvidenceClaim]:
        return await claim_repo.list_claims_by_subject_predicates(
            subject_concept_id=claim.subject_concept_id,
            predicates=[claim.predicate, antonym_of(claim.predicate)],
        )

    async def classify(self, claim: EvidenceClaim, claims: list[EvidenceClaim]) -> dict[str, str]:
        overlapping = [
            c for c in claims if c.claim_id == claim.claim_id or validity_overlaps(claim, c)
        ]
        return polarity_link_types(claim.predicate, overlapping)


class CorroborationTier:
    """Same (subject, predicate, object) claims corroborate (support)."""

    def applies_to(self, claim: EvidenceClaim) -> bool:
        return claim.subject_concept_id is not None

    async def fetch_candidates(self, claim: EvidenceClaim, claim_repo) -> list[EvidenceClaim]:
        return await claim_repo.list_claims_by_subject_predicates(
            subject_concept_id=claim.subject_concept_id,
            predicates=[claim.predicate],
        )

    async def classify(self, claim: EvidenceClaim, claims: list[EvidenceClaim]) -> dict[str, str]:
        return {
            c.claim_id: "support"
            for c in claims
            if c.predicate == claim.predicate and c.object_concept_id == claim.object_concept_id
        }


class SemanticTier:
    """LLM-judged contradiction for residual same-subject/same-predicate pairs.

    Applies only where the deterministic tiers don't: a resolved subject, a
    non-numeric claim, and a predicate with no antonym. Asks an injected judge
    whether each (bounded) same-predicate candidate contradicts the incoming
    claim, emitting a ``contradiction`` opinion only for high-confidence
    ``contradicts`` verdicts. Support is left to the corroboration tier.
    """

    def __init__(
        self, judge, *, confidence_threshold: float = 0.7, max_candidates: int = 5
    ) -> None:
        self._judge = judge
        self._threshold = confidence_threshold
        self._max_candidates = max_candidates

    def applies_to(self, claim: EvidenceClaim) -> bool:
        return (
            claim.subject_concept_id is not None
            and claim.numeric_value is None
            and antonym_of(claim.predicate) is None
        )

    async def fetch_candidates(self, claim: EvidenceClaim, claim_repo) -> list[EvidenceClaim]:
        return await claim_repo.list_claims_by_subject_predicates(
            subject_concept_id=claim.subject_concept_id,
            predicates=[claim.predicate],
        )

    @staticmethod
    def _text(claim: EvidenceClaim) -> str:
        if claim.source_text:
            return claim.source_text
        return f"{claim.subject_text} {claim.predicate} {claim.object_text or ''}".strip()

    async def classify(self, claim: EvidenceClaim, claims: list[EvidenceClaim]) -> dict[str, str]:
        candidates = [
            c for c in claims if c.claim_id != claim.claim_id and c.predicate == claim.predicate
        ][: self._max_candidates]

        opinions: dict[str, str] = {}
        for candidate in candidates:
            verdict = await self._judge.judge(self._text(claim), self._text(candidate))
            if (
                verdict is not None
                and verdict.relation == "contradicts"
                and verdict.confidence >= self._threshold
            ):
                opinions[candidate.claim_id] = "contradiction"
            elif verdict is not None:
                logger.debug(
                    "Semantic verdict below contradiction threshold",
                    claim_id=candidate.claim_id,
                    relation=verdict.relation,
                    confidence=verdict.confidence,
                )
        return opinions


# -- Engine ----------------------------------------------------------------


class ClaimReconciliationEngine:
    """Reconciles a claim's assertion across all applicable tiers, once."""

    def __init__(self, claim_repo, assertion_repo, *, tiers: list[ContradictionTier]) -> None:
        self._claim_repo = claim_repo
        self._assertion_repo = assertion_repo
        self._tiers = tiers

    async def reconcile_claim(self, claim: EvidenceClaim) -> ResolvedAssertion | None:
        """Reconcile the incoming claim's assertion; returns it or None."""
        if claim.subject_concept_id is None:
            return None

        applicable = [t for t in self._tiers if t.applies_to(claim)]
        if not applicable:
            return None

        # Gather candidate claims (incoming always included).
        claims_by_id: dict[str, EvidenceClaim] = {claim.claim_id: claim}
        for tier in applicable:
            for candidate in await tier.fetch_candidates(claim, self._claim_repo):
                claims_by_id.setdefault(candidate.claim_id, candidate)
        claims = list(claims_by_id.values())

        # Collect and merge each tier's opinion.
        merged = merge_link_types([await tier.classify(claim, claims) for tier in applicable])
        if not merged:
            return None

        assertion_id = make_assertion_id(
            claim.subject_concept_id, claim.predicate, claim.object_concept_id
        )
        links = [
            AssertionClaimLink(
                assertion_id=assertion_id,
                claim_id=claim_id,
                link_type=link_type,
                contribution_weight=1.0,
            )
            for claim_id, link_type in merged.items()
        ]
        linked_claims = [claims_by_id[claim_id] for claim_id in merged]

        existing = await self._assertion_repo.get_assertion(assertion_id)
        assertion, _delta = recompute_assertion(
            existing,
            linked_claims,
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
                "Claim contradiction detected",
                assertion_id=assertion_id,
                subject_concept_id=claim.subject_concept_id,
                predicate=claim.predicate,
                contradiction_count=assertion.contradiction_count,
            )

        return assertion
