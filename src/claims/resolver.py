"""Deterministic resolver cascade for entity/relation grounding.

Resolves raw text mentions to canonical concept IDs using a tiered
cascade that handles common cases cheaply before any LLM fallback.

Tier order:
    1. Exact: ticker, CIK, or exact concept name lookup
    2. Alias: case-insensitive alias dictionary resolution
    3. Fuzzy: pg_trgm similarity search with configurable threshold
    4. LLM Proposed: gated fallback for high-value unresolved mentions

Tiers 1-3 are deterministic. Tier 4 is gated by FallbackGate policy
and produces LLM_PROPOSED results that require review before becoming
authoritative. Each tier produces a ResolverResult with confidence
decomposition so downstream review logic can see which tier resolved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.claims.llm_gate import FallbackGate
from src.security_master.concept_repository import ConceptRepository
from src.security_master.concept_schemas import Concept

logger = logging.getLogger(__name__)


class ResolverTier(str, Enum):
    """Which resolution tier produced the result."""

    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    LLM_PROPOSED = "llm_proposed"
    UNRESOLVED = "unresolved"


@dataclass
class ResolverResult:
    """Result of attempting to resolve a text mention to a concept.

    Attributes:
        mention: The raw text that was resolved.
        concept: The resolved concept (None if unresolved).
        concept_id: Shortcut to concept.concept_id (None if unresolved).
        tier: Which resolution tier produced this result.
        confidence: Resolution confidence (0-1).
        alternatives: Other candidate concepts considered.
        metadata: Extensible metadata (similarity scores, etc.).
    """

    mention: str
    concept: Concept | None = None
    concept_id: str | None = None
    tier: ResolverTier = ResolverTier.UNRESOLVED
    confidence: float = 0.0
    alternatives: list[Concept] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        """Whether the mention was successfully resolved."""
        return self.concept is not None


class EntityResolver:
    """Tiered deterministic resolver for entity grounding.

    Resolves raw text mentions to canonical concept IDs through
    a cascade of increasingly expensive tiers. Designed to handle
    the common cases cheaply (exact ticker/alias) and only fall
    through to fuzzy matching for genuinely ambiguous mentions.

    Usage:
        resolver = EntityResolver(concept_repo)
        result = await resolver.resolve("TSMC")
        if result.resolved:
            print(f"Resolved to {result.concept_id} via {result.tier}")
    """

    def __init__(
        self,
        concept_repo: ConceptRepository,
        *,
        fuzzy_threshold: float = 0.4,
        fuzzy_limit: int = 5,
        fallback_gate: FallbackGate | None = None,
    ) -> None:
        self._repo = concept_repo
        self._fuzzy_threshold = fuzzy_threshold
        self._fuzzy_limit = fuzzy_limit
        self._fallback_gate = fallback_gate

    async def resolve(
        self,
        mention: str,
        *,
        concept_type: str | None = None,
        passage_length: int = 0,
        predicate: str | None = None,
        run_id: str | None = None,
    ) -> ResolverResult:
        """Resolve a text mention to a canonical concept.

        Runs through the tier cascade in order, returning the first
        successful resolution. If no tier resolves and a fallback gate
        is configured, evaluates whether LLM fallback is warranted.

        Args:
            mention: Raw text mention to resolve.
            concept_type: Optional filter (e.g., "issuer", "technology").
            passage_length: Character length of the source passage
                (used by the LLM fallback gate to assess value).
            predicate: The claim predicate, if known at resolution time.
            run_id: Current lane run ID (for per-run budget tracking).

        Returns:
            ResolverResult with concept, tier, and confidence.
        """
        mention = mention.strip()
        if not mention:
            return ResolverResult(mention=mention)

        # Tier 1: Exact lookup (ticker or concept_id)
        result = await self._resolve_exact(mention, concept_type)
        if result.resolved:
            return result

        # Tier 2: Alias dictionary (case-insensitive)
        result = await self._resolve_alias(mention)
        if result.resolved:
            return result

        # Tier 3: Fuzzy matching (pg_trgm similarity)
        result = await self._resolve_fuzzy(mention, concept_type)
        if result.resolved:
            return result

        # Tier 4: LLM fallback (gated — only if gate approves)
        if self._fallback_gate is not None:
            result = self._evaluate_llm_gate(
                mention,
                passage_length=passage_length,
                predicate=predicate,
                run_id=run_id,
            )
            return result

        logger.debug("Unresolved mention: %r", mention)
        return ResolverResult(mention=mention)

    async def resolve_batch(
        self,
        mentions: list[str],
        *,
        concept_type: str | None = None,
        passage_length: int = 0,
        predicate: str | None = None,
        run_id: str | None = None,
    ) -> list[ResolverResult]:
        """Resolve multiple mentions from the same source context.

        All mentions share passage_length, predicate, and run_id because
        they originate from the same passage (e.g., subject and object of
        one claim). For mentions from different passages, call resolve()
        individually.

        If any single resolution fails with a DB error, that mention
        gets an unresolved result rather than aborting the entire batch.
        """
        import asyncio

        results = await asyncio.gather(
            *(
                self.resolve(
                    m,
                    concept_type=concept_type,
                    passage_length=passage_length,
                    predicate=predicate,
                    run_id=run_id,
                )
                for m in mentions
            ),
            return_exceptions=True,
        )
        return [
            r if isinstance(r, ResolverResult) else ResolverResult(mention=m)
            for r, m in zip(results, mentions, strict=False)
        ]

    # -- Tier 4: LLM fallback gate -----------------------------------------

    def _evaluate_llm_gate(
        self,
        mention: str,
        *,
        passage_length: int = 0,
        predicate: str | None = None,
        run_id: str | None = None,
    ) -> ResolverResult:
        """Evaluate the LLM fallback gate and return an appropriate result.

        This method does NOT call the LLM — it only evaluates the gate
        policy. If approved, it returns an LLM_PROPOSED result with
        gate metadata so the caller knows an LLM call is warranted.
        The actual LLM call is the caller's responsibility.

        If denied, returns an unresolved result with gate denial info
        in metadata for audit.
        """
        decision = self._fallback_gate.evaluate(
            mention,
            passage_length=passage_length,
            predicate=predicate,
            run_id=run_id,
        )

        if decision.approved:
            # Signal to caller that LLM fallback is warranted.
            # concept is None — the caller must fill it after the LLM call.
            return ResolverResult(
                mention=mention,
                tier=ResolverTier.LLM_PROPOSED,
                confidence=0.0,  # Will be set after LLM call
                metadata={
                    "match_type": "llm_gate_approved",
                    "gate_verdict": decision.verdict.value,
                    "passage_length": decision.passage_length,
                    "predicate": decision.predicate,
                },
            )

        # Gate denied — return unresolved with denial info
        logger.debug(
            "LLM gate denied for %r: %s", mention, decision.deny_reason
        )
        return ResolverResult(
            mention=mention,
            metadata={
                "gate_verdict": decision.verdict.value,
                "gate_deny_reason": (
                    decision.deny_reason.value if decision.deny_reason else None
                ),
            },
        )

    # -- Tier 1: Exact lookup ----------------------------------------------

    async def _resolve_exact(
        self, mention: str, concept_type: str | None
    ) -> ResolverResult:
        """Try exact ticker or concept_id lookup."""
        # Try as ticker → security concept (US exchange first, then
        # try with exchange suffix like "005930.KS")
        ticker = mention.upper()
        exchange = "US"
        if "." in ticker:
            # Handle exchange suffix (e.g., "005930.KS" → ticker="005930", exchange="KS")
            # For ambiguous multi-dot inputs (e.g., "A.B.C"), use the last segment
            # as exchange and log for visibility.
            parts = ticker.rsplit(".", 1)
            ticker, exchange = parts[0], parts[1]
        concept = await self._repo.get_concept_for_security(
            ticker=ticker, exchange=exchange
        )
        if concept is not None and (
            concept_type is None or concept.concept_type == concept_type
        ):
                return ResolverResult(
                    mention=mention,
                    concept=concept,
                    concept_id=concept.concept_id,
                    tier=ResolverTier.EXACT,
                    confidence=1.0,
                    metadata={"match_type": "ticker"},
                )

        # Try direct concept lookup by ID (for pre-resolved references)
        if mention.startswith("concept_"):
            concept = await self._repo.get_concept(mention)
            if concept is not None:
                return ResolverResult(
                    mention=mention,
                    concept=concept,
                    concept_id=concept.concept_id,
                    tier=ResolverTier.EXACT,
                    confidence=1.0,
                    metadata={"match_type": "concept_id"},
                )

        return ResolverResult(mention=mention)

    # -- Tier 2: Alias dictionary ------------------------------------------

    async def _resolve_alias(self, mention: str) -> ResolverResult:
        """Try case-insensitive alias resolution."""
        concept = await self._repo.resolve_alias(mention)
        if concept is not None:
            return ResolverResult(
                mention=mention,
                concept=concept,
                concept_id=concept.concept_id,
                tier=ResolverTier.ALIAS,
                confidence=0.95,
                metadata={"match_type": "alias"},
            )
        return ResolverResult(mention=mention)

    # -- Tier 3: Fuzzy matching --------------------------------------------

    async def _resolve_fuzzy(
        self, mention: str, concept_type: str | None
    ) -> ResolverResult:
        """Try fuzzy matching via pg_trgm similarity."""
        candidates = await self._repo.search_concepts(
            mention,
            limit=self._fuzzy_limit,
            min_similarity=self._fuzzy_threshold,
        )
        if not candidates:
            return ResolverResult(mention=mention)

        # Filter by concept_type if specified
        if concept_type is not None:
            candidates = [
                c for c in candidates if c.concept_type == concept_type
            ]
            if not candidates:
                return ResolverResult(mention=mention)

        best = candidates[0]
        # search_concepts returns results ordered by similarity DESC,
        # but we don't have the exact score here. Assign confidence
        # based on position (best candidate gets highest confidence).
        confidence = 0.7 if len(candidates) == 1 else 0.6

        return ResolverResult(
            mention=mention,
            concept=best,
            concept_id=best.concept_id,
            tier=ResolverTier.FUZZY,
            confidence=confidence,
            alternatives=candidates[1:],
            metadata={"match_type": "fuzzy", "candidate_count": len(candidates)},
        )
