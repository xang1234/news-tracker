"""Tests for the deterministic entity resolver cascade.

Uses an in-memory mock ConceptRepository to test each tier
independently and the full cascade end-to-end.
"""

from __future__ import annotations

import pytest

from src.claims.resolver import EntityResolver, ResolverResult, ResolverTier
from src.security_master.concept_schemas import Concept

# -- Mock concept repository -----------------------------------------------


class _MockConceptRepo:
    """In-memory concept repository for resolver testing."""

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.aliases: dict[str, Concept] = {}  # lowercase alias → concept
        self.ticker_map: dict[str, Concept] = {}  # uppercase ticker → concept

    def add_concept(
        self,
        concept_id: str,
        concept_type: str,
        name: str,
        *,
        ticker: str | None = None,
        aliases: list[str] | None = None,
    ) -> Concept:
        c = Concept(
            concept_id=concept_id,
            concept_type=concept_type,
            canonical_name=name,
        )
        self.concepts[concept_id] = c
        if ticker:
            self.ticker_map[ticker.upper()] = c
        for alias in aliases or []:
            self.aliases[alias.lower()] = c
        # Also index the canonical name as an alias
        self.aliases[name.lower()] = c
        return c

    async def get_concept(self, concept_id: str) -> Concept | None:
        return self.concepts.get(concept_id)

    async def get_concept_for_security(self, ticker: str, exchange: str = "US") -> Concept | None:
        return self.ticker_map.get(ticker.upper())

    async def resolve_alias(self, alias: str) -> Concept | None:
        return self.aliases.get(alias.lower())

    async def search_concepts(
        self, query: str, limit: int = 10, *, min_similarity: float = 0.2
    ) -> list[Concept]:
        """Simple substring match for testing fuzzy tier."""
        query_lower = query.lower()
        matches = [c for c in self.concepts.values() if query_lower in c.canonical_name.lower()]
        return matches[:limit]


@pytest.fixture()
def repo() -> _MockConceptRepo:
    r = _MockConceptRepo()
    r.add_concept(
        "concept_issuer_tsmc",
        "issuer",
        "Taiwan Semiconductor Manufacturing",
        ticker="TSM",
        aliases=["TSMC", "Taiwan Semi", "TSM"],
    )
    r.add_concept(
        "concept_issuer_nvda",
        "issuer",
        "NVIDIA Corporation",
        ticker="NVDA",
        aliases=["NVIDIA", "Jensen Huang"],
    )
    r.add_concept(
        "concept_tech_hbm",
        "technology",
        "High Bandwidth Memory",
        aliases=["HBM", "HBM3", "HBM3E"],
    )
    return r


@pytest.fixture()
def resolver(repo: _MockConceptRepo) -> EntityResolver:
    return EntityResolver(repo)


# -- Tier 1: Exact lookup tests -------------------------------------------


class TestExactTier:
    """Tier 1: exact ticker/CIK/concept_id resolution."""

    async def test_resolve_by_ticker(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("NVDA")
        assert result.resolved
        assert result.concept_id == "concept_issuer_nvda"
        assert result.tier == ResolverTier.EXACT
        assert result.confidence == 1.0

    async def test_resolve_by_ticker_case_insensitive(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("nvda")
        assert result.resolved
        assert result.concept_id == "concept_issuer_nvda"
        assert result.tier == ResolverTier.EXACT

    async def test_resolve_by_concept_id(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("concept_issuer_tsmc")
        assert result.resolved
        assert result.concept_id == "concept_issuer_tsmc"
        assert result.tier == ResolverTier.EXACT

    async def test_exact_miss_falls_through(self, resolver: EntityResolver) -> None:
        """Unknown ticker falls through to later tiers."""
        result = await resolver.resolve("UNKNOWN_TICKER_XYZ")
        # Will fall through to alias and fuzzy
        assert result.tier != ResolverTier.EXACT or not result.resolved


# -- Tier 2: Alias resolution tests ---------------------------------------


class TestAliasTier:
    """Tier 2: case-insensitive alias dictionary resolution."""

    async def test_resolve_by_alias(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("TSMC")
        assert result.resolved
        assert result.concept_id == "concept_issuer_tsmc"
        assert result.tier in (ResolverTier.EXACT, ResolverTier.ALIAS)
        assert result.confidence >= 0.95

    async def test_resolve_by_full_name(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("Taiwan Semiconductor Manufacturing")
        assert result.resolved
        assert result.concept_id == "concept_issuer_tsmc"

    async def test_resolve_by_abbreviation(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("HBM")
        assert result.resolved
        assert result.concept_id == "concept_tech_hbm"

    async def test_resolve_case_insensitive(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("tsmc")
        assert result.resolved
        assert result.concept_id == "concept_issuer_tsmc"

    async def test_alias_miss(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("Completely Unknown Entity")
        assert not result.resolved
        assert result.tier == ResolverTier.UNRESOLVED


# -- Tier 3: Fuzzy matching tests -----------------------------------------


class TestFuzzyTier:
    """Tier 3: fuzzy matching via substring/similarity."""

    async def test_fuzzy_partial_match(self, resolver: EntityResolver) -> None:
        """Substring match finds the concept."""
        result = await resolver.resolve("Bandwidth Memory")
        assert result.resolved
        assert result.concept_id == "concept_tech_hbm"
        assert result.tier == ResolverTier.FUZZY
        assert result.confidence > 0

    async def test_fuzzy_with_type_filter(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("Semiconductor", concept_type="issuer")
        assert result.resolved
        assert result.concept.concept_type == "issuer"

    async def test_fuzzy_no_match(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("Quantum Computing Startup XYZ")
        assert not result.resolved
        assert result.tier == ResolverTier.UNRESOLVED


# -- Full cascade tests ----------------------------------------------------


class TestFullCascade:
    """End-to-end resolution through the full cascade."""

    async def test_empty_mention(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("")
        assert not result.resolved

    async def test_whitespace_mention(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("   ")
        assert not result.resolved

    async def test_resolve_batch(self, resolver: EntityResolver) -> None:
        results = await resolver.resolve_batch(["NVDA", "TSMC", "Unknown"])
        assert len(results) == 3
        assert results[0].resolved
        assert results[1].resolved
        assert not results[2].resolved

    async def test_confidence_ordering(self, resolver: EntityResolver) -> None:
        """Exact match has higher confidence than alias."""
        exact = await resolver.resolve("NVDA")  # ticker → exact
        # "Jensen Huang" is an alias, not a ticker
        alias = await resolver.resolve("Jensen Huang")
        assert exact.confidence >= alias.confidence

    async def test_result_metadata(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("TSM")
        assert "match_type" in result.metadata

    async def test_unresolved_has_no_concept(self, resolver: EntityResolver) -> None:
        result = await resolver.resolve("zzz_nonexistent_zzz")
        assert result.concept is None
        assert result.concept_id is None
        assert result.confidence == 0.0


class TestResolverResult:
    """ResolverResult dataclass."""

    def test_resolved_property(self) -> None:
        r = ResolverResult(
            mention="NVDA",
            concept=Concept(
                concept_id="c1",
                concept_type="issuer",
                canonical_name="NVIDIA",
            ),
            concept_id="c1",
        )
        assert r.resolved is True

    def test_unresolved_property(self) -> None:
        r = ResolverResult(mention="unknown")
        assert r.resolved is False

    def test_tier_enum_values(self) -> None:
        assert ResolverTier.EXACT == "exact"
        assert ResolverTier.ALIAS == "alias"
        assert ResolverTier.FUZZY == "fuzzy"
        assert ResolverTier.LLM_PROPOSED == "llm_proposed"
        assert ResolverTier.UNRESOLVED == "unresolved"
