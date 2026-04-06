"""Filing adoption score per issuer-theme pair.

Computes whether a theme is operationally reflected in an issuer's
SEC filings, using section content, XBRL facts, and concept mappings.
This is a structural check — it measures whether the theme shows up
in actual disclosure behavior rather than relying on semantic
similarity alone.

Adoption score components:
    - Section coverage: fraction of weighted sections mentioning theme terms
    - Section depth: density of theme terms within matching sections
    - Fact alignment: fraction of theme-relevant XBRL concepts present
    - Temporal consistency: fraction of filing periods showing adoption

The score is decomposed with full traceability: each signal points
back to a specific section or fact, enabling explanation UIs and
downstream divergence logic.

All functions are stateless — the caller fetches filing sections,
XBRL facts, and concept-derived keywords, then passes them here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.filing.alignment import normalize_section_name

# -- Section importance weights -----------------------------------------------

# Higher weights = more operationally significant for adoption scoring.
# MD&A and Business are where companies discuss operational realities.
# Risk Factors captures forward-looking operational concerns.
SECTION_WEIGHTS: dict[str, float] = {
    "management's discussion and analysis": 1.0,
    "risk factors": 0.9,
    "business": 0.8,
    "quantitative and qualitative disclosures about market risk": 0.7,
    "financial statements and supplementary data": 0.6,
    "properties": 0.3,
    "legal proceedings": 0.3,
}

DEFAULT_SECTION_WEIGHT = 0.4

# Composite score component weights (sum to 1.0)
WEIGHT_SECTION_COVERAGE = 0.35
WEIGHT_SECTION_DEPTH = 0.25
WEIGHT_FACT_ALIGNMENT = 0.20
WEIGHT_TEMPORAL_CONSISTENCY = 0.20

# Density at which section signal saturates (mentions per 1000 words)
DENSITY_SATURATION = 20.0


# -- Input dataclasses -------------------------------------------------------


@dataclass(frozen=True)
class SectionInput:
    """A filing section to evaluate for theme adoption.

    Attributes:
        section_id: Deterministic section ID from filing persistence.
        section_name: Human-readable section name.
        content: Raw section text.
        word_count: Number of words.
        filing_period: Filing period label (e.g., "2025-Q4", "2025-10K").
    """

    section_id: str
    section_name: str
    content: str
    word_count: int
    filing_period: str = ""


@dataclass(frozen=True)
class FactInput:
    """An XBRL fact to evaluate for theme adoption.

    Attributes:
        concept_name: XBRL concept (e.g., "ResearchAndDevelopmentExpense").
        value: The fact value as string.
        period_label: Filing period (e.g., "2025-Q4").
    """

    concept_name: str
    value: str
    period_label: str = ""


# -- Signal dataclasses (output evidence) ------------------------------------


@dataclass(frozen=True)
class SectionSignal:
    """Evidence of theme adoption within a filing section.

    Attributes:
        section_id: Which section produced this signal.
        section_name: Human-readable section name.
        matched_terms: Theme keywords found in the section.
        mention_count: Total mentions across all matched terms.
        density: Mentions per 1000 words.
        section_weight: Importance weight of this section type.
        strength: Overall signal strength (0-1).
    """

    section_id: str
    section_name: str
    matched_terms: list[str] = field(default_factory=list)
    mention_count: int = 0
    density: float = 0.0
    section_weight: float = 0.0
    strength: float = 0.0


@dataclass(frozen=True)
class FactSignal:
    """Evidence of theme adoption from XBRL facts.

    Attributes:
        concept_name: XBRL concept that matched.
        value: The fact value.
        period_label: Filing period.
        matched_theme_concept: Which theme XBRL concept linked this.
    """

    concept_name: str
    value: str
    period_label: str = ""
    matched_theme_concept: str = ""


# -- Score decomposition -----------------------------------------------------


@dataclass(frozen=True)
class AdoptionBreakdown:
    """Explainable decomposition of the adoption score.

    Each component is 0-1. The composite is a weighted combination.

    Attributes:
        section_coverage: Fraction of weighted sections with matches.
        section_depth: Density-weighted depth of mentions.
        fact_alignment: Fraction of theme XBRL concepts found.
        temporal_consistency: Fraction of periods showing adoption.
    """

    section_coverage: float
    section_depth: float
    fact_alignment: float
    temporal_consistency: float


# -- Result dataclass --------------------------------------------------------


@dataclass(frozen=True)
class FilingAdoptionScore:
    """Adoption score for an issuer-theme pair.

    Attributes:
        issuer_concept_id: Canonical issuer concept.
        theme_concept_id: Canonical theme concept.
        score: Composite adoption score (0-1).
        breakdown: Decomposed score components.
        section_signals: Per-section evidence.
        fact_signals: Per-fact evidence.
        filing_count: How many filings contributed.
        period_count: Distinct filing periods evaluated.
        periods_with_signal: Periods that showed adoption.
        computed_at: When this score was computed.
    """

    issuer_concept_id: str
    theme_concept_id: str
    score: float
    breakdown: AdoptionBreakdown
    section_signals: list[SectionSignal] = field(default_factory=list)
    fact_signals: list[FactSignal] = field(default_factory=list)
    filing_count: int = 0
    period_count: int = 0
    periods_with_signal: int = 0
    computed_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for publication payloads."""
        return {
            "issuer_concept_id": self.issuer_concept_id,
            "theme_concept_id": self.theme_concept_id,
            "score": round(self.score, 4),
            "breakdown": {
                "section_coverage": round(self.breakdown.section_coverage, 4),
                "section_depth": round(self.breakdown.section_depth, 4),
                "fact_alignment": round(self.breakdown.fact_alignment, 4),
                "temporal_consistency": round(
                    self.breakdown.temporal_consistency, 4
                ),
            },
            "section_signal_count": len(self.section_signals),
            "fact_signal_count": len(self.fact_signals),
            "filing_count": self.filing_count,
            "period_count": self.period_count,
            "periods_with_signal": self.periods_with_signal,
            "computed_at": self.computed_at.isoformat(),
        }


# -- Compute functions (stateless) -------------------------------------------


def _get_section_weight(section_name: str) -> float:
    """Get importance weight for a section by normalized name."""
    normalized = normalize_section_name(section_name)
    if normalized in SECTION_WEIGHTS:
        return SECTION_WEIGHTS[normalized]
    for key, weight in SECTION_WEIGHTS.items():
        if key in normalized or normalized in key:
            return weight
    return DEFAULT_SECTION_WEIGHT


def _count_term_mentions(
    content: str,
    terms: list[str],
    terms_lower: list[str],
) -> dict[str, int]:
    """Count case-insensitive mentions of each term in content.

    Returns dict mapping original term -> count. Only terms with >0 matches.
    Caller provides pre-lowercased terms to avoid per-section re-lowering.
    """
    content_lower = content.lower()
    counts: dict[str, int] = {}
    for term, term_lower in zip(terms, terms_lower, strict=False):
        count = content_lower.count(term_lower)
        if count > 0:
            counts[term] = count
    return counts


def _compute_section_signals(
    sections: list[SectionInput],
    keywords: list[str],
) -> list[SectionSignal]:
    """Score each section for theme keyword presence.

    Returns a SectionSignal for each section that has at least
    one keyword match. Sections with no matches are excluded.
    """
    if not keywords:
        return []

    keywords_lower = [k.lower() for k in keywords]
    signals: list[SectionSignal] = []
    for section in sections:
        if section.word_count == 0:
            continue
        mentions = _count_term_mentions(section.content, keywords, keywords_lower)
        if not mentions:
            continue
        total_mentions = sum(mentions.values())
        density = total_mentions / section.word_count * 1000
        weight = _get_section_weight(section.section_name)
        # Cap density contribution to prevent outlier sections from dominating
        density_factor = min(1.0, density / DENSITY_SATURATION)
        strength = density_factor * weight

        signals.append(
            SectionSignal(
                section_id=section.section_id,
                section_name=section.section_name,
                matched_terms=sorted(mentions.keys()),
                mention_count=total_mentions,
                density=round(density, 2),
                section_weight=weight,
                strength=round(strength, 4),
            )
        )
    return signals


def _compute_fact_signals(
    facts: list[FactInput],
    xbrl_concepts: list[str],
) -> list[FactSignal]:
    """Match XBRL facts against theme-relevant concept names.

    Uses case-insensitive matching on XBRL concept names.
    """
    if not xbrl_concepts:
        return []

    concept_lower = {c.lower(): c for c in xbrl_concepts}
    signals: list[FactSignal] = []
    for fact in facts:
        fact_lower = fact.concept_name.lower()
        if fact_lower in concept_lower:
            signals.append(
                FactSignal(
                    concept_name=fact.concept_name,
                    value=fact.value,
                    period_label=fact.period_label,
                    matched_theme_concept=concept_lower[fact_lower],
                )
            )
    return signals


def _compute_section_coverage(
    sections: list[SectionInput],
    signals: list[SectionSignal],
) -> float:
    """Fraction of weighted section capacity that has theme signals.

    Weighted by section importance — a match in MD&A counts more
    than a match in Properties.
    """
    if not sections:
        return 0.0

    # Compute weights once per section, reuse for both total and matched
    weights = {s.section_id: _get_section_weight(s.section_name) for s in sections}
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    matched_ids = {s.section_id for s in signals}
    matched_weight = sum(w for sid, w in weights.items() if sid in matched_ids)
    return matched_weight / total_weight


def _compute_section_depth(signals: list[SectionSignal]) -> float:
    """Weighted average strength across matching signals.

    Higher when theme terms are dense in important sections.
    """
    if not signals:
        return 0.0
    total = sum(s.strength for s in signals)
    return min(1.0, total / len(signals))


def _compute_fact_alignment(
    xbrl_concepts: list[str],
    signals: list[FactSignal],
) -> float:
    """Fraction of theme XBRL concepts found in actual filings."""
    if not xbrl_concepts:
        return 0.0
    matched = {s.matched_theme_concept.lower() for s in signals}
    return len(matched) / len(xbrl_concepts)


def _compute_temporal_consistency(
    sections: list[SectionInput],
    signals: list[SectionSignal],
) -> tuple[int, int]:
    """Count periods with signals vs total periods.

    Returns (periods_with_signal, total_periods).
    """
    all_periods = {s.filing_period for s in sections if s.filing_period}
    if not all_periods:
        return 0, 0

    signal_section_ids = {s.section_id for s in signals}
    periods_with = {
        s.filing_period
        for s in sections
        if s.section_id in signal_section_ids and s.filing_period
    }
    return len(periods_with), len(all_periods)


def compute_filing_adoption(
    issuer_concept_id: str,
    theme_concept_id: str,
    sections: list[SectionInput],
    facts: list[FactInput],
    keywords: list[str],
    xbrl_concepts: list[str],
    *,
    filing_count: int = 0,
    now: datetime | None = None,
) -> FilingAdoptionScore:
    """Compute filing adoption score for an issuer-theme pair.

    Stateless scorer. Receives pre-fetched filing sections, XBRL facts,
    and theme-derived keywords/concepts. Returns a fully traceable
    adoption score with explanation hooks.

    Args:
        issuer_concept_id: Canonical issuer concept ID.
        theme_concept_id: Canonical theme concept ID.
        sections: Filing sections for this issuer.
        facts: XBRL facts for this issuer.
        keywords: Theme-derived keywords to search in sections.
        xbrl_concepts: Theme-relevant XBRL concept names.
        filing_count: Number of distinct filings (for metadata).
        now: Current time for timestamp.

    Returns:
        FilingAdoptionScore with decomposed breakdown and signals.
    """
    if now is None:
        now = datetime.now(UTC)

    # Compute signals
    section_signals = _compute_section_signals(sections, keywords)
    fact_signals = _compute_fact_signals(facts, xbrl_concepts)

    # Compute breakdown components
    section_coverage = _compute_section_coverage(sections, section_signals)
    section_depth = _compute_section_depth(section_signals)
    fact_alignment = _compute_fact_alignment(xbrl_concepts, fact_signals)
    periods_with, period_count = _compute_temporal_consistency(
        sections, section_signals
    )
    temporal_consistency = (
        periods_with / period_count if period_count > 0 else 0.0
    )

    breakdown = AdoptionBreakdown(
        section_coverage=section_coverage,
        section_depth=section_depth,
        fact_alignment=fact_alignment,
        temporal_consistency=temporal_consistency,
    )

    # Composite score
    score = (
        WEIGHT_SECTION_COVERAGE * section_coverage
        + WEIGHT_SECTION_DEPTH * section_depth
        + WEIGHT_FACT_ALIGNMENT * fact_alignment
        + WEIGHT_TEMPORAL_CONSISTENCY * temporal_consistency
    )

    return FilingAdoptionScore(
        issuer_concept_id=issuer_concept_id,
        theme_concept_id=theme_concept_id,
        score=round(score, 4),
        breakdown=breakdown,
        section_signals=section_signals,
        fact_signals=fact_signals,
        filing_count=filing_count,
        period_count=period_count,
        periods_with_signal=periods_with,
        computed_at=now,
    )
