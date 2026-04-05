"""Tests for filing adoption score per issuer-theme pair.

Verifies that section coverage, section depth, XBRL fact alignment,
and temporal consistency combine into a traceable adoption score.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.filing.adoption import (
    DENSITY_SATURATION,
    DEFAULT_SECTION_WEIGHT,
    WEIGHT_FACT_ALIGNMENT,
    WEIGHT_SECTION_COVERAGE,
    WEIGHT_SECTION_DEPTH,
    WEIGHT_TEMPORAL_CONSISTENCY,
    AdoptionBreakdown,
    FactInput,
    FactSignal,
    FilingAdoptionScore,
    SectionInput,
    SectionSignal,
    compute_filing_adoption,
    _compute_fact_signals,
    _compute_section_coverage,
    _compute_section_depth,
    _compute_section_signals,
    _compute_temporal_consistency,
    _count_term_mentions,
    _get_section_weight,
)

NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)

ISSUER = "concept_issuer_abc123"
THEME = "concept_theme_xyz789"


# -- Helpers ---------------------------------------------------------------


def _make_section(
    section_id: str = "sec_001",
    section_name: str = "Risk Factors",
    content: str = "HBM memory demand is accelerating across data center GPUs.",
    word_count: int | None = None,
    filing_period: str = "2025-10K",
) -> SectionInput:
    if word_count is None:
        word_count = len(content.split())
    return SectionInput(
        section_id=section_id,
        section_name=section_name,
        content=content,
        word_count=word_count,
        filing_period=filing_period,
    )


def _make_fact(
    concept_name: str = "ResearchAndDevelopmentExpense",
    value: str = "1500000000",
    period_label: str = "2025-Q4",
) -> FactInput:
    return FactInput(
        concept_name=concept_name,
        value=value,
        period_label=period_label,
    )


# -- Term counting tests ---------------------------------------------------


def _count(content: str, terms: list[str]) -> dict[str, int]:
    """Test helper wrapping _count_term_mentions with pre-lowered terms."""
    return _count_term_mentions(content, terms, [t.lower() for t in terms])


class TestTermCounting:
    """Case-insensitive term matching in section content."""

    def test_single_term(self) -> None:
        counts = _count("HBM memory is growing. HBM demand.", ["HBM"])
        assert counts == {"HBM": 2}

    def test_case_insensitive(self) -> None:
        counts = _count("hbm HBM Hbm", ["HBM"])
        assert counts == {"HBM": 3}

    def test_multiple_terms(self) -> None:
        counts = _count(
            "HBM memory is used in GPU data centers",
            ["HBM", "GPU", "DRAM"],
        )
        assert counts == {"HBM": 1, "GPU": 1}
        assert "DRAM" not in counts

    def test_no_matches(self) -> None:
        counts = _count("nothing relevant here", ["HBM"])
        assert counts == {}

    def test_empty_content(self) -> None:
        counts = _count("", ["HBM"])
        assert counts == {}

    def test_empty_terms(self) -> None:
        counts = _count("some content", [])
        assert counts == {}

    def test_substring_matching(self) -> None:
        """Terms match as substrings — 'HBM' matches in 'HBM3e'."""
        counts = _count("HBM3e production ramping", ["HBM"])
        assert counts == {"HBM": 1}


# -- Section weight tests -------------------------------------------------


class TestSectionWeight:
    """Section importance weighting."""

    def test_risk_factors(self) -> None:
        assert _get_section_weight("Risk Factors") == 0.9

    def test_mda(self) -> None:
        assert _get_section_weight(
            "Item 7. Management's Discussion and Analysis"
        ) == 1.0

    def test_business(self) -> None:
        assert _get_section_weight("Item 1. Business") == 0.8

    def test_unknown_section(self) -> None:
        assert _get_section_weight("Exhibits") == DEFAULT_SECTION_WEIGHT

    def test_case_insensitive(self) -> None:
        assert _get_section_weight("RISK FACTORS") == 0.9

    def test_partial_match(self) -> None:
        """Substring matching picks up sections with extra context."""
        w = _get_section_weight("Management's Discussion and Analysis of Financial Condition")
        assert w == 1.0


# -- Section signal tests --------------------------------------------------


class TestSectionSignals:
    """Per-section keyword scoring."""

    def test_basic_signal(self) -> None:
        section = _make_section(content="HBM memory HBM demand HBM supply", word_count=6)
        signals = _compute_section_signals([section], ["HBM"])
        assert len(signals) == 1
        assert signals[0].mention_count == 3
        assert signals[0].matched_terms == ["HBM"]
        assert signals[0].density == round(3 / 6 * 1000, 2)

    def test_no_keywords(self) -> None:
        signals = _compute_section_signals([_make_section()], [])
        assert signals == []

    def test_no_match_excluded(self) -> None:
        section = _make_section(content="nothing about that topic here")
        signals = _compute_section_signals([section], ["HBM"])
        assert signals == []

    def test_zero_word_count_excluded(self) -> None:
        section = _make_section(content="HBM", word_count=0)
        signals = _compute_section_signals([section], ["HBM"])
        assert signals == []

    def test_strength_incorporates_weight(self) -> None:
        """MD&A (weight 1.0) produces higher strength than Properties (0.3)."""
        mda = _make_section(
            section_id="mda",
            section_name="Management's Discussion and Analysis",
            content="HBM " * 50,
            word_count=50,
        )
        props = _make_section(
            section_id="props",
            section_name="Properties",
            content="HBM " * 50,
            word_count=50,
        )
        [mda_sig] = _compute_section_signals([mda], ["HBM"])
        [prop_sig] = _compute_section_signals([props], ["HBM"])
        assert mda_sig.strength > prop_sig.strength

    def test_density_saturation(self) -> None:
        """Density factor caps at 1.0 to prevent outlier sections."""
        # 100 mentions in 100 words = 1000 per 1k words, well above saturation
        section = _make_section(content="HBM " * 100, word_count=100)
        [signal] = _compute_section_signals([section], ["HBM"])
        assert signal.density == round(100 / 100 * 1000, 2)
        # strength = min(1.0, density/SATURATION) * weight
        # density/SATURATION = 1000/20 = 50 → capped at 1.0
        assert signal.strength <= 1.0

    def test_multiple_keywords(self) -> None:
        section = _make_section(
            content="HBM memory in GPU accelerators for AI workloads"
        )
        signals = _compute_section_signals([section], ["HBM", "GPU", "AI"])
        assert len(signals) == 1
        assert set(signals[0].matched_terms) == {"AI", "GPU", "HBM"}
        assert signals[0].mention_count == 3


# -- Fact signal tests -----------------------------------------------------


class TestFactSignals:
    """XBRL fact matching against theme concepts."""

    def test_basic_match(self) -> None:
        facts = [_make_fact("ResearchAndDevelopmentExpense", "1.5B")]
        signals = _compute_fact_signals(
            facts, ["ResearchAndDevelopmentExpense"]
        )
        assert len(signals) == 1
        assert signals[0].matched_theme_concept == "ResearchAndDevelopmentExpense"

    def test_case_insensitive(self) -> None:
        facts = [_make_fact("researchanddevelopmentexpense", "1.5B")]
        signals = _compute_fact_signals(
            facts, ["ResearchAndDevelopmentExpense"]
        )
        assert len(signals) == 1

    def test_no_match(self) -> None:
        facts = [_make_fact("Revenue", "10B")]
        signals = _compute_fact_signals(facts, ["CapitalExpenditures"])
        assert signals == []

    def test_no_concepts(self) -> None:
        signals = _compute_fact_signals([_make_fact()], [])
        assert signals == []

    def test_multiple_matches(self) -> None:
        facts = [
            _make_fact("ResearchAndDevelopmentExpense", "1.5B"),
            _make_fact("CapitalExpenditures", "3B"),
            _make_fact("Revenue", "20B"),
        ]
        signals = _compute_fact_signals(
            facts,
            ["ResearchAndDevelopmentExpense", "CapitalExpenditures"],
        )
        assert len(signals) == 2

    def test_period_preserved(self) -> None:
        facts = [_make_fact("RnD", "1B", period_label="2025-Q4")]
        signals = _compute_fact_signals(facts, ["RnD"])
        assert signals[0].period_label == "2025-Q4"


# -- Section coverage tests ------------------------------------------------


class TestSectionCoverage:
    """Weighted coverage of sections showing theme adoption."""

    def test_all_sections_match(self) -> None:
        sections = [_make_section(section_id="s1"), _make_section(section_id="s2")]
        signals = [
            SectionSignal(section_id="s1", section_name="Risk Factors"),
            SectionSignal(section_id="s2", section_name="Risk Factors"),
        ]
        coverage = _compute_section_coverage(sections, signals)
        assert coverage == 1.0

    def test_no_sections(self) -> None:
        assert _compute_section_coverage([], []) == 0.0

    def test_partial_coverage(self) -> None:
        s1 = _make_section(section_id="s1", section_name="Business")
        s2 = _make_section(section_id="s2", section_name="Properties")
        signals = [SectionSignal(section_id="s1", section_name="Business")]
        coverage = _compute_section_coverage([s1, s2], signals)
        # Business weight 0.8, Properties weight 0.3 → 0.8 / 1.1
        expected = 0.8 / (0.8 + 0.3)
        assert abs(coverage - expected) < 0.001

    def test_weighted_favors_important_sections(self) -> None:
        """Matching MD&A alone gives higher coverage than matching Properties alone."""
        mda = _make_section(section_id="mda", section_name="Management's Discussion and Analysis")
        props = _make_section(section_id="props", section_name="Properties")
        both = [mda, props]

        cov_mda = _compute_section_coverage(
            both, [SectionSignal(section_id="mda", section_name="MD&A")]
        )
        cov_props = _compute_section_coverage(
            both, [SectionSignal(section_id="props", section_name="Properties")]
        )
        assert cov_mda > cov_props


# -- Section depth tests ---------------------------------------------------


class TestSectionDepth:
    """Average strength across matching sections."""

    def test_single_signal(self) -> None:
        depth = _compute_section_depth([
            SectionSignal(section_id="s1", section_name="RF", strength=0.6)
        ])
        assert depth == 0.6

    def test_no_signals(self) -> None:
        assert _compute_section_depth([]) == 0.0

    def test_caps_at_one(self) -> None:
        """If all signals have strength 1.0, depth is exactly 1.0."""
        depth = _compute_section_depth([
            SectionSignal(section_id="s1", section_name="RF", strength=1.0),
            SectionSignal(section_id="s2", section_name="B", strength=1.0),
        ])
        assert depth == 1.0

    def test_average(self) -> None:
        depth = _compute_section_depth([
            SectionSignal(section_id="s1", section_name="RF", strength=0.4),
            SectionSignal(section_id="s2", section_name="B", strength=0.2),
        ])
        assert abs(depth - 0.3) < 0.001


# -- Temporal consistency tests --------------------------------------------


class TestTemporalConsistency:
    """Period-level adoption consistency."""

    def test_all_periods_signal(self) -> None:
        sections = [
            _make_section(section_id="s1", filing_period="2025-Q1"),
            _make_section(section_id="s2", filing_period="2025-Q2"),
        ]
        signals = [
            SectionSignal(section_id="s1", section_name="RF"),
            SectionSignal(section_id="s2", section_name="RF"),
        ]
        periods_with, total = _compute_temporal_consistency(sections, signals)
        assert periods_with == 2
        assert total == 2

    def test_partial_period_coverage(self) -> None:
        sections = [
            _make_section(section_id="s1", filing_period="2025-Q1"),
            _make_section(section_id="s2", filing_period="2025-Q2"),
            _make_section(section_id="s3", filing_period="2025-Q3"),
        ]
        signals = [SectionSignal(section_id="s1", section_name="RF")]
        periods_with, total = _compute_temporal_consistency(sections, signals)
        assert periods_with == 1
        assert total == 3

    def test_no_periods(self) -> None:
        """Sections without filing_period don't count."""
        sections = [_make_section(section_id="s1", filing_period="")]
        periods_with, total = _compute_temporal_consistency(sections, [])
        assert total == 0

    def test_same_period_multiple_sections(self) -> None:
        """Multiple sections in one period count as one period."""
        sections = [
            _make_section(section_id="s1", filing_period="2025-10K"),
            _make_section(section_id="s2", filing_period="2025-10K"),
        ]
        signals = [SectionSignal(section_id="s1", section_name="RF")]
        periods_with, total = _compute_temporal_consistency(sections, signals)
        assert periods_with == 1
        assert total == 1


# -- compute_filing_adoption integration tests -----------------------------


class TestComputeFilingAdoption:
    """Full adoption score computation pipeline."""

    def test_basic_adoption(self) -> None:
        sections = [
            _make_section(
                section_id="s1",
                section_name="Risk Factors",
                content="HBM memory demand is accelerating in data centers",
                filing_period="2025-10K",
            ),
        ]
        facts = [_make_fact("ResearchAndDevelopmentExpense", "1.5B")]
        result = compute_filing_adoption(
            ISSUER, THEME, sections, facts,
            keywords=["HBM"],
            xbrl_concepts=["ResearchAndDevelopmentExpense"],
            filing_count=1,
            now=NOW,
        )
        assert result.issuer_concept_id == ISSUER
        assert result.theme_concept_id == THEME
        assert result.score > 0
        assert result.filing_count == 1
        assert len(result.section_signals) == 1
        assert len(result.fact_signals) == 1

    def test_empty_inputs(self) -> None:
        result = compute_filing_adoption(
            ISSUER, THEME, [], [],
            keywords=[], xbrl_concepts=[], now=NOW,
        )
        assert result.score == 0.0
        assert result.breakdown.section_coverage == 0.0
        assert result.breakdown.fact_alignment == 0.0
        assert result.section_signals == []
        assert result.fact_signals == []

    def test_no_keywords_no_concepts(self) -> None:
        """Sections and facts present, but no keywords to match."""
        sections = [_make_section()]
        facts = [_make_fact()]
        result = compute_filing_adoption(
            ISSUER, THEME, sections, facts,
            keywords=[], xbrl_concepts=[], now=NOW,
        )
        assert result.score == 0.0

    def test_section_only_adoption(self) -> None:
        """Adoption from section signals alone, no XBRL."""
        sections = [
            _make_section(
                section_id="s1",
                section_name="Management's Discussion and Analysis",
                content="HBM " * 30,
                word_count=30,
                filing_period="2025-10K",
            ),
        ]
        result = compute_filing_adoption(
            ISSUER, THEME, sections, [],
            keywords=["HBM"], xbrl_concepts=[], now=NOW,
        )
        assert result.score > 0
        assert result.breakdown.section_coverage > 0
        assert result.breakdown.section_depth > 0
        assert result.breakdown.fact_alignment == 0.0
        assert result.breakdown.temporal_consistency > 0

    def test_fact_only_adoption(self) -> None:
        """Adoption from XBRL facts alone, no section keywords."""
        facts = [
            _make_fact("ResearchAndDevelopmentExpense", "1.5B"),
            _make_fact("CapitalExpenditures", "3B"),
        ]
        result = compute_filing_adoption(
            ISSUER, THEME, [], facts,
            keywords=[],
            xbrl_concepts=["ResearchAndDevelopmentExpense", "CapitalExpenditures"],
            now=NOW,
        )
        assert result.score > 0
        assert result.breakdown.section_coverage == 0.0
        assert result.breakdown.fact_alignment == 1.0

    def test_multi_period_temporal_consistency(self) -> None:
        """Three periods, two with HBM mentions."""
        sections = [
            _make_section(section_id="s1", content="HBM demand", filing_period="2025-Q1"),
            _make_section(section_id="s2", content="HBM supply", filing_period="2025-Q2"),
            _make_section(section_id="s3", content="No topic match", filing_period="2025-Q3"),
        ]
        result = compute_filing_adoption(
            ISSUER, THEME, sections, [],
            keywords=["HBM"], xbrl_concepts=[], now=NOW,
        )
        assert result.periods_with_signal == 2
        assert result.period_count == 3
        assert abs(result.breakdown.temporal_consistency - 2 / 3) < 0.01

    def test_composite_weights_sum_to_one(self) -> None:
        total = (
            WEIGHT_SECTION_COVERAGE
            + WEIGHT_SECTION_DEPTH
            + WEIGHT_FACT_ALIGNMENT
            + WEIGHT_TEMPORAL_CONSISTENCY
        )
        assert abs(total - 1.0) < 1e-10

    def test_perfect_score(self) -> None:
        """All components at 1.0 should yield score of 1.0."""
        # Dense HBM mentions in MD&A + XBRL match + single period
        sections = [
            _make_section(
                section_id="s1",
                section_name="Management's Discussion and Analysis",
                content="HBM " * 100,
                word_count=100,
                filing_period="2025-10K",
            ),
        ]
        facts = [_make_fact("RnDExpense", "1B")]
        result = compute_filing_adoption(
            ISSUER, THEME, sections, facts,
            keywords=["HBM"],
            xbrl_concepts=["RnDExpense"],
            filing_count=1,
            now=NOW,
        )
        assert result.breakdown.section_coverage == 1.0
        assert result.breakdown.fact_alignment == 1.0
        assert result.breakdown.temporal_consistency == 1.0
        assert result.score == 1.0

    def test_score_bounded_zero_to_one(self) -> None:
        sections = [_make_section(content="HBM " * 200, word_count=200)]
        result = compute_filing_adoption(
            ISSUER, THEME, sections, [],
            keywords=["HBM"], xbrl_concepts=[], now=NOW,
        )
        assert 0.0 <= result.score <= 1.0

    def test_to_dict(self) -> None:
        result = compute_filing_adoption(
            ISSUER, THEME,
            [_make_section(content="HBM test")],
            [_make_fact("RnD", "1B")],
            keywords=["HBM"],
            xbrl_concepts=["RnD"],
            now=NOW,
        )
        d = result.to_dict()
        assert d["issuer_concept_id"] == ISSUER
        assert d["theme_concept_id"] == THEME
        assert "breakdown" in d
        assert "section_coverage" in d["breakdown"]
        assert isinstance(d["computed_at"], str)

    def test_higher_adoption_for_relevant_filings(self) -> None:
        """Issuer with HBM-heavy filings scores higher than unrelated."""
        relevant = [
            _make_section(
                section_id="r1",
                section_name="Risk Factors",
                content="HBM memory HBM demand HBM shortage HBM technology",
                word_count=8,
            ),
            _make_section(
                section_id="r2",
                section_name="Business",
                content="HBM packaging and HBM capacity expansion",
                word_count=7,
            ),
        ]
        irrelevant = [
            _make_section(
                section_id="i1",
                section_name="Risk Factors",
                content="General market conditions and foreign exchange risk",
                word_count=8,
            ),
            _make_section(
                section_id="i2",
                section_name="Business",
                content="We manufacture consumer electronics and appliances",
                word_count=7,
            ),
        ]
        score_relevant = compute_filing_adoption(
            ISSUER, THEME, relevant, [],
            keywords=["HBM"], xbrl_concepts=[], now=NOW,
        ).score
        score_irrelevant = compute_filing_adoption(
            ISSUER, THEME, irrelevant, [],
            keywords=["HBM"], xbrl_concepts=[], now=NOW,
        ).score
        assert score_relevant > score_irrelevant
        assert score_irrelevant == 0.0

    def test_computed_at_uses_now(self) -> None:
        result = compute_filing_adoption(
            ISSUER, THEME, [], [],
            keywords=[], xbrl_concepts=[], now=NOW,
        )
        assert result.computed_at == NOW

    def test_filing_count_passthrough(self) -> None:
        result = compute_filing_adoption(
            ISSUER, THEME, [], [],
            keywords=[], xbrl_concepts=[],
            filing_count=5, now=NOW,
        )
        assert result.filing_count == 5


# -- Dataclass tests -------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass construction and serialization."""

    def test_section_input_frozen(self) -> None:
        s = _make_section()
        try:
            s.content = "modified"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_fact_input_frozen(self) -> None:
        f = _make_fact()
        try:
            f.value = "0"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_adoption_score_frozen(self) -> None:
        result = compute_filing_adoption(
            ISSUER, THEME, [], [],
            keywords=[], xbrl_concepts=[], now=NOW,
        )
        try:
            result.score = 0.5  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_breakdown_frozen(self) -> None:
        b = AdoptionBreakdown(
            section_coverage=0.5,
            section_depth=0.3,
            fact_alignment=0.2,
            temporal_consistency=0.8,
        )
        try:
            b.section_coverage = 1.0  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass
