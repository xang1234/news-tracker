"""Tests for peer-normalized drift decomposition.

Verifies that filing changes are decomposed by business dimension,
peer-normalized, and flagged when unusual.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.filing.alignment import SectionAlignment, SectionDiff, normalize_section_name
from src.filing.drift import (
    DRIFT_DIMENSIONS,
    Z_SCORE_CAP,
    DimensionDrift,
    SectionChange,
    _peer_stats,
    _section_to_dimension,
    _z_score,
    classify_by_dimension,
    compute_dimension_magnitude,
    compute_drift_decomposition,
    extract_section_changes,
)
from src.filing.persistence import FilingSectionRecord

NOW = datetime(2026, 4, 1, tzinfo=UTC)
ISSUER = "concept_issuer_abc123"


# -- Helpers ---------------------------------------------------------------


def _section_record(
    name: str,
    content: str = "some content here",
    index: int = 0,
    accession: str = "acc-001",
) -> FilingSectionRecord:
    return FilingSectionRecord(
        section_id=f"sec_{index}_{name[:8]}",
        accession_number=accession,
        section_index=index,
        section_name=name,
        content=content,
        word_count=len(content.split()) if content else 0,
        content_hash=f"hash_{name[:8]}",
    )


def _make_change(
    section_name: str = "risk factors",
    change_magnitude: float = 0.3,
    word_count_delta: int = 50,
    change_type: str = "modified",
    diff_ratio: float = 0.7,
) -> SectionChange:
    return SectionChange(
        section_name=section_name,
        change_magnitude=change_magnitude,
        word_count_delta=word_count_delta,
        change_type=change_type,
        diff_ratio=diff_ratio,
    )


def _make_matched_alignment(
    name: str,
    content_base: str = "old content",
    content_target: str = "new content changed",
) -> tuple[SectionAlignment, SectionDiff]:
    """Create a matched alignment + diff pair."""
    base = _section_record(name, content_base, accession="acc-base")
    target = _section_record(name, content_target, accession="acc-target")
    alignment = SectionAlignment(
        base_section=base,
        target_section=target,
        similarity=1.0,
        normalized_name=normalize_section_name(name),
    )
    # Compute a realistic diff_ratio
    import difflib

    ratio = difflib.SequenceMatcher(
        None,
        content_base.splitlines(),
        content_target.splitlines(),
    ).ratio()
    diff = SectionDiff(
        alignment=alignment,
        content_changed=(content_base != content_target),
        word_count_delta=len(content_target.split()) - len(content_base.split()),
        hash_changed=(content_base != content_target),
        diff_ratio=ratio,
    )
    return alignment, diff


# -- Section-to-dimension mapping tests ------------------------------------


class TestSectionToDimension:
    """Normalized section name → dimension classification."""

    def test_risk_factors(self) -> None:
        assert _section_to_dimension("risk factors") == "risk"

    def test_business(self) -> None:
        assert _section_to_dimension("business") == "strategy"

    def test_mda(self) -> None:
        assert _section_to_dimension("management's discussion and analysis") == "capex"

    def test_legal_proceedings(self) -> None:
        assert _section_to_dimension("legal proceedings") == "regulatory"

    def test_market_risk(self) -> None:
        assert (
            _section_to_dimension("quantitative and qualitative disclosures about market risk")
            == "regulatory"
        )

    def test_customers(self) -> None:
        assert _section_to_dimension("major customers") == "customer_supplier"

    def test_supply_chain(self) -> None:
        assert _section_to_dimension("supply chain") == "customer_supplier"

    def test_unknown_section(self) -> None:
        assert _section_to_dimension("exhibits") is None

    def test_substring_match(self) -> None:
        """Longer section name containing a known key still matches."""
        dim = _section_to_dimension("management's discussion and analysis of financial condition")
        assert dim == "capex"

    def test_all_dimensions_have_at_least_one_section(self) -> None:
        from src.filing.drift import DIMENSION_SECTIONS

        for dim in DRIFT_DIMENSIONS:
            assert len(DIMENSION_SECTIONS[dim]) > 0


# -- Extract section changes tests ----------------------------------------


class TestExtractSectionChanges:
    """Convert alignment/diff outputs to SectionChange objects."""

    def test_modified_section(self) -> None:
        alignment, diff = _make_matched_alignment(
            "Risk Factors",
            "Risk of supply disruption.",
            "Risk of supply disruption and demand volatility.",
        )
        changes = extract_section_changes([diff], [alignment])
        assert len(changes) == 1
        assert changes[0].change_type == "modified"
        assert changes[0].change_magnitude > 0

    def test_unchanged_section(self) -> None:
        alignment, diff = _make_matched_alignment(
            "Risk Factors",
            "Same content",
            "Same content",
        )
        changes = extract_section_changes([diff], [alignment])
        assert len(changes) == 1
        assert changes[0].change_type == "unchanged"
        assert changes[0].change_magnitude == 0.0
        assert changes[0].diff_ratio == 1.0

    def test_added_section(self) -> None:
        target = _section_record("Supply Chain", "New supply chain section", accession="acc-target")
        alignment = SectionAlignment(
            base_section=None,
            target_section=target,
            similarity=0.0,
            normalized_name=normalize_section_name("Supply Chain"),
        )
        changes = extract_section_changes([], [alignment])
        assert len(changes) == 1
        assert changes[0].change_type == "added"
        assert changes[0].change_magnitude == 1.0
        assert changes[0].word_count_delta == target.word_count

    def test_removed_section(self) -> None:
        base = _section_record("Legal Proceedings", "Old legal content", accession="acc-base")
        alignment = SectionAlignment(
            base_section=base,
            target_section=None,
            similarity=0.0,
            normalized_name=normalize_section_name("Legal Proceedings"),
        )
        changes = extract_section_changes([], [alignment])
        assert len(changes) == 1
        assert changes[0].change_type == "removed"
        assert changes[0].change_magnitude == 1.0
        assert changes[0].word_count_delta == -base.word_count

    def test_mixed_changes(self) -> None:
        """Modified + added + removed all handled in one call."""
        a1, d1 = _make_matched_alignment("Risk Factors", "old", "new")
        added_target = _section_record("Customers", "New section")
        a_added = SectionAlignment(
            base_section=None,
            target_section=added_target,
            normalized_name="customers",
        )
        removed_base = _section_record("Legal Proceedings", "Old legal")
        a_removed = SectionAlignment(
            base_section=removed_base,
            target_section=None,
            normalized_name="legal proceedings",
        )
        changes = extract_section_changes([d1], [a1, a_added, a_removed])
        types = {c.change_type for c in changes}
        assert types == {"modified", "added", "removed"}


# -- Classify by dimension tests -------------------------------------------


class TestClassifyByDimension:
    """Group section changes into drift dimensions."""

    def test_single_dimension(self) -> None:
        changes = [_make_change(section_name="risk factors")]
        classified = classify_by_dimension(changes)
        assert len(classified["risk"]) == 1
        assert classified["strategy"] == []

    def test_multiple_dimensions(self) -> None:
        changes = [
            _make_change(section_name="risk factors"),
            _make_change(section_name="business"),
            _make_change(section_name="legal proceedings"),
        ]
        classified = classify_by_dimension(changes)
        assert len(classified["risk"]) == 1
        assert len(classified["strategy"]) == 1
        assert len(classified["regulatory"]) == 1

    def test_unclassified_excluded(self) -> None:
        changes = [_make_change(section_name="exhibits")]
        classified = classify_by_dimension(changes)
        for dim in DRIFT_DIMENSIONS:
            assert classified[dim] == []

    def test_all_dimensions_present(self) -> None:
        """Result always has all 5 dimensions, even if empty."""
        classified = classify_by_dimension([])
        assert set(classified.keys()) == set(DRIFT_DIMENSIONS)

    def test_multiple_sections_same_dimension(self) -> None:
        changes = [
            _make_change(section_name="legal proceedings"),
            _make_change(section_name="regulatory matters"),
        ]
        classified = classify_by_dimension(changes)
        assert len(classified["regulatory"]) == 2


# -- Dimension magnitude tests --------------------------------------------


class TestDimensionMagnitude:
    """Per-dimension change magnitude computation."""

    def test_single_change(self) -> None:
        changes = [_make_change(change_magnitude=0.4)]
        assert compute_dimension_magnitude(changes) == 0.4

    def test_average_of_multiple(self) -> None:
        changes = [
            _make_change(change_magnitude=0.2),
            _make_change(change_magnitude=0.6),
        ]
        assert abs(compute_dimension_magnitude(changes) - 0.4) < 1e-10

    def test_empty(self) -> None:
        assert compute_dimension_magnitude([]) == 0.0

    def test_all_unchanged(self) -> None:
        changes = [
            _make_change(change_magnitude=0.0),
            _make_change(change_magnitude=0.0),
        ]
        assert compute_dimension_magnitude(changes) == 0.0

    def test_all_rewrites(self) -> None:
        changes = [
            _make_change(change_magnitude=1.0),
            _make_change(change_magnitude=1.0),
        ]
        assert compute_dimension_magnitude(changes) == 1.0


# -- Peer stats tests ------------------------------------------------------


class TestPeerStats:
    """Mean and standard deviation of peer magnitudes."""

    def test_basic(self) -> None:
        mean, std, has = _peer_stats([0.2, 0.4, 0.6])
        assert has is True
        assert abs(mean - 0.4) < 0.001
        assert std > 0

    def test_single_peer(self) -> None:
        mean, std, has = _peer_stats([0.3])
        assert has is True
        assert mean == 0.3
        assert std == 0.0

    def test_no_peers(self) -> None:
        mean, std, has = _peer_stats([])
        assert has is False
        assert mean == 0.0
        assert std == 0.0

    def test_identical_values(self) -> None:
        mean, std, has = _peer_stats([0.5, 0.5, 0.5])
        assert has is True
        assert mean == 0.5
        assert std == 0.0


# -- Z-score tests ---------------------------------------------------------


class TestZScore:
    """Capped z-score computation."""

    def test_basic(self) -> None:
        z = _z_score(0.6, 0.3, 0.1)
        assert abs(z - 3.0) < 0.01  # (0.6-0.3)/0.1 = 3, capped at 3

    def test_zero_std_same_value(self) -> None:
        z = _z_score(0.5, 0.5, 0.0)
        assert z == 0.0

    def test_zero_std_different_value_positive(self) -> None:
        z = _z_score(0.8, 0.5, 0.0)
        assert z == Z_SCORE_CAP

    def test_zero_std_different_value_negative(self) -> None:
        z = _z_score(0.2, 0.5, 0.0)
        assert z == -Z_SCORE_CAP

    def test_capped_positive(self) -> None:
        z = _z_score(10.0, 0.0, 1.0)
        assert z == Z_SCORE_CAP

    def test_capped_negative(self) -> None:
        z = _z_score(-10.0, 0.0, 1.0)
        assert z == -Z_SCORE_CAP

    def test_moderate_z(self) -> None:
        z = _z_score(0.4, 0.2, 0.1)
        assert abs(z - 2.0) < 0.01


# -- Full decomposition integration tests ---------------------------------


class TestComputeDriftDecomposition:
    """End-to-end drift decomposition pipeline."""

    def test_basic_decomposition(self) -> None:
        issuer_changes = [
            _make_change(section_name="risk factors", change_magnitude=0.4),
            _make_change(section_name="business", change_magnitude=0.2),
        ]
        peer1 = [
            _make_change(section_name="risk factors", change_magnitude=0.1),
            _make_change(section_name="business", change_magnitude=0.1),
        ]
        peer2 = [
            _make_change(section_name="risk factors", change_magnitude=0.15),
            _make_change(section_name="business", change_magnitude=0.12),
        ]
        result = compute_drift_decomposition(
            ISSUER,
            issuer_changes,
            [peer1, peer2],
            base_accession="acc-base",
            target_accession="acc-target",
            now=NOW,
        )
        assert result.issuer_concept_id == ISSUER
        assert result.base_accession == "acc-base"
        assert len(result.dimensions) == 5
        # Risk dimension should have magnitude 0.4
        risk = next(d for d in result.dimensions if d.dimension == "risk")
        assert risk.magnitude == 0.4
        assert risk.peer_mean > 0
        assert risk.z_score > 0  # Issuer changed more than peers

    def test_all_five_dimensions_present(self) -> None:
        result = compute_drift_decomposition(
            ISSUER,
            [],
            [],
            now=NOW,
        )
        dims = {d.dimension for d in result.dimensions}
        assert dims == set(DRIFT_DIMENSIONS)

    def test_no_changes_no_drift(self) -> None:
        result = compute_drift_decomposition(
            ISSUER,
            [],
            [],
            now=NOW,
        )
        for dim in result.dimensions:
            assert dim.magnitude == 0.0
            assert dim.z_score == 0.0
            assert not dim.is_unusual

    def test_unusual_flagging(self) -> None:
        """Issuer with high magnitude in risk, peers all low."""
        issuer = [_make_change(section_name="risk factors", change_magnitude=0.8)]
        peers = [
            [_make_change(section_name="risk factors", change_magnitude=0.1)],
            [_make_change(section_name="risk factors", change_magnitude=0.12)],
            [_make_change(section_name="risk factors", change_magnitude=0.08)],
            [_make_change(section_name="risk factors", change_magnitude=0.11)],
        ]
        result = compute_drift_decomposition(
            ISSUER,
            issuer,
            peers,
            now=NOW,
        )
        risk = next(d for d in result.dimensions if d.dimension == "risk")
        assert risk.is_unusual
        assert "risk" in result.unusual_dimensions

    def test_no_peers_no_normalization(self) -> None:
        """Without peers, z_score is 0 and nothing is unusual."""
        issuer = [_make_change(section_name="risk factors", change_magnitude=0.8)]
        result = compute_drift_decomposition(
            ISSUER,
            issuer,
            [],
            now=NOW,
        )
        risk = next(d for d in result.dimensions if d.dimension == "risk")
        assert risk.magnitude == 0.8
        assert risk.z_score == 0.0
        assert not risk.is_unusual

    def test_single_peer(self) -> None:
        """Single peer: std=0, so z-score is capped or zero."""
        issuer = [_make_change(section_name="risk factors", change_magnitude=0.5)]
        peer = [_make_change(section_name="risk factors", change_magnitude=0.1)]
        result = compute_drift_decomposition(
            ISSUER,
            issuer,
            [peer],
            now=NOW,
        )
        risk = next(d for d in result.dimensions if d.dimension == "risk")
        assert risk.z_score == Z_SCORE_CAP  # Different from peer, std=0

    def test_word_count_delta_aggregated(self) -> None:
        issuer = [
            _make_change(section_name="legal proceedings", word_count_delta=100),
            _make_change(section_name="regulatory matters", word_count_delta=-30),
        ]
        result = compute_drift_decomposition(
            ISSUER,
            issuer,
            [],
            now=NOW,
        )
        reg = next(d for d in result.dimensions if d.dimension == "regulatory")
        assert reg.word_count_delta == 70

    def test_section_names_tracked(self) -> None:
        issuer = [
            _make_change(section_name="legal proceedings"),
            _make_change(section_name="regulatory matters"),
        ]
        result = compute_drift_decomposition(
            ISSUER,
            issuer,
            [],
            now=NOW,
        )
        reg = next(d for d in result.dimensions if d.dimension == "regulatory")
        assert "legal proceedings" in reg.section_names
        assert "regulatory matters" in reg.section_names
        assert len(reg.section_names) == 2

    def test_custom_unusual_threshold(self) -> None:
        """Higher threshold makes it harder to flag as unusual."""
        issuer = [_make_change(section_name="risk factors", change_magnitude=0.5)]
        peers = [
            [_make_change(section_name="risk factors", change_magnitude=0.1)],
            [_make_change(section_name="risk factors", change_magnitude=0.15)],
            [_make_change(section_name="risk factors", change_magnitude=0.12)],
        ]
        result_strict = compute_drift_decomposition(
            ISSUER,
            issuer,
            peers,
            unusual_threshold=10.0,  # Very strict
            now=NOW,
        )
        assert result_strict.unusual_dimensions == []

    def test_to_dict(self) -> None:
        issuer = [_make_change(section_name="risk factors", change_magnitude=0.3)]
        result = compute_drift_decomposition(
            ISSUER,
            issuer,
            [],
            base_accession="acc-001",
            target_accession="acc-002",
            now=NOW,
        )
        d = result.to_dict()
        assert d["issuer_concept_id"] == ISSUER
        assert d["base_accession"] == "acc-001"
        assert len(d["dimensions"]) == 5
        assert isinstance(d["computed_at"], str)
        risk_d = next(x for x in d["dimensions"] if x["dimension"] == "risk")
        assert "magnitude" in risk_d
        assert "z_score" in risk_d

    def test_computed_at_uses_now(self) -> None:
        result = compute_drift_decomposition(
            ISSUER,
            [],
            [],
            now=NOW,
        )
        assert result.computed_at == NOW

    def test_peer_with_different_sections(self) -> None:
        """Peers may have sections in different dimensions than issuer."""
        issuer = [_make_change(section_name="risk factors", change_magnitude=0.5)]
        # Peer only changed business section, not risk
        peer = [_make_change(section_name="business", change_magnitude=0.3)]
        result = compute_drift_decomposition(
            ISSUER,
            issuer,
            [peer],
            now=NOW,
        )
        risk = next(d for d in result.dimensions if d.dimension == "risk")
        # Peer has 0 risk magnitude, issuer has 0.5
        assert risk.magnitude == 0.5
        assert risk.peer_mean == 0.0  # Peer had no risk changes


# -- Dataclass tests -------------------------------------------------------


class TestDataclasses:
    """Frozen dataclass invariants."""

    def test_section_change_frozen(self) -> None:
        c = _make_change()
        try:
            c.change_magnitude = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_dimension_drift_frozen(self) -> None:
        d = DimensionDrift(
            dimension="risk",
            magnitude=0.3,
            word_count_delta=10,
            peer_mean=0.1,
            peer_std=0.05,
            z_score=2.0,
            is_unusual=True,
            section_names=["risk factors"],
        )
        try:
            d.magnitude = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_decomposition_frozen(self) -> None:
        result = compute_drift_decomposition(ISSUER, [], [], now=NOW)
        try:
            result.issuer_concept_id = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass
