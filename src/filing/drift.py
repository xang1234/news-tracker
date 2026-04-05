"""Peer-normalized drift decomposition by filing section dimension.

Decomposes filing-over-filing changes into five business-meaning
dimensions (strategy, risk, capex, customer-supplier, regulatory)
and normalizes against a peer group to identify unusual changes.

This is NOT a single distance metric — the decomposition IS the
product. Downstream alerting and explanation work consumes
individual dimension scores, not a rolled-up scalar.

Drift pipeline:
    1. Extract section-level changes from filing comparisons
    2. Classify each change into a business dimension
    3. Compute per-dimension magnitude for the issuer
    4. Normalize against peer magnitudes
    5. Return decomposed result with z-scores

All functions are stateless — the caller fetches filing comparisons
and peer context, then passes them here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.filing.alignment import SectionAlignment, SectionDiff


# -- Drift dimensions --------------------------------------------------------

DRIFT_DIMENSIONS: list[str] = [
    "strategy",
    "risk",
    "capex",
    "customer_supplier",
    "regulatory",
]

# Normalized section names → drift dimension.
# Each section maps to at most one dimension. Unmapped sections are
# excluded from decomposition but still available in raw comparison.
DIMENSION_SECTIONS: dict[str, frozenset[str]] = {
    "strategy": frozenset({
        "business",
        "overview",
        "corporate strategy",
    }),
    "risk": frozenset({
        "risk factors",
    }),
    "capex": frozenset({
        "management's discussion and analysis",
        "capital expenditures",
        "liquidity and capital resources",
    }),
    "customer_supplier": frozenset({
        "customers",
        "major customers",
        "suppliers",
        "supply chain",
        "customer concentration",
    }),
    "regulatory": frozenset({
        "legal proceedings",
        "quantitative and qualitative disclosures about market risk",
        "government regulation",
        "regulatory matters",
        "compliance",
    }),
}

# Z-score caps and thresholds
Z_SCORE_CAP = 3.0
UNUSUAL_THRESHOLD = 1.5

# Reverse lookup: normalized section name → dimension (O(1) exact match)
_SECTION_DIM_EXACT: dict[str, str] = {}
for _dim, _names in DIMENSION_SECTIONS.items():
    for _name in _names:
        _SECTION_DIM_EXACT[_name] = _dim


# -- Section change (intermediate representation) ----------------------------


@dataclass(frozen=True)
class SectionChange:
    """A section's change between two consecutive filings.

    Lightweight intermediate that decouples drift scoring from
    the full FilingComparison/SectionDiff infrastructure.

    Attributes:
        section_name: Normalized section name.
        change_magnitude: 0-1 (0=no change, 1=complete rewrite/add/remove).
        word_count_delta: Change in word count (target - base).
        change_type: "modified", "added", "removed", "unchanged".
        diff_ratio: Raw SequenceMatcher ratio (1=identical, 0=completely different).
    """

    section_name: str
    change_magnitude: float
    word_count_delta: int
    change_type: str
    diff_ratio: float = 1.0


# -- Per-dimension drift result -----------------------------------------------


@dataclass(frozen=True)
class DimensionDrift:
    """Drift measurement for a single filing dimension.

    Attributes:
        dimension: Which business dimension (strategy, risk, etc.).
        magnitude: 0-1, how much this dimension changed for the issuer.
        word_count_delta: Net word change across dimension sections.
        peer_mean: Average magnitude across peers (0 if no peers).
        peer_std: Std dev of peer magnitudes (0 if no peers).
        z_score: Peer-normalized deviation (0 if no peers).
        is_unusual: Whether |z_score| exceeds threshold.
        section_names: Which sections contributed to this dimension.
    """

    dimension: str
    magnitude: float
    word_count_delta: int
    peer_mean: float
    peer_std: float
    z_score: float
    is_unusual: bool
    section_names: list[str] = field(default_factory=list)


# -- Complete decomposition result -------------------------------------------


@dataclass(frozen=True)
class DriftDecomposition:
    """Complete peer-normalized drift decomposition for an issuer.

    Attributes:
        issuer_concept_id: Canonical issuer concept.
        base_accession: Earlier filing.
        target_accession: Later filing.
        dimensions: Per-dimension drift results.
        unusual_dimensions: Dimensions flagged as unusual.
        computed_at: When this decomposition was computed.
    """

    issuer_concept_id: str
    base_accession: str
    target_accession: str
    dimensions: list[DimensionDrift] = field(default_factory=list)
    unusual_dimensions: list[str] = field(default_factory=list)
    computed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for publication payloads."""
        return {
            "issuer_concept_id": self.issuer_concept_id,
            "base_accession": self.base_accession,
            "target_accession": self.target_accession,
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "magnitude": round(d.magnitude, 4),
                    "word_count_delta": d.word_count_delta,
                    "peer_mean": round(d.peer_mean, 4),
                    "peer_std": round(d.peer_std, 4),
                    "z_score": round(d.z_score, 4),
                    "is_unusual": d.is_unusual,
                    "section_names": d.section_names,
                    "section_count": len(d.section_names),
                }
                for d in self.dimensions
            ],
            "unusual_dimensions": self.unusual_dimensions,
            "computed_at": self.computed_at.isoformat(),
        }


# -- Helpers (stateless) -----------------------------------------------------


def _section_to_dimension(section_name: str) -> str | None:
    """Map a normalized section name to its drift dimension.

    O(1) exact lookup first, then substring containment fallback
    for long SEC headers like "management's discussion and analysis
    of financial condition and results of operations".
    """
    exact = _SECTION_DIM_EXACT.get(section_name)
    if exact is not None:
        return exact
    for dim, names in DIMENSION_SECTIONS.items():
        for name in names:
            if name in section_name or section_name in name:
                return dim
    return None


def extract_section_changes(
    diffs: list[SectionDiff],
    alignments: list[SectionAlignment],
) -> list[SectionChange]:
    """Extract section-level changes from filing comparison outputs.

    Converts matched diffs and added/removed alignments into a
    flat list of SectionChange objects for drift scoring.

    Args:
        diffs: Section diffs from diff_aligned_sections().
        alignments: All alignments from align_sections().

    Returns:
        List of SectionChange, one per section.
    """
    changes: list[SectionChange] = []

    # From matched diffs
    for diff in diffs:
        name = diff.alignment.normalized_name
        if diff.content_changed:
            changes.append(SectionChange(
                section_name=name,
                change_magnitude=round(1.0 - diff.diff_ratio, 4),
                word_count_delta=diff.word_count_delta,
                change_type="modified",
                diff_ratio=diff.diff_ratio,
            ))
        else:
            changes.append(SectionChange(
                section_name=name,
                change_magnitude=0.0,
                word_count_delta=0,
                change_type="unchanged",
                diff_ratio=1.0,
            ))

    # From unmatched alignments (added/removed)
    for alignment in alignments:
        if alignment.is_added:
            target = alignment.target_section
            assert target is not None
            changes.append(SectionChange(
                section_name=alignment.normalized_name,
                change_magnitude=1.0,
                word_count_delta=target.word_count,
                change_type="added",
                diff_ratio=0.0,
            ))
        elif alignment.is_removed:
            base = alignment.base_section
            assert base is not None
            changes.append(SectionChange(
                section_name=alignment.normalized_name,
                change_magnitude=1.0,
                word_count_delta=-base.word_count,
                change_type="removed",
                diff_ratio=0.0,
            ))

    return changes


def classify_by_dimension(
    changes: list[SectionChange],
) -> dict[str, list[SectionChange]]:
    """Classify section changes into drift dimensions.

    Returns a dict with an entry for every dimension, even if empty.
    Changes that don't map to any dimension are excluded.
    """
    result: dict[str, list[SectionChange]] = {d: [] for d in DRIFT_DIMENSIONS}
    for change in changes:
        dim = _section_to_dimension(change.section_name)
        if dim is not None:
            result[dim].append(change)
    return result


def compute_dimension_magnitude(changes: list[SectionChange]) -> float:
    """Average change magnitude across sections in a dimension.

    Returns 0.0 if no sections — absence of data is not drift.
    """
    if not changes:
        return 0.0
    return sum(c.change_magnitude for c in changes) / len(changes)


def _z_score(value: float, mean: float, std: float) -> float:
    """Compute capped z-score, handling zero std."""
    if std < 1e-10:
        if abs(value - mean) < 1e-10:
            return 0.0
        return Z_SCORE_CAP if value > mean else -Z_SCORE_CAP
    raw = (value - mean) / std
    return max(-Z_SCORE_CAP, min(Z_SCORE_CAP, raw))


def _peer_stats(values: list[float]) -> tuple[float, float, bool]:
    """Compute mean and std for peer magnitudes.

    Returns (mean, std, has_peers). When has_peers is False,
    mean and std are 0.0 — caller should skip normalization.
    """
    if not values:
        return 0.0, 0.0, False
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, True
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance), True


# -- Main compute function ---------------------------------------------------


def compute_drift_decomposition(
    issuer_concept_id: str,
    issuer_changes: list[SectionChange],
    peer_changes_list: list[list[SectionChange]],
    *,
    base_accession: str = "",
    target_accession: str = "",
    unusual_threshold: float = UNUSUAL_THRESHOLD,
    now: datetime | None = None,
) -> DriftDecomposition:
    """Compute peer-normalized drift decomposition for an issuer.

    Classifies section changes by business dimension, computes
    per-dimension magnitude, and normalizes against the peer group.

    Args:
        issuer_concept_id: Canonical issuer concept ID.
        issuer_changes: Section changes for this issuer.
        peer_changes_list: Section changes for each peer.
        base_accession: Earlier filing accession number.
        target_accession: Later filing accession number.
        unusual_threshold: Z-score threshold for flagging unusual drift.
        now: Current time for timestamp.

    Returns:
        DriftDecomposition with per-dimension z-scores and flags.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Classify issuer changes
    issuer_by_dim = classify_by_dimension(issuer_changes)

    # Classify each peer's changes
    peer_by_dim: dict[str, list[float]] = {d: [] for d in DRIFT_DIMENSIONS}
    for peer_changes in peer_changes_list:
        peer_classified = classify_by_dimension(peer_changes)
        for dim in DRIFT_DIMENSIONS:
            peer_by_dim[dim].append(
                compute_dimension_magnitude(peer_classified[dim])
            )

    # Compute per-dimension drift
    dimensions: list[DimensionDrift] = []
    unusual: list[str] = []

    for dim in DRIFT_DIMENSIONS:
        dim_changes = issuer_by_dim[dim]
        magnitude = compute_dimension_magnitude(dim_changes)
        word_delta = sum(c.word_count_delta for c in dim_changes)
        section_names = sorted({c.section_name for c in dim_changes})

        peer_mean, peer_std, has_peers = _peer_stats(peer_by_dim[dim])
        z = _z_score(magnitude, peer_mean, peer_std) if has_peers else 0.0

        is_unusual = abs(z) >= unusual_threshold
        if is_unusual:
            unusual.append(dim)

        dimensions.append(DimensionDrift(
            dimension=dim,
            magnitude=round(magnitude, 4),
            word_count_delta=word_delta,
            peer_mean=round(peer_mean, 4),
            peer_std=round(peer_std, 4),
            z_score=round(z, 4),
            is_unusual=is_unusual,
            section_names=section_names,
        ))

    return DriftDecomposition(
        issuer_concept_id=issuer_concept_id,
        base_accession=base_accession,
        target_accession=target_accession,
        dimensions=dimensions,
        unusual_dimensions=unusual,
        computed_at=now,
    )
