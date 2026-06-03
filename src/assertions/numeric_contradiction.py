"""Numeric contradiction classification for assertion building.

Takes a set of comparable numeric claims (already narrowed to one
subject+metric+period by the caller, but defensively re-grouped here) and
decides, per claim, whether it ``support``s or ``contradiction``s the
prevailing value. The output feeds ``aggregate_assertion``, which flips an
assertion to ``disputed`` when contradiction links are present and support
falls below threshold.

Algorithm:
    1. Claims with no numeric value/metric are non-numeric — they always
       ``support`` (a relationship claim never contradicts a number).
    2. Numeric claims are grouped by ``(metric, period, unit)`` — only facts
       in the same group are comparable.
    3. Within each group an *anchor* is chosen (highest confidence, then most
       recently published, then claim_id for determinism). The anchor
       ``support``s; every other fact is compared to the anchor and
       ``support``s if it agrees within tolerance, else ``contradiction``s.

Stateless pure functions — no I/O. The caller persists the resulting links.
"""

from __future__ import annotations

from src.claims.numeric import (
    DEFAULT_REL_TOLERANCE,
    NumericClaimLike,
    numeric_link_type,
)


def _is_numeric(claim: NumericClaimLike) -> bool:
    """Whether a claim carries a comparable numeric fact."""
    return claim.numeric_value is not None and claim.metric is not None


def _anchor_sort_key(claim: NumericClaimLike) -> tuple[float, float, str]:
    """Anchor preference: highest confidence, then most recent, then id.

    Returned as a sort key (descending intent achieved via ``max``).
    """
    published_ts = (
        claim.source_published_at.timestamp() if claim.source_published_at is not None else 0.0
    )
    return (claim.confidence, published_ts, claim.claim_id)


def numeric_link_types(
    claims: list[NumericClaimLike],
    *,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
) -> dict[str, str]:
    """Classify each claim as ``support`` or ``contradiction``.

    Returns a mapping of ``claim_id`` → link type. Non-numeric claims and
    every per-group anchor map to ``support``; only facts that diverge from
    their group's anchor beyond ``rel_tolerance`` map to ``contradiction``.
    """
    result: dict[str, str] = {}

    # Group comparable numeric claims by their comparison context.
    groups: dict[tuple[str | None, str | None, str | None], list[NumericClaimLike]] = {}
    for claim in claims:
        if not _is_numeric(claim):
            result[claim.claim_id] = "support"
            continue
        key = (claim.metric, claim.period, claim.unit)
        groups.setdefault(key, []).append(claim)

    for group in groups.values():
        anchor = max(group, key=_anchor_sort_key)
        for claim in group:
            if claim.claim_id == anchor.claim_id:
                result[claim.claim_id] = "support"
                continue
            link = numeric_link_type(anchor, claim, rel_tolerance=rel_tolerance)
            # link is "support"/"contradiction"; within a group facts share
            # metric/unit/period so they are always comparable (never None).
            result[claim.claim_id] = link or "support"

    return result


def classify_numeric_links(
    assertion_id: str,
    claims: list[NumericClaimLike],
    *,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
):
    """Build ``AssertionClaimLink`` records from numeric classification.

    Each claim gets one link to ``assertion_id`` with its classified type.
    Returns a list of ``AssertionClaimLink``.
    """
    # Imported lazily to keep the numeric primitives free of schema imports
    # at module load and avoid any import cycle through the assertions package.
    from src.assertions.schemas import AssertionClaimLink

    link_types = numeric_link_types(claims, rel_tolerance=rel_tolerance)
    return [
        AssertionClaimLink(
            assertion_id=assertion_id,
            claim_id=claim_id,
            link_type=link_type,
            contribution_weight=1.0,
            metadata={"detector": "numeric", "rel_tolerance": rel_tolerance},
        )
        for claim_id, link_type in link_types.items()
    ]
