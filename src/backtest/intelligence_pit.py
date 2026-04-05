"""Point-in-time filters for the intelligence layer substrate.

Extends the backtest engine to reconstruct claims, assertions,
filings, lane runs, and manifests at any evaluation timestamp
without look-ahead leakage.

No-lookahead rules:
    - Claims: visible if created_at <= as_of
    - Assertions: visible if created_at <= as_of
    - Filings: visible if ingested_at <= as_of
    - Lane runs: visible if completed_at <= as_of AND completed
    - Manifests: visible if published_at <= as_of

All functions are stateless — the caller provides pre-fetched
entity lists, the filters select what was known at as_of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.assertions.schemas import ResolvedAssertion
from src.contracts.intelligence.db_schemas import LaneRun, Manifest


# -- Point-in-time filter functions -------------------------------------------


def filter_claims_pit(
    claims: list[dict[str, Any]],
    as_of: datetime,
) -> list[dict[str, Any]]:
    """Filter claims to those known at as_of.

    Uses created_at as the PIT anchor — when the claim was
    first ingested, not when the source was published.

    Claims are passed as dicts (from repository queries) since
    the EvidenceClaim schema varies by claim type.
    """
    return [
        c for c in claims
        if c.get("created_at") is not None and c["created_at"] <= as_of
    ]


def filter_assertions_pit(
    assertions: list[ResolvedAssertion],
    as_of: datetime,
) -> list[ResolvedAssertion]:
    """Filter assertions to those known at as_of.

    Uses created_at as the PIT anchor. All assertion states
    (active, disputed, retracted, superseded) are included —
    the state at as_of tells the backtest whether the assertion
    was believed at that time.
    """
    return [a for a in assertions if a.created_at <= as_of]


def filter_filings_pit(
    filings: list[dict[str, Any]],
    as_of: datetime,
) -> list[dict[str, Any]]:
    """Filter filings to those ingested by as_of.

    Uses ingested_at (when we fetched from EDGAR), NOT
    source_published_at (when SEC published). This prevents
    look-ahead from late-fetched filings.

    Filings are passed as dicts for flexibility across
    FilingRecord and raw query results.
    """
    return [
        f for f in filings
        if f.get("ingested_at") is not None and f["ingested_at"] <= as_of
    ]


def filter_lane_runs_pit(
    runs: list[LaneRun],
    as_of: datetime,
    *,
    lane: str | None = None,
) -> list[LaneRun]:
    """Filter lane runs to those completed by as_of.

    Only completed runs are visible — a running or pending run
    hasn't produced results yet. Uses completed_at as the PIT
    anchor, NOT started_at.

    Args:
        runs: All lane runs to filter.
        as_of: Evaluation timestamp.
        lane: Optional lane filter.
    """
    return [
        r for r in runs
        if r.status == "completed"
        and r.completed_at is not None
        and r.completed_at <= as_of
        and (lane is None or r.lane == lane)
    ]


def filter_manifests_pit(
    manifests: list[Manifest],
    as_of: datetime,
    *,
    lane: str | None = None,
) -> list[Manifest]:
    """Filter manifests to those sealed by as_of.

    Uses published_at (when sealed), NOT created_at (when the
    empty manifest was created). An unsealed manifest has no
    published_at and is never visible in PIT queries.

    Args:
        manifests: All manifests to filter.
        as_of: Evaluation timestamp.
        lane: Optional lane filter.
    """
    return [
        m for m in manifests
        if m.published_at is not None
        and m.published_at <= as_of
        and (lane is None or m.lane == lane)
    ]


def get_active_manifest_pit(
    manifests: list[Manifest],
    as_of: datetime,
    lane: str,
) -> Manifest | None:
    """Get the most recently sealed manifest for a lane at as_of.

    This reconstructs which manifest the pointer would have been
    pointing to at as_of — the latest sealed manifest for the lane.

    Returns None if no manifest was sealed for this lane by as_of.
    """
    visible = filter_manifests_pit(manifests, as_of, lane=lane)
    if not visible:
        return None
    return max(visible, key=lambda m: m.published_at)  # type: ignore[arg-type]


# -- Intelligence snapshot -----------------------------------------------------


@dataclass(frozen=True)
class IntelligenceSnapshot:
    """Point-in-time snapshot of the intelligence layer.

    Represents the complete state that was known at as_of,
    enabling backtest replay without look-ahead.

    Attributes:
        as_of: The evaluation timestamp.
        claims: Claims known at as_of.
        assertions: Assertions known at as_of.
        filings: Filings ingested by as_of.
        lane_runs: Completed lane runs by as_of.
        manifests: Sealed manifests by as_of.
        active_manifests: Most recent manifest per lane at as_of.
    """

    as_of: datetime
    claims: list[dict[str, Any]] = field(default_factory=list)
    assertions: list[ResolvedAssertion] = field(default_factory=list)
    filings: list[dict[str, Any]] = field(default_factory=list)
    lane_runs: list[LaneRun] = field(default_factory=list)
    manifests: list[Manifest] = field(default_factory=list)
    active_manifests: dict[str, Manifest] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Summary for audit logging."""
        return {
            "as_of": self.as_of.isoformat(),
            "claim_count": len(self.claims),
            "assertion_count": len(self.assertions),
            "filing_count": len(self.filings),
            "lane_run_count": len(self.lane_runs),
            "manifest_count": len(self.manifests),
            "active_manifest_lanes": sorted(self.active_manifests.keys()),
        }


# -- Snapshot builder ----------------------------------------------------------


def build_intelligence_snapshot(
    as_of: datetime,
    claims: list[dict[str, Any]],
    assertions: list[ResolvedAssertion],
    filings: list[dict[str, Any]],
    lane_runs: list[LaneRun],
    manifests: list[Manifest],
) -> IntelligenceSnapshot:
    """Build a point-in-time snapshot from pre-fetched data.

    Applies PIT filters to all entity types and reconstructs
    the active manifest per lane.

    Args:
        as_of: Evaluation timestamp.
        claims: All claims (unfiltered).
        assertions: All assertions (unfiltered).
        filings: All filings (unfiltered).
        lane_runs: All lane runs (unfiltered).
        manifests: All manifests (unfiltered).

    Returns:
        IntelligenceSnapshot with only data known at as_of.
    """
    pit_claims = filter_claims_pit(claims, as_of)
    pit_assertions = filter_assertions_pit(assertions, as_of)
    pit_filings = filter_filings_pit(filings, as_of)
    pit_runs = filter_lane_runs_pit(lane_runs, as_of)
    pit_manifests = filter_manifests_pit(manifests, as_of)

    active: dict[str, Manifest] = {}
    for m in pit_manifests:
        if m.lane not in active or (
            m.published_at is not None
            and (active[m.lane].published_at is None
                 or m.published_at > active[m.lane].published_at)
        ):
            active[m.lane] = m

    return IntelligenceSnapshot(
        as_of=as_of,
        claims=pit_claims,
        assertions=pit_assertions,
        filings=pit_filings,
        lane_runs=pit_runs,
        manifests=pit_manifests,
        active_manifests=active,
    )


# -- No-lookahead validation ---------------------------------------------------


def validate_no_lookahead(
    snapshot: IntelligenceSnapshot,
) -> list[str]:
    """Validate that no data in the snapshot violates no-lookahead.

    Returns a list of violation descriptions (empty = clean).
    """
    violations: list[str] = []
    as_of = snapshot.as_of

    for c in snapshot.claims:
        if c.get("created_at") and c["created_at"] > as_of:
            violations.append(
                f"Claim {c.get('claim_id', '?')} created_at "
                f"{c['created_at']} > as_of {as_of}"
            )

    for a in snapshot.assertions:
        if a.created_at > as_of:
            violations.append(
                f"Assertion {a.assertion_id} created_at "
                f"{a.created_at} > as_of {as_of}"
            )

    for f in snapshot.filings:
        if f.get("ingested_at") and f["ingested_at"] > as_of:
            violations.append(
                f"Filing {f.get('accession_number', '?')} ingested_at "
                f"{f['ingested_at']} > as_of {as_of}"
            )

    for r in snapshot.lane_runs:
        if r.completed_at and r.completed_at > as_of:
            violations.append(
                f"LaneRun {r.run_id} completed_at "
                f"{r.completed_at} > as_of {as_of}"
            )

    for m in snapshot.manifests:
        if m.published_at and m.published_at > as_of:
            violations.append(
                f"Manifest {m.manifest_id} published_at "
                f"{m.published_at} > as_of {as_of}"
            )

    return violations
