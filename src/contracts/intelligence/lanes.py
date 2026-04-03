"""Canonical lane definitions for the intelligence layer.

A lane is an independent processing pipeline that produces evidence,
claims, or assertions from a specific class of source material.

Lane names are a closed set — adding a new lane requires a contract
MINOR version bump and an entry here. Lane-specific modules MUST NOT
define their own lane identifiers ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

# -- Canonical lane name constants -----------------------------------------
# Use these instead of bare strings throughout the codebase.
LANE_NARRATIVE = "narrative"
LANE_FILING = "filing"
LANE_STRUCTURAL = "structural"
LANE_BACKTEST = "backtest"

# Ordered tuple of all recognized lane names.
ALL_LANES: tuple[str, ...] = (
    LANE_NARRATIVE,
    LANE_FILING,
    LANE_STRUCTURAL,
    LANE_BACKTEST,
)

# Frozen set for O(1) membership checks.
VALID_LANES: frozenset[str] = frozenset(ALL_LANES)


def validate_lane(lane: str) -> str:
    """Validate and return a lane name.

    Raises:
        ValueError: If the lane name is not in the canonical set.
    """
    if lane not in VALID_LANES:
        raise ValueError(
            f"Unknown lane {lane!r}. Must be one of {sorted(VALID_LANES)}"
        )
    return lane


@dataclass(frozen=True, slots=True)
class LaneDescriptor:
    """Metadata about a lane for registry and documentation purposes.

    Attributes:
        name: Canonical lane identifier (must be in VALID_LANES).
        description: Human-readable purpose of the lane.
        source_types: What kind of source material this lane processes.
        produces: What kind of objects this lane outputs.
    """

    name: str
    description: str
    source_types: tuple[str, ...]
    produces: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_lane(self.name)


class LaneRegistry:
    """Descriptors for all recognized intelligence lanes.

    This is documentation-as-code: downstream tasks can introspect lane
    metadata without hardcoding knowledge about what each lane does.
    """

    DESCRIPTORS: ClassVar[dict[str, LaneDescriptor]] = {
        LANE_NARRATIVE: LaneDescriptor(
            name=LANE_NARRATIVE,
            description=(
                "Real-time narrative momentum from news and social media. "
                "Detects surges, cross-platform convergence, sentiment shifts."
            ),
            source_types=("news", "twitter", "reddit", "substack"),
            produces=("claims", "narrative_runs", "signals"),
        ),
        LANE_FILING: LaneDescriptor(
            name=LANE_FILING,
            description=(
                "SEC filing analysis via edgartools. Extracts structured "
                "facts from 10-K, 10-Q, 8-K, and other filing types."
            ),
            source_types=("sec_filing",),
            produces=("claims", "filing_facts"),
        ),
        LANE_STRUCTURAL: LaneDescriptor(
            name=LANE_STRUCTURAL,
            description=(
                "Structural intelligence from supply chain, competitive, "
                "and sector relationships. Second-order basket analysis."
            ),
            source_types=("graph_edges", "domain_packs"),
            produces=("claims", "structural_edges", "baskets"),
        ),
        LANE_BACKTEST: LaneDescriptor(
            name=LANE_BACKTEST,
            description=(
                "Point-in-time backtest evaluation of published intelligence "
                "against historical price data."
            ),
            source_types=("published_manifests",),
            produces=("backtest_results", "evaluation_scores"),
        ),
    }

    @classmethod
    def get(cls, lane: str) -> LaneDescriptor:
        """Get the descriptor for a lane.

        Raises:
            ValueError: If the lane is not recognized.
        """
        validate_lane(lane)
        return cls.DESCRIPTORS[lane]
