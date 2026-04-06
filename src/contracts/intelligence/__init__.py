"""Intelligence contract package — the canonical source of truth.

This package defines the public contract surface for news-tracker's
intelligence layer. All published objects (claims, assertions, manifests,
bundles) conform to the schemas defined here.

Ownership: news-tracker is the sole producer. Downstream consumers
(e.g., stock-screener) import these types but must not modify them.

Versioning: ContractRegistry.CURRENT is the active contract version.
Bump MINOR for additive changes, MAJOR for breaking changes.
"""

from src.contracts.intelligence.db_schemas import (
    VALID_EXPORT_FORMATS,
    VALID_PUBLISH_STATES,
    VALID_RUN_STATUSES,
    ExportBundle,
    LaneRun,
    Manifest,
    ManifestPointer,
    PublishedObject,
)
from src.contracts.intelligence.lanes import (
    ALL_LANES,
    LANE_BACKTEST,
    LANE_FILING,
    LANE_NARRATIVE,
    LANE_STRUCTURAL,
    VALID_LANES,
    LaneDescriptor,
    LaneRegistry,
    validate_lane,
)
from src.contracts.intelligence.ownership import (
    CONSUMER_REPOS,
    OWNER_REPO,
    CompatibilityResult,
    OwnershipPolicy,
    check_compatibility,
)
from src.contracts.intelligence.schemas import (
    Lineage,
    ManifestHeader,
    PublishedObjectRef,
    PublishState,
    ReviewDecision,
)
from src.contracts.intelligence.version import (
    ContractRegistry,
    ContractVersion,
)

__all__ = [
    # Version
    "ContractRegistry",
    "ContractVersion",
    # Lanes
    "ALL_LANES",
    "LANE_BACKTEST",
    "LANE_FILING",
    "LANE_NARRATIVE",
    "LANE_STRUCTURAL",
    "VALID_LANES",
    "LaneDescriptor",
    "LaneRegistry",
    "validate_lane",
    # Schemas (Pydantic contract models)
    "Lineage",
    "ManifestHeader",
    "PublishedObjectRef",
    "PublishState",
    "ReviewDecision",
    # DB schemas (dataclasses mirroring intelligence tables)
    "VALID_EXPORT_FORMATS",
    "VALID_PUBLISH_STATES",
    "VALID_RUN_STATUSES",
    "ExportBundle",
    "LaneRun",
    "Manifest",
    "ManifestPointer",
    "PublishedObject",
    # Ownership
    "CONSUMER_REPOS",
    "OWNER_REPO",
    "CompatibilityResult",
    "OwnershipPolicy",
    "check_compatibility",
]
