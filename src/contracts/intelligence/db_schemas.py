"""Database-mapped schemas for the intelligence layer tables.

These dataclasses map 1:1 to the tables created in migration 018.
They are used by repositories to materialize query results and by
services to construct records for persistence.

Table mapping:
    news_intel.lane_runs       → LaneRun
    intel_pub.manifests        → Manifest
    intel_pub.manifest_pointers → ManifestPointer
    intel_pub.published_objects → PublishedObject
    intel_export.export_bundles → ExportBundle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.contracts.intelligence.lanes import VALID_LANES

# -- Valid state sets -------------------------------------------------------

VALID_RUN_STATUSES = frozenset(
    {"pending", "running", "completed", "failed", "cancelled"}
)

VALID_PUBLISH_STATES = frozenset(
    {"draft", "review", "published", "retracted"}
)

VALID_EXPORT_FORMATS = frozenset({"jsonl", "parquet", "csv"})


# -- news_intel.lane_runs --------------------------------------------------


@dataclass
class LaneRun:
    """A single lane execution record (news_intel.lane_runs).

    Attributes:
        run_id: Unique identifier for this lane execution.
        lane: Which processing lane (narrative, filing, etc.).
        status: Lifecycle state (pending → running → completed/failed/cancelled).
        contract_version: Contract version governing this run's outputs.
        started_at: When the run began executing.
        completed_at: When the run finished (success or failure).
        error_message: Error details if status is 'failed'.
        config_snapshot: Frozen config at run start for reproducibility.
        metrics: Run-level metrics (doc count, duration, etc.).
        metadata: Extensible metadata.
    """

    run_id: str
    lane: str
    status: str = "pending"
    contract_version: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if self.lane not in VALID_LANES:
            raise ValueError(
                f"Invalid lane {self.lane!r}. "
                f"Must be one of {sorted(VALID_LANES)}"
            )
        if self.status not in VALID_RUN_STATUSES:
            raise ValueError(
                f"Invalid run status {self.status!r}. "
                f"Must be one of {sorted(VALID_RUN_STATUSES)}"
            )


# -- intel_pub.manifests ----------------------------------------------------


@dataclass
class Manifest:
    """A versioned published manifest (intel_pub.manifests).

    Attributes:
        manifest_id: Unique manifest identifier.
        lane: Lane that produced the manifest contents.
        run_id: Lane-run that populated this manifest.
        contract_version: Contract version at publication time.
        published_at: When the manifest was sealed.
        object_count: Number of published objects in this manifest.
        checksum: Integrity checksum over manifest contents.
        metadata: Extensible metadata (coverage tier, model version, etc.).
    """

    manifest_id: str
    lane: str
    run_id: str
    contract_version: str
    published_at: datetime | None = None
    object_count: int = 0
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if self.lane not in VALID_LANES:
            raise ValueError(
                f"Invalid lane {self.lane!r}. "
                f"Must be one of {sorted(VALID_LANES)}"
            )


# -- intel_pub.manifest_pointers -------------------------------------------


@dataclass
class ManifestPointer:
    """Current serving pointer for a lane (intel_pub.manifest_pointers).

    Each lane has exactly one active pointer. Advancing the pointer
    atomically switches downstream consumers to a new manifest.

    Attributes:
        lane: The lane this pointer serves.
        manifest_id: Currently active manifest.
        activated_at: When this pointer was last advanced.
        previous_manifest_id: The manifest that was active before this one.
        metadata: Extensible metadata.
    """

    lane: str
    manifest_id: str
    activated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    previous_manifest_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lane not in VALID_LANES:
            raise ValueError(
                f"Invalid lane {self.lane!r}. "
                f"Must be one of {sorted(VALID_LANES)}"
            )


# -- intel_pub.published_objects --------------------------------------------


@dataclass
class PublishedObject:
    """A single published item within a manifest (intel_pub.published_objects).

    Attributes:
        object_id: Unique identifier.
        object_type: Kind of object (claim, assertion, signal, etc.).
        manifest_id: Which manifest contains this object.
        lane: Lane that produced it.
        publish_state: Lifecycle state (draft → review → published/retracted).
        contract_version: Contract version governing the object.
        source_ids: IDs of source documents/filings.
        run_id: Lane-run that produced it.
        valid_from: Bitemporal: when the fact became true.
        valid_to: Bitemporal: when the fact ceased to be true.
        payload: The object's domain-specific content.
        lineage: Full lineage metadata.
    """

    object_id: str
    object_type: str
    manifest_id: str
    lane: str
    publish_state: str = "draft"
    contract_version: str = ""
    source_ids: list[str] = field(default_factory=list)
    run_id: str = ""
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if self.lane not in VALID_LANES:
            raise ValueError(
                f"Invalid lane {self.lane!r}. "
                f"Must be one of {sorted(VALID_LANES)}"
            )
        if self.publish_state not in VALID_PUBLISH_STATES:
            raise ValueError(
                f"Invalid publish state {self.publish_state!r}. "
                f"Must be one of {sorted(VALID_PUBLISH_STATES)}"
            )


# -- intel_export.export_bundles --------------------------------------------


@dataclass
class ExportBundle:
    """An exported bundle artifact (intel_export.export_bundles).

    Attributes:
        bundle_id: Unique bundle identifier.
        manifest_id: Which manifest was exported.
        lane: Lane of the exported content.
        contract_version: Contract version at export time.
        format: Export format (jsonl, parquet, csv).
        object_count: Number of objects in the bundle.
        size_bytes: Bundle file size.
        checksum: Integrity checksum.
        exported_at: When the export was produced.
        exported_by: Who/what triggered the export.
        metadata: Extensible metadata.
    """

    bundle_id: str
    manifest_id: str
    lane: str
    contract_version: str
    format: str = "jsonl"
    object_count: int = 0
    size_bytes: int | None = None
    checksum: str | None = None
    exported_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    exported_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if self.lane not in VALID_LANES:
            raise ValueError(
                f"Invalid lane {self.lane!r}. "
                f"Must be one of {sorted(VALID_LANES)}"
            )
        if self.format not in VALID_EXPORT_FORMATS:
            raise ValueError(
                f"Invalid export format {self.format!r}. "
                f"Must be one of {sorted(VALID_EXPORT_FORMATS)}"
            )
