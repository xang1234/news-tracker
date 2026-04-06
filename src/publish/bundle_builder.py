"""Composite bundle builder for immutable export artifacts.

Builds a complete bundle from a composite manifest and lane data,
producing named artifacts (manifest.json, objects.jsonl, rollups.jsonl)
with per-artifact checksums and an overall integrity root.

The builder is stateless — it receives pre-serialized data and
produces immutable bundle content. The caller handles writing to
disk, S3, or any other storage.

Bundle structure:
    manifest.json   — composite metadata + artifact checksums
    objects.jsonl   — all published objects across lanes (sorted by ID)
    rollups.jsonl   — all rollup/summary records

Integrity:
    Each artifact has a SHA-256 checksum. The overall bundle checksum
    is computed from the manifest.json content (which includes all
    artifact checksums), creating a single verification root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.publish.exporter import compute_bundle_checksum
from src.publish.manifest_assembly import CompositeManifest

# -- Artifact dataclass --------------------------------------------------------


@dataclass(frozen=True)
class BundleArtifact:
    """A single named artifact within a bundle.

    Attributes:
        name: Artifact filename (e.g., "objects.jsonl").
        lines: Content lines (one per record for JSONL, single for JSON).
        checksum: SHA-256 of the content bytes.
        record_count: Number of records in this artifact.
    """

    name: str
    lines: list[str] = field(default_factory=list)
    checksum: str = ""
    record_count: int = 0


# -- Bundle dataclass ----------------------------------------------------------


@dataclass(frozen=True)
class CompositeBundle:
    """Complete immutable bundle ready for writing.

    Attributes:
        composite_id: From the source composite manifest.
        contract_version: Contract version for this bundle.
        artifacts: Named artifacts (name → artifact).
        overall_checksum: SHA-256 of manifest.json (integrity root).
        total_records: Sum of records across all artifacts.
        created_at: When the bundle was built.
    """

    composite_id: str
    contract_version: str
    artifacts: dict[str, BundleArtifact] = field(default_factory=dict)
    overall_checksum: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_records(self) -> int:
        """Sum of records across all artifacts."""
        return sum(a.record_count for a in self.artifacts.values())

    def to_dict(self) -> dict[str, Any]:
        """Summary serialization for audit."""
        return {
            "composite_id": self.composite_id,
            "contract_version": self.contract_version,
            "artifact_names": sorted(self.artifacts.keys()),
            "overall_checksum": self.overall_checksum,
            "total_records": self.total_records,
            "created_at": self.created_at.isoformat(),
        }


# -- Artifact builders (stateless) ---------------------------------------------


def _build_jsonl_artifact(
    name: str,
    items: list[dict[str, Any]],
) -> BundleArtifact:
    """Build a JSONL artifact from a list of serialized dicts."""
    lines = [json.dumps(item, sort_keys=True) for item in items]
    return BundleArtifact(
        name=name,
        lines=lines,
        checksum=compute_bundle_checksum(lines),
        record_count=len(items),
    )


def build_objects_artifact(
    lane_objects: dict[str, list[dict[str, Any]]],
) -> BundleArtifact:
    """Build objects.jsonl from per-lane serialized objects.

    Combines all lane objects into a single sorted JSONL artifact.
    Each object dict should include at minimum an "object_id" key.

    Args:
        lane_objects: Mapping of lane → list of serialized object dicts.

    Returns:
        BundleArtifact named "objects.jsonl".
    """
    all_objects: list[dict[str, Any]] = []
    for lane in sorted(lane_objects.keys()):
        all_objects.extend(lane_objects[lane])
    all_objects.sort(key=lambda o: o.get("object_id", ""))
    return _build_jsonl_artifact("objects.jsonl", all_objects)


def build_rollups_artifact(
    rollups: list[dict[str, Any]],
) -> BundleArtifact:
    """Build rollups.jsonl from rollup/summary records.

    Accepts any serialized rollup dicts (symbol rollups, issuer
    summaries, basket payloads, path explanations, etc.).

    Args:
        rollups: List of serialized rollup dicts.

    Returns:
        BundleArtifact named "rollups.jsonl".
    """
    return _build_jsonl_artifact("rollups.jsonl", rollups)


def build_manifest_artifact(
    composite: CompositeManifest,
    artifact_checksums: dict[str, str],
    now: datetime,
) -> BundleArtifact:
    """Build manifest.json with composite metadata and artifact checksums.

    The manifest.json is the integrity root — its checksum is the
    overall bundle checksum.

    Args:
        composite: The source composite manifest.
        artifact_checksums: Checksums of other artifacts.
        now: Bundle creation timestamp.

    Returns:
        BundleArtifact named "manifest.json".
    """
    manifest_data = {
        "_type": "composite_bundle_manifest",
        "composite_id": composite.composite_id,
        "contract_version": composite.contract_version,
        "lanes_included": composite.lanes_included,
        "lanes_excluded": composite.lanes_excluded,
        "exclusion_reasons": composite.exclusion_reasons,
        "total_object_count": composite.total_object_count,
        "contributions": [
            {
                "lane": c.lane,
                "manifest_id": c.manifest_id,
                "object_count": c.object_count,
                "readiness": c.readiness.value,
            }
            for c in composite.contributions
        ],
        "artifact_checksums": artifact_checksums,
        "assembled_at": composite.assembled_at.isoformat(),
        "bundle_created_at": now.isoformat(),
    }
    content = json.dumps(manifest_data, sort_keys=True, indent=2)
    lines = [content]
    return BundleArtifact(
        name="manifest.json",
        lines=lines,
        checksum=compute_bundle_checksum(lines),
        record_count=1,
    )


# -- Bundle builder (stateless) ------------------------------------------------


def build_composite_bundle(
    composite: CompositeManifest,
    lane_objects: dict[str, list[dict[str, Any]]],
    rollups: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> CompositeBundle:
    """Build a complete composite bundle from manifest and data.

    Produces named artifacts with checksums. The caller writes the
    artifacts to disk, S3, or any other storage backend.

    Args:
        composite: The assembled composite manifest.
        lane_objects: Per-lane serialized published objects.
        rollups: All rollup/summary records (serialized dicts).
        now: Bundle creation timestamp.

    Returns:
        CompositeBundle with all artifacts and overall checksum.
    """
    if now is None:
        now = datetime.now(UTC)

    objects_art = build_objects_artifact(lane_objects)
    rollups_art = build_rollups_artifact(rollups)

    artifact_checksums = {
        objects_art.name: objects_art.checksum,
        rollups_art.name: rollups_art.checksum,
    }
    manifest_art = build_manifest_artifact(composite, artifact_checksums, now)

    artifacts = {
        manifest_art.name: manifest_art,
        objects_art.name: objects_art,
        rollups_art.name: rollups_art,
    }

    return CompositeBundle(
        composite_id=composite.composite_id,
        contract_version=composite.contract_version,
        artifacts=artifacts,
        overall_checksum=manifest_art.checksum,
        created_at=now,
    )


# -- Integrity verification ---------------------------------------------------


def verify_bundle_integrity(bundle: CompositeBundle) -> bool:
    """Verify all artifact checksums match their content.

    Recomputes checksums from artifact lines and compares against
    stored checksums. Returns True only if all match.
    """
    for artifact in bundle.artifacts.values():
        recomputed = compute_bundle_checksum(artifact.lines)
        if recomputed != artifact.checksum:
            return False
    return True


def check_bundle_parity(
    bundle: CompositeBundle,
    expected_objects: dict[str, list[dict[str, Any]]],
    expected_rollups: list[dict[str, Any]],
) -> list[str]:
    """Check bundle parity with expected state.

    Compares the bundle's object and rollup artifacts against
    expected data. Returns a list of mismatch descriptions
    (empty if parity holds).

    Args:
        bundle: The bundle to verify.
        expected_objects: Per-lane expected objects (same format as build input).
        expected_rollups: Expected rollup records.

    Returns:
        List of mismatch descriptions (empty = parity holds).
    """
    mismatches: list[str] = []

    expected_obj_art = build_objects_artifact(expected_objects)
    actual_obj_art = bundle.artifacts.get("objects.jsonl")
    if actual_obj_art is None:
        mismatches.append("Missing objects.jsonl artifact")
    elif actual_obj_art.checksum != expected_obj_art.checksum:
        mismatches.append(
            f"objects.jsonl checksum mismatch: "
            f"bundle={actual_obj_art.checksum}, "
            f"expected={expected_obj_art.checksum}"
        )

    expected_roll_art = build_rollups_artifact(expected_rollups)
    actual_roll_art = bundle.artifacts.get("rollups.jsonl")
    if actual_roll_art is None:
        mismatches.append("Missing rollups.jsonl artifact")
    elif actual_roll_art.checksum != expected_roll_art.checksum:
        mismatches.append(
            f"rollups.jsonl checksum mismatch: "
            f"bundle={actual_roll_art.checksum}, "
            f"expected={expected_roll_art.checksum}"
        )

    return mismatches
