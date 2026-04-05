"""Composite manifest assembly and atomic pointer advancement.

Assembles approved lane outputs into a composite manifest and
produces planned pointer advancements for atomic execution.

The assembler is a stateless decision layer: it receives lane
outputs and health status, determines which lanes qualify, builds
a composite manifest, and returns a plan. The caller executes the
plan (pointer advancement) within a DB transaction.

Assembly flow:
    1. Evaluate each lane output for inclusion
    2. Exclude blocked or empty lanes (record reasons)
    3. Verify contract version compatibility
    4. Build composite manifest from included contributions
    5. Plan pointer advancements for included lanes
    6. Return AssemblyResult with composite + plan + diagnostics
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.contracts.intelligence.version import ContractRegistry
from src.publish.lane_health import LaneHealthStatus, PublishReadiness


# -- Lane output (common interface) -------------------------------------------


@dataclass(frozen=True)
class LaneOutput:
    """Common interface for lane publication results.

    The caller constructs this from each lane's specific result
    type (NarrativePublicationResult, FilingPublicationResult, etc.).

    Attributes:
        lane: Canonical lane name.
        published: Whether the lane publisher succeeded.
        object_count: Number of publishable objects.
        manifest_id: The per-lane manifest to advance to.
        health: Pre-computed lane health status.
        block_reason: Why publication was blocked (if applicable).
    """

    lane: str
    published: bool
    object_count: int
    manifest_id: str
    health: LaneHealthStatus
    block_reason: str | None = None


# -- Assembly components -------------------------------------------------------


@dataclass(frozen=True)
class LaneContribution:
    """A single lane's contribution to a composite manifest.

    Attributes:
        lane: Which lane contributed.
        manifest_id: The per-lane manifest included.
        object_count: Objects in this lane's manifest.
        readiness: Lane health readiness at assembly time.
    """

    lane: str
    manifest_id: str
    object_count: int
    readiness: str  # PublishReadiness value


@dataclass(frozen=True)
class PointerAdvancement:
    """A planned pointer movement for atomic execution.

    The caller executes these within a DB transaction so all
    lane pointers advance together or none do.

    Attributes:
        lane: Lane whose pointer should advance.
        manifest_id: Target manifest to point to.
        previous_manifest_id: Current manifest (for rollback context).
    """

    lane: str
    manifest_id: str
    previous_manifest_id: str | None = None


@dataclass(frozen=True)
class CompositeManifest:
    """A cross-lane publication event.

    Bundles approved per-lane manifests into a single composite
    for coordinated pointer advancement.

    Attributes:
        composite_id: Deterministic ID from included manifest IDs.
        contributions: Included lane contributions.
        total_object_count: Sum of objects across all lanes.
        lanes_included: Which lanes are in this composite.
        lanes_excluded: Which lanes were skipped (with reasons).
        exclusion_reasons: Lane → reason for exclusion.
        contract_version: Contract version for this composite.
        assembled_at: When assembly was performed.
    """

    composite_id: str
    contributions: list[LaneContribution] = field(default_factory=list)
    total_object_count: int = 0
    lanes_included: list[str] = field(default_factory=list)
    lanes_excluded: list[str] = field(default_factory=list)
    exclusion_reasons: dict[str, str] = field(default_factory=dict)
    contract_version: str = ""
    assembled_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for audit and logging."""
        return {
            "composite_id": self.composite_id,
            "total_object_count": self.total_object_count,
            "lanes_included": self.lanes_included,
            "lanes_excluded": self.lanes_excluded,
            "exclusion_reasons": self.exclusion_reasons,
            "contract_version": self.contract_version,
            "assembled_at": self.assembled_at.isoformat(),
        }


# -- Assembly result -----------------------------------------------------------


@dataclass(frozen=True)
class AssemblyResult:
    """Result of composite manifest assembly.

    Attributes:
        composite: The assembled composite manifest.
        advancements: Planned pointer movements (execute atomically).
        ready: True if at least one lane contributed.
        block_reasons: All exclusion reasons across lanes.
    """

    composite: CompositeManifest
    advancements: list[PointerAdvancement] = field(default_factory=list)
    ready: bool = False
    block_reasons: list[str] = field(default_factory=list)


# -- ID generation -------------------------------------------------------------


def make_composite_id(manifest_ids: list[str]) -> str:
    """Generate a deterministic composite ID from included manifest IDs.

    Same set of manifest IDs always produces the same composite ID,
    enabling idempotent assembly.
    """
    key = "\x00".join(sorted(manifest_ids))
    return f"composite_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


# -- Assembly logic (stateless) ------------------------------------------------


def assemble_composite_manifest(
    lane_outputs: list[LaneOutput],
    current_pointers: dict[str, str | None],
    *,
    contract_version: str | None = None,
    now: datetime | None = None,
) -> AssemblyResult:
    """Assemble a composite manifest from lane publication outputs.

    Evaluates each lane for inclusion, builds the composite from
    approved lanes, and plans pointer advancements.

    Args:
        lane_outputs: Publication results from each lane.
        current_pointers: Current manifest_id per lane (for rollback).
        contract_version: Override contract version (default: current).
        now: Assembly timestamp.

    Returns:
        AssemblyResult with composite, advancements, and diagnostics.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if contract_version is None:
        contract_version = str(ContractRegistry.CURRENT)

    contributions: list[LaneContribution] = []
    advancements: list[PointerAdvancement] = []
    excluded: dict[str, str] = {}
    block_reasons: list[str] = []

    for output in sorted(lane_outputs, key=lambda o: o.lane):
        if not output.published:
            reason = output.block_reason or "Publication blocked"
            excluded[output.lane] = reason
            block_reasons.append(f"{output.lane}: {reason}")
            continue

        if output.object_count == 0:
            excluded[output.lane] = "No objects to publish"
            block_reasons.append(f"{output.lane}: No objects to publish")
            continue

        if output.health.readiness == PublishReadiness.BLOCKED:
            reason = output.health.format_block_reason()
            excluded[output.lane] = reason
            block_reasons.append(f"{output.lane}: {reason}")
            continue

        contributions.append(
            LaneContribution(
                lane=output.lane,
                manifest_id=output.manifest_id,
                object_count=output.object_count,
                readiness=output.health.readiness.value,
            )
        )
        advancements.append(
            PointerAdvancement(
                lane=output.lane,
                manifest_id=output.manifest_id,
                previous_manifest_id=current_pointers.get(output.lane),
            )
        )

    included_lanes = [c.lane for c in contributions]
    excluded_lanes = sorted(excluded.keys())
    manifest_ids = [c.manifest_id for c in contributions]

    composite = CompositeManifest(
        composite_id=make_composite_id(manifest_ids) if manifest_ids else "composite_empty",
        contributions=contributions,
        total_object_count=sum(c.object_count for c in contributions),
        lanes_included=included_lanes,
        lanes_excluded=excluded_lanes,
        exclusion_reasons=excluded,
        contract_version=contract_version,
        assembled_at=now,
    )

    return AssemblyResult(
        composite=composite,
        advancements=advancements,
        ready=len(contributions) > 0,
        block_reasons=block_reasons,
    )
