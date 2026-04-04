"""Publish service — orchestration layer for the intelligence publish lifecycle.

Provides lane-neutral business logic for:
    - Lane run lifecycle: create → start → complete / fail / cancel
    - Manifest lifecycle: create → seal → publish (via pointer advance)
    - Published object state transitions: draft → review → published / retracted
    - Atomic pointer advancement: switch downstream consumers to a new manifest

All state transitions are validated before persistence. Invalid transitions
raise ValueError with a descriptive message.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import json

from src.contracts.intelligence.db_schemas import (
    VALID_PUBLISH_STATES,
    VALID_RUN_STATUSES,
    LaneRun,
    Manifest,
    ManifestPointer,
    PublishedObject,
)
from src.contracts.intelligence.lanes import validate_lane
from src.contracts.intelligence.ownership import OwnershipPolicy
from src.contracts.intelligence.version import ContractRegistry
from src.publish.repository import PublishRepository

logger = logging.getLogger(__name__)

# -- State machine definitions ---------------------------------------------

# Valid run status transitions: from → set of allowed targets.
RUN_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running", "cancelled"}),
    "running": frozenset({"completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}

# Valid publish state transitions: from → set of allowed targets.
PUBLISH_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"review", "published", "retracted"}),
    "review": frozenset({"published", "retracted", "draft"}),
    "published": frozenset({"retracted"}),
    "retracted": frozenset(),
}


def _validate_run_transition(current: str, target: str) -> None:
    """Validate a lane run status transition."""
    if current not in VALID_RUN_STATUSES:
        raise ValueError(f"Unknown current run status: {current!r}")
    if target not in VALID_RUN_STATUSES:
        raise ValueError(f"Unknown target run status: {target!r}")
    allowed = RUN_TRANSITIONS[current]
    if target not in allowed:
        raise ValueError(
            f"Invalid run transition: {current!r} → {target!r}. "
            f"Allowed from {current!r}: {sorted(allowed) or 'none (terminal)'}"
        )


def _validate_publish_transition(current: str, target: str) -> None:
    """Validate a published object state transition."""
    if current not in VALID_PUBLISH_STATES:
        raise ValueError(f"Unknown current publish state: {current!r}")
    if target not in VALID_PUBLISH_STATES:
        raise ValueError(f"Unknown target publish state: {target!r}")
    allowed = PUBLISH_TRANSITIONS[current]
    if target not in allowed:
        raise ValueError(
            f"Invalid publish transition: {current!r} → {target!r}. "
            f"Allowed from {current!r}: {sorted(allowed) or 'none (terminal)'}"
        )


def _generate_id(prefix: str) -> str:
    """Generate a unique ID with a descriptive prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class PublishService:
    """Lane-neutral orchestration for the publish lifecycle.

    This service is the single entry point for all publish operations.
    Lane-specific workers call these methods rather than writing to
    the publish tables directly.
    """

    def __init__(self, repository: PublishRepository) -> None:
        self._repo = repository

    # -- Lane run lifecycle ------------------------------------------------

    async def create_run(
        self,
        lane: str,
        *,
        config_snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LaneRun:
        """Create a new lane run in 'pending' status.

        Args:
            lane: Canonical lane name.
            config_snapshot: Frozen config at run creation for reproducibility.
            metadata: Extensible metadata.

        Returns:
            The persisted LaneRun.
        """
        validate_lane(lane)
        run = LaneRun(
            run_id=_generate_id("run"),
            lane=lane,
            status="pending",
            contract_version=str(ContractRegistry.CURRENT),
            config_snapshot=config_snapshot or {},
            metadata=metadata or {},
        )
        return await self._repo.create_lane_run(run)

    async def start_run(self, run_id: str) -> LaneRun:
        """Transition a lane run from 'pending' to 'running'."""
        run = await self._repo.get_lane_run(run_id)
        if run is None:
            raise ValueError(f"Lane run not found: {run_id}")
        _validate_run_transition(run.status, "running")
        result = await self._repo.update_lane_run_status(run_id, "running")
        if result is None:
            raise ValueError(f"Failed to start run: {run_id}")
        logger.info("Lane run started: %s (lane=%s)", run_id, result.lane)
        return result

    async def complete_run(
        self,
        run_id: str,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> LaneRun:
        """Transition a lane run to 'completed'."""
        run = await self._repo.get_lane_run(run_id)
        if run is None:
            raise ValueError(f"Lane run not found: {run_id}")
        _validate_run_transition(run.status, "completed")
        result = await self._repo.update_lane_run_status(
            run_id, "completed", metrics=metrics
        )
        if result is None:
            raise ValueError(f"Failed to complete run: {run_id}")
        logger.info("Lane run completed: %s (lane=%s)", run_id, result.lane)
        return result

    async def fail_run(
        self,
        run_id: str,
        error_message: str,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> LaneRun:
        """Transition a lane run to 'failed' with an error message."""
        run = await self._repo.get_lane_run(run_id)
        if run is None:
            raise ValueError(f"Lane run not found: {run_id}")
        _validate_run_transition(run.status, "failed")
        result = await self._repo.update_lane_run_status(
            run_id, "failed", error_message=error_message, metrics=metrics
        )
        if result is None:
            raise ValueError(f"Failed to mark run failed: {run_id}")
        logger.warning(
            "Lane run failed: %s (lane=%s): %s",
            run_id,
            result.lane,
            error_message,
        )
        return result

    async def cancel_run(self, run_id: str) -> LaneRun:
        """Transition a lane run to 'cancelled'."""
        run = await self._repo.get_lane_run(run_id)
        if run is None:
            raise ValueError(f"Lane run not found: {run_id}")
        _validate_run_transition(run.status, "cancelled")
        result = await self._repo.update_lane_run_status(run_id, "cancelled")
        if result is None:
            raise ValueError(f"Failed to cancel run: {run_id}")
        logger.info("Lane run cancelled: %s (lane=%s)", run_id, result.lane)
        return result

    async def get_run(self, run_id: str) -> LaneRun | None:
        """Fetch a lane run by ID."""
        return await self._repo.get_lane_run(run_id)

    async def list_runs(
        self,
        lane: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[LaneRun]:
        """List lane runs with optional filters."""
        return await self._repo.list_lane_runs(
            lane=lane, status=status, limit=limit
        )

    # -- Manifest lifecycle ------------------------------------------------

    async def create_manifest(
        self,
        lane: str,
        run_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Manifest:
        """Create a new manifest for a lane run.

        The manifest starts with object_count=0 and no checksum.
        Call seal_manifest() after populating it with objects.

        Args:
            lane: Canonical lane name.
            run_id: The lane run that produced this manifest's contents.
            metadata: Extensible metadata (coverage tier, etc.).

        Returns:
            The persisted Manifest.

        Raises:
            ValueError: If the lane is invalid or the run doesn't exist
                or belongs to a different lane.
        """
        validate_lane(lane)
        run = await self._repo.get_lane_run(run_id)
        if run is None:
            raise ValueError(f"Lane run not found: {run_id}")
        if run.lane != lane:
            raise ValueError(
                f"Run {run_id} belongs to lane {run.lane!r}, not {lane!r}"
            )
        manifest = Manifest(
            manifest_id=_generate_id("manifest"),
            lane=lane,
            run_id=run_id,
            contract_version=run.contract_version,
            metadata=metadata or {},
        )
        return await self._repo.create_manifest(manifest)

    async def seal_manifest(
        self,
        manifest_id: str,
        *,
        checksum: str | None = None,
    ) -> Manifest:
        """Seal a manifest after all its objects are in 'published' state.

        Validates that every object in the manifest has been transitioned
        to 'published'. Derives the object count from the DB and computes
        a content checksum if none is provided.

        Args:
            manifest_id: Manifest to seal.
            checksum: Integrity checksum (optional; auto-computed if omitted).

        Returns:
            The updated Manifest.

        Raises:
            ValueError: If the manifest doesn't exist, is already sealed,
                or contains non-published objects.
        """
        manifest = await self._repo.get_manifest(manifest_id)
        if manifest is None:
            raise ValueError(f"Manifest not found: {manifest_id}")
        if manifest.published_at is not None:
            raise ValueError(f"Manifest {manifest_id} is already sealed")

        # Verify all objects are published
        all_objects = await self._repo.list_objects_by_manifest(manifest_id)
        non_published = [
            o for o in all_objects if o.publish_state != "published"
        ]
        if non_published:
            states = {o.publish_state for o in non_published}
            raise ValueError(
                f"Cannot seal manifest {manifest_id}: "
                f"{len(non_published)} object(s) still in {sorted(states)}"
            )

        object_count = len(all_objects)
        if checksum is None:
            checksum = self.compute_checksum(all_objects)

        result = await self._repo.update_manifest(
            manifest_id,
            object_count=object_count,
            checksum=checksum,
            published_at=datetime.now(timezone.utc),
        )
        if result is None:
            raise ValueError(f"Failed to seal manifest: {manifest_id}")
        logger.info(
            "Manifest sealed: %s (objects=%d, checksum=%s)",
            manifest_id,
            object_count,
            checksum or "none",
        )
        return result

    async def get_manifest(self, manifest_id: str) -> Manifest | None:
        """Fetch a manifest by ID."""
        return await self._repo.get_manifest(manifest_id)

    # -- Pointer advancement -----------------------------------------------

    async def advance_pointer(
        self,
        lane: str,
        manifest_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ManifestPointer:
        """Atomically advance the serving pointer for a lane.

        This is the publish operation: downstream consumers reading
        the pointer will see the new manifest after this call.

        Args:
            lane: Canonical lane name.
            manifest_id: Manifest to point to (must exist).
            metadata: Extensible metadata.

        Returns:
            The updated ManifestPointer.

        Raises:
            ValueError: If the manifest doesn't exist.
        """
        validate_lane(lane)
        manifest = await self._repo.get_manifest(manifest_id)
        if manifest is None:
            raise ValueError(
                f"Cannot advance pointer: manifest {manifest_id} not found"
            )
        if manifest.lane != lane:
            raise ValueError(
                f"Manifest {manifest_id} belongs to lane {manifest.lane!r}, "
                f"not {lane!r}"
            )
        if manifest.published_at is None:
            raise ValueError(
                f"Cannot advance pointer: manifest {manifest_id} has not "
                f"been sealed. Call seal_manifest() first."
            )
        run = await self._repo.get_lane_run(manifest.run_id)
        if run is None or run.status != "completed":
            run_status = run.status if run else "missing"
            raise ValueError(
                f"Cannot advance pointer: lane run {manifest.run_id} "
                f"is {run_status!r}, not 'completed'"
            )
        pointer = await self._repo.advance_pointer(
            lane, manifest_id, metadata=metadata
        )
        logger.info(
            "Pointer advanced: lane=%s → manifest=%s (previous=%s)",
            lane,
            manifest_id,
            pointer.previous_manifest_id or "none",
        )
        return pointer

    async def get_pointer(self, lane: str) -> ManifestPointer | None:
        """Get the current serving pointer for a lane."""
        validate_lane(lane)
        return await self._repo.get_pointer(lane)

    # -- Published object state transitions --------------------------------

    async def add_object(
        self,
        manifest_id: str,
        *,
        object_type: str,
        lane: str,
        run_id: str,
        payload: dict[str, Any] | None = None,
        source_ids: list[str] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        lineage: dict[str, Any] | None = None,
    ) -> PublishedObject:
        """Add a published object to a manifest in 'draft' state.

        Args:
            manifest_id: Target manifest.
            object_type: Kind of object (must be in PUBLISHABLE_OBJECT_TYPES).
            lane: Lane that produced it.
            run_id: Lane run that produced it.
            payload: The object's domain-specific content.
            source_ids: IDs of source documents/filings.
            valid_from: Bitemporal: when the fact became true.
            valid_to: Bitemporal: when the fact ceased to be true.
            lineage: Full lineage metadata.

        Returns:
            The persisted PublishedObject in 'draft' state.

        Raises:
            ValueError: If the manifest doesn't exist, the lane doesn't
                match, or the object_type is not publishable.
        """
        # Validate manifest exists, is unsealed, and fields match
        manifest = await self._repo.get_manifest(manifest_id)
        if manifest is None:
            raise ValueError(f"Manifest not found: {manifest_id}")
        if manifest.published_at is not None:
            raise ValueError(
                f"Cannot add objects to sealed manifest {manifest_id}"
            )
        if manifest.lane != lane:
            raise ValueError(
                f"Object lane {lane!r} does not match manifest lane "
                f"{manifest.lane!r} for manifest {manifest_id}"
            )
        if manifest.run_id != run_id:
            raise ValueError(
                f"Object run_id {run_id!r} does not match manifest run_id "
                f"{manifest.run_id!r} for manifest {manifest_id}"
            )
        # Validate object_type is publishable
        OwnershipPolicy.validate_publishable_type(object_type)

        obj = PublishedObject(
            object_id=_generate_id("obj"),
            object_type=object_type,
            manifest_id=manifest_id,
            lane=lane,
            publish_state="draft",
            contract_version=manifest.contract_version,
            source_ids=source_ids or [],
            run_id=run_id,
            valid_from=valid_from,
            valid_to=valid_to,
            payload=payload or {},
            lineage=lineage or {},
        )
        return await self._repo.create_published_object(obj)

    async def transition_object(
        self, object_id: str, target_state: str
    ) -> tuple[str, PublishedObject]:
        """Transition a published object to a new state.

        Validates the transition against the publish state machine
        before persisting.

        Args:
            object_id: Object to transition.
            target_state: Target publish state.

        Returns:
            Tuple of (previous_state, updated PublishedObject).

        Raises:
            ValueError: If the object doesn't exist or the transition
                is invalid.
        """
        obj = await self._repo.get_published_object(object_id)
        if obj is None:
            raise ValueError(f"Published object not found: {object_id}")
        previous_state = obj.publish_state
        # Block most transitions on objects in sealed manifests.
        # Only published → retracted is allowed post-seal.
        manifest = await self._repo.get_manifest(obj.manifest_id)
        if manifest is not None and manifest.published_at is not None:
            if target_state != "retracted":
                raise ValueError(
                    f"Cannot transition object {object_id}: manifest "
                    f"{obj.manifest_id} is sealed (only retracted allowed)"
                )
        _validate_publish_transition(previous_state, target_state)
        result = await self._repo.update_publish_state(object_id, target_state)
        if result is None:
            raise ValueError(
                f"Failed to transition object {object_id} to {target_state}"
            )
        logger.info(
            "Object %s transitioned: %s → %s",
            object_id,
            previous_state,
            target_state,
        )
        return previous_state, result

    async def get_object(self, object_id: str) -> PublishedObject | None:
        """Fetch a published object by ID."""
        return await self._repo.get_published_object(object_id)

    async def list_manifest_objects(
        self,
        manifest_id: str,
        *,
        publish_state: str | None = None,
    ) -> list[PublishedObject]:
        """List objects within a manifest, optionally filtered by state."""
        return await self._repo.list_objects_by_manifest(
            manifest_id, publish_state=publish_state
        )

    # -- Convenience: compute manifest checksum ----------------------------

    @staticmethod
    def compute_checksum(objects: list[PublishedObject]) -> str:
        """Compute a deterministic checksum over published object contents.

        Hashes a stable serialization of each object (sorted by ID),
        covering payload, lineage, validity windows, and all other
        content fields — not just identifiers.

        Note: this is the *manifest seal* checksum over object content.
        The *bundle export* checksum (in exporter.py) is computed over
        the full JSONL output including the manifest header. The two
        checksums serve different purposes and will differ by design.
        """
        serialized = []
        for obj in sorted(objects, key=lambda o: o.object_id):
            record = {
                "object_id": obj.object_id,
                "object_type": obj.object_type,
                "lane": obj.lane,
                "contract_version": obj.contract_version,
                "source_ids": sorted(obj.source_ids),
                "run_id": obj.run_id,
                "valid_from": obj.valid_from.isoformat() if obj.valid_from else None,
                "valid_to": obj.valid_to.isoformat() if obj.valid_to else None,
                "payload": obj.payload,
                "lineage": obj.lineage,
            }
            serialized.append(json.dumps(record, sort_keys=True))
        content = "\n".join(serialized)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
