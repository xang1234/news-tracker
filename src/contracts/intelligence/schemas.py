"""Core schemas for the intelligence contract surface.

These are the canonical shapes for manifest-keyed published objects.
All schemas here are producer-owned by news-tracker. Downstream consumers
(e.g., stock-screener) import these types but MUST NOT modify them.

Schema families (defined in downstream task q88.1.2):
    news_intel  — work-in-progress tables (claims, runs, intermediate state)
    intel_pub   — published/resolved rows (assertions, manifests, pointers)
    intel_export — bundle export artifacts (snapshots for offline consumers)

This module defines the shared building blocks that all three families use.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.contracts.intelligence.lanes import validate_lane
from src.contracts.intelligence.ownership import OwnershipPolicy
from src.contracts.intelligence.version import ContractRegistry, ContractVersion

# -- Shared validators -----------------------------------------------------


def _check_contract_version(v: str) -> str:
    """Validate a contract_version field value."""
    cv = ContractVersion.parse(v)
    if not ContractRegistry.is_supported(cv):
        raise ValueError(
            f"Contract version {v} is not supported. Minimum: {ContractRegistry.MINIMUM_SUPPORTED}"
        )
    return v


# -- Enums -----------------------------------------------------------------


class PublishState(str, Enum):
    """Lifecycle states for publishable objects.

    Objects start as DRAFT, move through REVIEW, and land at
    PUBLISHED or RETRACTED. Only PUBLISHED objects are visible
    to downstream consumers via manifest pointers.
    """

    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    RETRACTED = "retracted"


class ReviewDecision(str, Enum):
    """Possible outcomes of a review pass."""

    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"


# -- Shared field defaults -------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


# -- Building-block schemas ------------------------------------------------


class Lineage(BaseModel):
    """Provenance tracking for any intelligence object.

    Every publishable artifact must carry lineage so that it can be
    traced back to source material and reproduced point-in-time.

    Attributes:
        source_ids: IDs of the documents/filings that produced this object.
        lane: Which processing lane created it.
        run_id: The lane-run that produced it (for replay).
        contract_version: Which contract version governs the shape.
        created_at: When this lineage record was minted (knowledge time).
        valid_from: Bitemporal: when the fact became true in the real world.
        valid_to: Bitemporal: when the fact ceased to be true (None = current).
    """

    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source documents or filings",
    )
    lane: str = Field(
        ...,
        description="Canonical lane that produced this object",
    )
    run_id: str = Field(
        ...,
        description="Lane-run identifier for replay",
    )
    contract_version: str = Field(
        default_factory=lambda: str(ContractRegistry.CURRENT),
        description="Contract version governing this object's shape",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Knowledge time: when this record was created",
    )
    valid_from: datetime | None = Field(
        default=None,
        description="Bitemporal: when the fact became true",
    )
    valid_to: datetime | None = Field(
        default=None,
        description="Bitemporal: when the fact ceased to be true (None = current)",
    )

    @field_validator("lane")
    @classmethod
    def _validate_lane(cls, v: str) -> str:
        return validate_lane(v)

    @field_validator("contract_version")
    @classmethod
    def _validate_contract_version(cls, v: str) -> str:
        return _check_contract_version(v)

    @model_validator(mode="after")
    def _validate_validity_window(self) -> Lineage:
        """Reject impossible bitemporal intervals."""
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to < self.valid_from
        ):
            raise ValueError(
                f"valid_to ({self.valid_to}) must not be before valid_from ({self.valid_from})"
            )
        return self


class ManifestHeader(BaseModel):
    """Header for a versioned manifest.

    A manifest groups a set of published objects from a single lane-run
    into an addressable, versioned unit. Downstream consumers use the
    manifest_id as the serving pointer — they never read work-in-progress
    tables directly.

    Attributes:
        manifest_id: Unique identifier for this manifest.
        lane: Which lane produced the manifest contents.
        run_id: Lane-run that populated this manifest.
        contract_version: Contract version at publication time.
        published_at: When the manifest was sealed.
        object_count: Number of objects in this manifest.
        checksum: Optional integrity checksum over manifest contents.
        metadata: Extensible metadata (e.g., coverage tier, model version).
    """

    manifest_id: str = Field(
        ...,
        description="Unique manifest identifier",
    )
    lane: str = Field(
        ...,
        description="Lane that produced the manifest contents",
    )
    run_id: str = Field(
        ...,
        description="Lane-run that populated this manifest",
    )
    contract_version: str = Field(
        default_factory=lambda: str(ContractRegistry.CURRENT),
        description="Contract version at time of publication",
    )
    published_at: datetime = Field(
        default_factory=_utc_now,
        description="When the manifest was sealed and made available",
    )
    object_count: int = Field(
        default=0,
        ge=0,
        description="Number of published objects in this manifest",
    )
    checksum: str | None = Field(
        default=None,
        description="Integrity checksum (e.g., SHA-256 over sorted contents)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (coverage tier, model version, etc.)",
    )

    @field_validator("lane")
    @classmethod
    def _validate_lane(cls, v: str) -> str:
        return validate_lane(v)

    @field_validator("contract_version")
    @classmethod
    def _validate_contract_version(cls, v: str) -> str:
        return _check_contract_version(v)


class PublishedObjectRef(BaseModel):
    """Reference to a single published object within a manifest.

    This is the pointer that downstream consumers use to fetch
    a specific claim, assertion, or artifact from the published surface.

    Attributes:
        object_id: Unique ID of the published object.
        object_type: What kind of object (claim, assertion, etc.).
        manifest_id: Which manifest contains this object.
        lane: Lane that produced it.
        publish_state: Current lifecycle state.
        contract_version: Contract version governing the object's shape.
    """

    object_id: str = Field(
        ...,
        description="Unique identifier for the published object",
    )
    object_type: str = Field(
        ...,
        description="Type of the object (e.g., 'claim', 'assertion', 'signal')",
    )
    manifest_id: str = Field(
        ...,
        description="Manifest that contains this object",
    )
    lane: str = Field(
        ...,
        description="Lane that produced this object",
    )
    publish_state: PublishState = Field(
        default=PublishState.DRAFT,
        description="Current lifecycle state",
    )
    contract_version: str = Field(
        default_factory=lambda: str(ContractRegistry.CURRENT),
        description="Contract version governing this object",
    )

    @field_validator("object_type")
    @classmethod
    def _validate_object_type(cls, v: str) -> str:
        return OwnershipPolicy.validate_publishable_type(v)

    @field_validator("lane")
    @classmethod
    def _validate_lane(cls, v: str) -> str:
        return validate_lane(v)

    @field_validator("contract_version")
    @classmethod
    def _validate_contract_version(cls, v: str) -> str:
        return _check_contract_version(v)
