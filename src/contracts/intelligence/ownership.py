"""Ownership rules and compatibility checking for intelligence contracts.

Ownership invariant:
    news-tracker is the sole producer and owner of intelligence contracts.
    Downstream consumers (e.g., stock-screener) may import contract types
    and read published manifests, but MUST NOT:
      - Modify contract schemas
      - Publish objects into news-tracker's manifest space
      - Define ad-hoc lane names or object types

Compatibility rules:
    - Same MAJOR version = wire-compatible (consumers can read without changes)
    - MINOR bump = additive only (new optional fields, new lanes)
    - PATCH bump = no schema changes (docs, validation, bug fixes)
    - Deprecated versions remain readable until removed from the registry

This module provides utilities for checking compatibility and enforcing
ownership boundaries at import time and in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.contracts.intelligence.version import ContractRegistry, ContractVersion

# -- Ownership constants ---------------------------------------------------

OWNER_REPO = "news-tracker"
"""The repository that owns and publishes intelligence contracts."""

CONSUMER_REPOS: frozenset[str] = frozenset({"stock-screener"})
"""Known downstream consumers. Used for documentation and test fixtures."""


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """Result of a contract compatibility check.

    Attributes:
        compatible: Whether the versions are wire-compatible.
        current: The current contract version.
        checked: The version being checked.
        message: Human-readable explanation.
    """

    compatible: bool
    current: ContractVersion
    checked: ContractVersion
    message: str


def check_compatibility(version: ContractVersion | str) -> CompatibilityResult:
    """Check whether a contract version is compatible with the current contract.

    This is the primary entry point for compatibility validation. Call it
    when deserializing objects from external sources or when a consumer
    reports its expected contract version.

    Args:
        version: The version to check (string or ContractVersion).

    Returns:
        CompatibilityResult with compatibility status and explanation.
    """
    if isinstance(version, str):
        version = ContractVersion.parse(version)

    current = ContractRegistry.CURRENT

    if version == current:
        return CompatibilityResult(
            compatible=True,
            current=current,
            checked=version,
            message=f"Exact match with current contract {current}",
        )

    if not version.is_compatible_with(current):
        return CompatibilityResult(
            compatible=False,
            current=current,
            checked=version,
            message=(
                f"Major version mismatch: {version} is not compatible "
                f"with current {current}. Consumer must upgrade."
            ),
        )

    if not ContractRegistry.is_supported(version):
        return CompatibilityResult(
            compatible=False,
            current=current,
            checked=version,
            message=(
                f"Version {version} is below the minimum supported "
                f"version {ContractRegistry.MINIMUM_SUPPORTED}."
            ),
        )

    deprecated = ContractRegistry.is_deprecated(version)
    if deprecated:
        replacement = ContractRegistry.DEPRECATION_SCHEDULE[version]
        return CompatibilityResult(
            compatible=True,
            current=current,
            checked=version,
            message=(
                f"Version {version} is deprecated. "
                f"Upgrade to {replacement} before support is removed."
            ),
        )

    return CompatibilityResult(
        compatible=True,
        current=current,
        checked=version,
        message=(f"Version {version} is compatible with current {current} (same major version)."),
    )


class OwnershipPolicy:
    """Codified ownership rules for the intelligence contract surface.

    These rules are enforced by contract tests and serve as living
    documentation of the producer/consumer boundary.
    """

    OWNER: ClassVar[str] = OWNER_REPO

    # Paths within the owner repo that constitute the contract surface.
    # Contract tests verify that only these paths define publishable types.
    CONTRACT_PATHS: ClassVar[tuple[str, ...]] = ("src/contracts/intelligence/",)

    # Object types that may appear in published manifests.
    # Lane-specific modules may define working types, but only these
    # types are allowed in the intel_pub / intel_export families.
    PUBLISHABLE_OBJECT_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "claim",
            "assertion",
            "signal",
            "narrative_run",
            "filing_fact",
            "structural_edge",
            "basket",
            "backtest_result",
            "evaluation_score",
        }
    )

    @classmethod
    def is_publishable_type(cls, object_type: str) -> bool:
        """Check whether an object type is allowed in published manifests."""
        return object_type in cls.PUBLISHABLE_OBJECT_TYPES

    @classmethod
    def validate_publishable_type(cls, object_type: str) -> str:
        """Validate and return an object type, raising on unknown types.

        Raises:
            ValueError: If the object type is not in the publishable set.
        """
        if not cls.is_publishable_type(object_type):
            raise ValueError(
                f"Object type {object_type!r} is not publishable. "
                f"Allowed types: {sorted(cls.PUBLISHABLE_OBJECT_TYPES)}"
            )
        return object_type
