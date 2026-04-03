"""Contract versioning for the intelligence layer.

Provides semantic versioning for the intelligence contract surface.
Every published object (claim, assertion, manifest, bundle) embeds a
contract_version that traces back to a specific version in this registry.

Version semantics:
    MAJOR — breaking change to published object shapes or manifest keys.
    MINOR — additive fields, new lanes, new optional capabilities.
    PATCH — documentation, validation tightening, bug fixes.

Compatibility rule:
    Consumers built against contract N.x.y MUST be able to read any
    object published under N.*.* (same major). A major bump signals
    that consumers must update.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import total_ordering
from typing import ClassVar

_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)


@total_ordering
@dataclass(frozen=True, slots=True)
class ContractVersion:
    """Immutable semantic version identifier for an intelligence contract.

    Attributes:
        major: Incremented on breaking schema changes.
        minor: Incremented on backwards-compatible additions.
        patch: Incremented on non-functional fixes.
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for name in ("major", "minor", "patch"):
            val = getattr(self, name)
            if not isinstance(val, int) or val < 0:
                raise ValueError(
                    f"ContractVersion.{name} must be a non-negative int, got {val!r}"
                )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContractVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (
            other.major,
            other.minor,
            other.patch,
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ContractVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    def is_compatible_with(self, other: ContractVersion) -> bool:
        """Check if this version is compatible with another (same major)."""
        return self.major == other.major

    @classmethod
    def parse(cls, version_string: str) -> ContractVersion:
        """Parse a 'MAJOR.MINOR.PATCH' string into a ContractVersion.

        Raises:
            ValueError: If the string is not valid semver.
        """
        match = _SEMVER_RE.match(version_string)
        if not match:
            raise ValueError(
                f"Invalid contract version string: {version_string!r}. "
                "Expected format: MAJOR.MINOR.PATCH"
            )
        return cls(
            major=int(match["major"]),
            minor=int(match["minor"]),
            patch=int(match["patch"]),
        )


class ContractRegistry:
    """Registry of known intelligence contract versions.

    The registry is the single source of truth for which contract versions
    exist. Downstream tasks (schema definitions, migration generators,
    compatibility tests) import from here rather than hardcoding versions.

    Attributes:
        CURRENT: The active contract version that new publications use.
        MINIMUM_SUPPORTED: The oldest version consumers are expected to read.
        ALL_VERSIONS: Ordered list of every released contract version.
        DEPRECATION_SCHEDULE: Versions slated for removal, mapped to the
            version that replaces them.
    """

    # -- Current contract version ------------------------------------------
    # Bump MINOR when adding optional fields or new lanes.
    # Bump MAJOR when published object shapes change incompatibly.
    CURRENT: ClassVar[ContractVersion] = ContractVersion(0, 1, 0)

    # -- Support floor -----------------------------------------------------
    MINIMUM_SUPPORTED: ClassVar[ContractVersion] = ContractVersion(0, 1, 0)

    # -- Full version history (oldest → newest) ----------------------------
    ALL_VERSIONS: ClassVar[tuple[ContractVersion, ...]] = (
        ContractVersion(0, 1, 0),
    )

    # -- Deprecation schedule ----------------------------------------------
    # Maps a deprecated version to its replacement.
    # Empty until a version is actually superseded.
    DEPRECATION_SCHEDULE: ClassVar[dict[ContractVersion, ContractVersion]] = {}

    @classmethod
    def is_supported(cls, version: ContractVersion) -> bool:
        """Check whether a contract version is still supported.

        A version is supported only if it was actually released
        (present in ALL_VERSIONS) and is at or above the support floor.
        """
        return (
            version in cls.ALL_VERSIONS
            and version >= cls.MINIMUM_SUPPORTED
        )

    @classmethod
    def is_deprecated(cls, version: ContractVersion) -> bool:
        """Check whether a contract version is deprecated."""
        return version in cls.DEPRECATION_SCHEDULE
