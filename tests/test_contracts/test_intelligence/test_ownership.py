"""Tests for intelligence contract ownership and compatibility."""

import pytest

from src.contracts.intelligence.ownership import (
    OWNER_REPO,
    OwnershipPolicy,
    check_compatibility,
)
from src.contracts.intelligence.version import ContractRegistry, ContractVersion


class TestCheckCompatibility:
    """check_compatibility() function."""

    def test_exact_match(self) -> None:
        result = check_compatibility(ContractRegistry.CURRENT)
        assert result.compatible is True
        assert "Exact match" in result.message

    def test_same_major_compatible(self) -> None:
        current = ContractRegistry.CURRENT
        older = ContractVersion(current.major, 0, 0)
        if older >= ContractRegistry.MINIMUM_SUPPORTED:
            result = check_compatibility(older)
            assert result.compatible is True

    def test_different_major_incompatible(self) -> None:
        future = ContractVersion(ContractRegistry.CURRENT.major + 1, 0, 0)
        result = check_compatibility(future)
        assert result.compatible is False
        assert "Major version mismatch" in result.message

    def test_string_input(self) -> None:
        result = check_compatibility(str(ContractRegistry.CURRENT))
        assert result.compatible is True

    def test_result_fields_populated(self) -> None:
        result = check_compatibility(ContractRegistry.CURRENT)
        assert result.current == ContractRegistry.CURRENT
        assert result.checked == ContractRegistry.CURRENT
        assert len(result.message) > 0


class TestOwnershipPolicy:
    """OwnershipPolicy invariants."""

    def test_owner_is_news_tracker(self) -> None:
        assert OwnershipPolicy.OWNER == OWNER_REPO
        assert OwnershipPolicy.OWNER == "news-tracker"

    def test_contract_paths_under_src(self) -> None:
        for path in OwnershipPolicy.CONTRACT_PATHS:
            assert path.startswith("src/contracts/")

    def test_publishable_types_non_empty(self) -> None:
        assert len(OwnershipPolicy.PUBLISHABLE_OBJECT_TYPES) > 0

    def test_claim_is_publishable(self) -> None:
        assert OwnershipPolicy.is_publishable_type("claim")

    def test_assertion_is_publishable(self) -> None:
        assert OwnershipPolicy.is_publishable_type("assertion")

    def test_unknown_type_not_publishable(self) -> None:
        assert not OwnershipPolicy.is_publishable_type("random_thing")

    def test_validate_publishable_type_raises(self) -> None:
        with pytest.raises(ValueError, match="not publishable"):
            OwnershipPolicy.validate_publishable_type("random_thing")

    def test_validate_publishable_type_returns(self) -> None:
        assert OwnershipPolicy.validate_publishable_type("claim") == "claim"

    def test_all_publishable_types_are_lowercase_identifiers(self) -> None:
        for obj_type in OwnershipPolicy.PUBLISHABLE_OBJECT_TYPES:
            assert obj_type == obj_type.lower()
            assert obj_type.replace("_", "").isalnum()
