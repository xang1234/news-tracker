"""Tests for intelligence contract versioning."""

import pytest

from src.contracts.intelligence.version import ContractRegistry, ContractVersion


class TestContractVersion:
    """ContractVersion semantics and parsing."""

    def test_create_valid(self) -> None:
        v = ContractVersion(1, 2, 3)
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_str_representation(self) -> None:
        assert str(ContractVersion(0, 1, 0)) == "0.1.0"
        assert str(ContractVersion(2, 10, 3)) == "2.10.3"

    def test_parse_valid(self) -> None:
        v = ContractVersion.parse("1.2.3")
        assert v == ContractVersion(1, 2, 3)

    def test_parse_zero_version(self) -> None:
        v = ContractVersion.parse("0.0.0")
        assert v == ContractVersion(0, 0, 0)

    @pytest.mark.parametrize(
        "bad_input",
        ["1.2", "1.2.3.4", "v1.2.3", "abc", "", "1.2.3-beta", "01.2.3"],
    )
    def test_parse_invalid(self, bad_input: str) -> None:
        with pytest.raises(ValueError, match="Invalid contract version"):
            ContractVersion.parse(bad_input)

    def test_negative_component_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            ContractVersion(-1, 0, 0)

    def test_non_int_component_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            ContractVersion(1, "2", 0)  # type: ignore[arg-type]

    def test_ordering(self) -> None:
        v010 = ContractVersion(0, 1, 0)
        v020 = ContractVersion(0, 2, 0)
        v100 = ContractVersion(1, 0, 0)
        assert v010 < v020 < v100

    def test_equality(self) -> None:
        assert ContractVersion(1, 0, 0) == ContractVersion(1, 0, 0)
        assert ContractVersion(1, 0, 0) != ContractVersion(1, 0, 1)

    def test_hash_consistency(self) -> None:
        v1 = ContractVersion(1, 2, 3)
        v2 = ContractVersion(1, 2, 3)
        assert hash(v1) == hash(v2)
        assert len({v1, v2}) == 1

    def test_immutable(self) -> None:
        v = ContractVersion(1, 0, 0)
        with pytest.raises(AttributeError):
            v.major = 2  # type: ignore[misc]

    def test_compatible_same_major(self) -> None:
        v1 = ContractVersion(1, 0, 0)
        v2 = ContractVersion(1, 5, 3)
        assert v1.is_compatible_with(v2)

    def test_incompatible_different_major(self) -> None:
        v1 = ContractVersion(1, 0, 0)
        v2 = ContractVersion(2, 0, 0)
        assert not v1.is_compatible_with(v2)

    def test_roundtrip_parse_str(self) -> None:
        original = ContractVersion(3, 14, 159)
        assert ContractVersion.parse(str(original)) == original


class TestContractRegistry:
    """Registry invariants and support checks."""

    def test_current_is_in_all_versions(self) -> None:
        assert ContractRegistry.CURRENT in ContractRegistry.ALL_VERSIONS

    def test_minimum_supported_is_in_all_versions(self) -> None:
        assert ContractRegistry.MINIMUM_SUPPORTED in ContractRegistry.ALL_VERSIONS

    def test_minimum_not_above_current(self) -> None:
        assert ContractRegistry.MINIMUM_SUPPORTED <= ContractRegistry.CURRENT

    def test_all_versions_sorted(self) -> None:
        versions = ContractRegistry.ALL_VERSIONS
        assert versions == tuple(sorted(versions))

    def test_current_is_supported(self) -> None:
        assert ContractRegistry.is_supported(ContractRegistry.CURRENT)

    def test_future_major_not_supported(self) -> None:
        future = ContractVersion(ContractRegistry.CURRENT.major + 1, 0, 0)
        assert not ContractRegistry.is_supported(future)

    def test_deprecated_versions_have_replacements(self) -> None:
        for deprecated, replacement in ContractRegistry.DEPRECATION_SCHEDULE.items():
            assert replacement > deprecated
            assert replacement in ContractRegistry.ALL_VERSIONS

    def test_current_not_deprecated(self) -> None:
        assert not ContractRegistry.is_deprecated(ContractRegistry.CURRENT)
