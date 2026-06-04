"""Tests for environment-backed application settings."""

from __future__ import annotations

from src.config.settings import Settings


def test_rss_ingestion_feature_flag_defaults_off() -> None:
    settings = Settings(_env_file=None)

    assert settings.rss_enabled is False


def test_semantic_requires_claim_reconciliation() -> None:
    import pytest

    with pytest.raises(ValueError, match="claim_reconciliation_enabled"):
        Settings(
            _env_file=None,
            semantic_contradiction_enabled=True,
            claim_reconciliation_enabled=False,
        )


def test_claim_reconciliation_requires_narrative_extraction() -> None:
    import pytest

    with pytest.raises(ValueError, match="narrative_claim_extraction_enabled"):
        Settings(
            _env_file=None,
            claim_reconciliation_enabled=True,
            narrative_claim_extraction_enabled=False,
        )


def test_valid_reconciliation_flag_chain_passes() -> None:
    settings = Settings(
        _env_file=None,
        narrative_claim_extraction_enabled=True,
        claim_reconciliation_enabled=True,
        semantic_contradiction_enabled=True,
    )
    assert settings.semantic_contradiction_enabled is True
