"""Tests for environment-backed application settings."""

from __future__ import annotations

from src.config.settings import Settings


def test_rss_ingestion_feature_flag_defaults_off() -> None:
    settings = Settings(_env_file=None)

    assert settings.rss_enabled is False
