"""Tests for safe credentialed CORS defaults and validation."""

import pytest

from src.config.settings import Settings


def test_default_cors_origins_are_explicit_local_dev_hosts() -> None:
    settings = Settings(_env_file=None)

    assert settings.cors_allow_credentials is True
    assert settings.cors_origin_list == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]


def test_wildcard_cors_with_credentials_is_rejected() -> None:
    with pytest.raises(ValueError, match="wildcard CORS origins"):
        Settings(_env_file=None, cors_origins="*", cors_allow_credentials=True)
