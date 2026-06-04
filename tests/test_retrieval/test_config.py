"""Config + package-surface tests for the retrieval substrate."""

from __future__ import annotations

import pytest

from src.config.settings import Settings


def test_config_defaults_and_env_prefix() -> None:
    from src.retrieval import ClaimRetrievalConfig

    cfg = ClaimRetrievalConfig()
    assert cfg.default_limit == 10
    assert cfg.similarity_threshold == 0.3
    assert cfg.index_batch_size == 128
    assert ClaimRetrievalConfig.model_config["env_prefix"] == "CLAIM_RETRIEVAL_"


def test_config_validates_bounds() -> None:
    from pydantic import ValidationError

    from src.retrieval import ClaimRetrievalConfig

    with pytest.raises(ValidationError):
        ClaimRetrievalConfig(similarity_threshold=1.5)
    with pytest.raises(ValidationError):
        ClaimRetrievalConfig(default_limit=0)


def test_feature_flag_defaults_off() -> None:
    # _env_file=None isolates from a local .env (project convention for
    # Settings tests — see test_settings.py / test_cors_settings.py).
    assert Settings(_env_file=None).claim_retrieval_enabled is False


def test_public_surface_is_importable() -> None:
    import src.retrieval as r

    for name in r.__all__:
        assert hasattr(r, name), name
