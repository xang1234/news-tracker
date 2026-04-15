"""Tests for ingestion service configuration behavior."""

from unittest.mock import patch

import pytest

from src.services.ingestion_service import IngestionConfigurationError, IngestionService


class TestIngestionServiceConfiguration:
    """Validate startup configuration behavior for ingestion."""

    def test_raises_when_no_real_adapters_and_mock_disabled(self) -> None:
        with (
            patch.object(IngestionService, "_create_adapters", return_value={}),
            pytest.raises(
                IngestionConfigurationError,
                match="No ingestion sources are configured",
            ),
        ):
            IngestionService(use_mock=False)

    def test_mock_mode_still_uses_mock_adapters(self) -> None:
        service = IngestionService(use_mock=True)
        assert service._adapters
