"""Tests for admin API models."""

import pytest
from pydantic import ValidationError

from src.api.admin_models import CreateSourceRequest, TriggerIngestionResponse


def test_create_source_request_trims_identifier() -> None:
    request = CreateSourceRequest(
        platform="twitter",
        identifier="  @SemiAnalysis  ",
    )

    assert request.identifier == "@SemiAnalysis"


def test_create_source_request_rejects_blank_identifier() -> None:
    with pytest.raises(
        ValidationError,
        match="identifier must contain at least one non-empty character",
    ):
        CreateSourceRequest(
            platform="twitter",
            identifier="   ",
        )


def test_trigger_ingestion_response_status_is_started_only() -> None:
    response = TriggerIngestionResponse(status="started", message="ok")
    assert response.status == "started"

    with pytest.raises(ValidationError):
        TriggerIngestionResponse(status="already_running", message="nope")
