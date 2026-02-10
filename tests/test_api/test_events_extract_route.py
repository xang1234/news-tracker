"""Tests for Events extract REST API endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_pattern_extractor
from src.event_extraction.patterns import PatternExtractor
from src.event_extraction.schemas import EventRecord


def _mock_events() -> list[EventRecord]:
    return [
        EventRecord(
            doc_id="playground_0",
            event_type="capacity_expansion",
            actor="TSMC",
            action="is expanding",
            object="fab capacity",
            time_ref="Q3 2026",
            quantity="$20 billion",
            tickers=["TSM"],
            confidence=0.85,
            span_start=0,
            span_end=50,
            extractor_version="v1",
        ),
    ]


@pytest.fixture
def mock_extractor():
    extractor = MagicMock(spec=PatternExtractor)
    return extractor


@pytest.fixture
def client_events_enabled(mock_extractor):
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_pattern_extractor] = lambda: mock_extractor

    with patch("src.api.routes.events_extract._get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(events_enabled=True)
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


@pytest.fixture
def client_events_disabled(mock_extractor):
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_pattern_extractor] = lambda: mock_extractor

    with patch("src.api.routes.events_extract._get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(events_enabled=False)
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


class TestEventsExtractEndpoint:
    def test_success(self, client_events_enabled, mock_extractor):
        mock_extractor.extract.return_value = _mock_events()

        resp = client_events_enabled.post(
            "/events/extract",
            json={"text": "TSMC is expanding fab capacity by $20 billion in Q3 2026"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["events"][0]["event_type"] == "capacity_expansion"
        assert data["events"][0]["actor"] == "TSMC"
        assert data["events"][0]["action"] == "is expanding"
        assert data["events"][0]["object"] == "fab capacity"
        assert data["events"][0]["tickers"] == ["TSM"]
        assert data["events"][0]["confidence"] == 0.85
        assert "latency_ms" in data

    def test_with_tickers(self, client_events_enabled, mock_extractor):
        mock_extractor.extract.return_value = _mock_events()

        resp = client_events_enabled.post(
            "/events/extract",
            json={"text": "TSMC expansion news", "tickers": ["TSM", "INTC"]},
        )

        assert resp.status_code == 200
        # Verify the extractor received a doc with tickers
        call_args = mock_extractor.extract.call_args[0][0]
        assert call_args.tickers_mentioned == ["TSM", "INTC"]

    def test_empty_text_validation(self, client_events_enabled, mock_extractor):
        resp = client_events_enabled.post("/events/extract", json={"text": ""})
        assert resp.status_code == 422

    def test_no_events_found(self, client_events_enabled, mock_extractor):
        mock_extractor.extract.return_value = []

        resp = client_events_enabled.post(
            "/events/extract", json={"text": "The weather is nice today."}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["events"] == []

    def test_disabled_returns_503(self, client_events_disabled, mock_extractor):
        resp = client_events_disabled.post("/events/extract", json={"text": "test"})
        assert resp.status_code == 503
        assert "disabled" in resp.json()["detail"].lower()

    def test_server_error(self, client_events_enabled, mock_extractor):
        mock_extractor.extract.side_effect = RuntimeError("Pattern compile failed")

        resp = client_events_enabled.post("/events/extract", json={"text": "test"})
        assert resp.status_code == 500
        assert "Event extraction failed" in resp.json()["detail"]
