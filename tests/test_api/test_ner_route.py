"""Tests for NER REST API endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_ner_service
from src.ner.schemas import FinancialEntity
from src.ner.service import NERService


def _mock_entities() -> list[FinancialEntity]:
    return [
        FinancialEntity(
            text="Nvidia",
            type="COMPANY",
            normalized="NVIDIA",
            start=0,
            end=6,
            confidence=0.95,
            metadata={"ticker": "NVDA"},
        ),
        FinancialEntity(
            text="HBM3E",
            type="TECHNOLOGY",
            normalized="HBM3E",
            start=17,
            end=22,
            confidence=0.85,
            metadata={},
        ),
    ]


@pytest.fixture
def mock_ner_service():
    service = MagicMock(spec=NERService)
    return service


@pytest.fixture
def client_ner_enabled(mock_ner_service):
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_ner_service] = lambda: mock_ner_service

    with patch("src.api.routes.ner._get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(ner_enabled=True)
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


@pytest.fixture
def client_ner_disabled(mock_ner_service):
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_ner_service] = lambda: mock_ner_service

    with patch("src.api.routes.ner._get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(ner_enabled=False)
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


class TestNEREndpoint:
    def test_success(self, client_ner_enabled, mock_ner_service):
        mock_ner_service.extract_batch.return_value = [_mock_entities()]

        resp = client_ner_enabled.post("/ner", json={"texts": ["Nvidia announced HBM3E support"]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["results"]) == 1
        assert len(data["results"][0]["entities"]) == 2
        assert data["results"][0]["entities"][0]["type"] == "COMPANY"
        assert data["results"][0]["entities"][0]["normalized"] == "NVIDIA"
        assert data["results"][0]["text_length"] == 30
        assert "latency_ms" in data

    def test_multiple_texts(self, client_ner_enabled, mock_ner_service):
        mock_ner_service.extract_batch.return_value = [_mock_entities(), []]

        resp = client_ner_enabled.post("/ner", json={"texts": ["Nvidia text", "Empty text"]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"][0]["entities"]) == 2
        assert len(data["results"][1]["entities"]) == 0

    def test_empty_texts_validation(self, client_ner_enabled, mock_ner_service):
        resp = client_ner_enabled.post("/ner", json={"texts": []})
        assert resp.status_code == 422

    def test_disabled_returns_503(self, client_ner_disabled, mock_ner_service):
        resp = client_ner_disabled.post("/ner", json={"texts": ["test"]})
        assert resp.status_code == 503
        assert "disabled" in resp.json()["detail"].lower()

    def test_server_error(self, client_ner_enabled, mock_ner_service):
        mock_ner_service.extract_batch.side_effect = RuntimeError("Model load failed")

        resp = client_ner_enabled.post("/ner", json={"texts": ["test"]})
        assert resp.status_code == 500
        assert "NER extraction failed" in resp.json()["detail"]
