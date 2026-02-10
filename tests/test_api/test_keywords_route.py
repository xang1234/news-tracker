"""Tests for Keywords REST API endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_keywords_service
from src.keywords.schemas import ExtractedKeyword
from src.keywords.service import KeywordsService


def _mock_keywords() -> list[ExtractedKeyword]:
    return [
        ExtractedKeyword(text="semiconductor chip", score=0.125, rank=1, lemma="semiconductor chip", count=3),
        ExtractedKeyword(text="HBM3E memory", score=0.108, rank=2, lemma="hbm3e memory", count=2),
        ExtractedKeyword(text="Nvidia", score=0.095, rank=3, lemma="nvidia", count=1),
    ]


@pytest.fixture
def mock_keywords_service():
    service = MagicMock(spec=KeywordsService)
    return service


@pytest.fixture
def client_kw_enabled(mock_keywords_service):
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_keywords_service] = lambda: mock_keywords_service

    with patch("src.api.routes.keywords_route.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(keywords_enabled=True)
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


@pytest.fixture
def client_kw_disabled(mock_keywords_service):
    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_keywords_service] = lambda: mock_keywords_service

    with patch("src.api.routes.keywords_route.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(keywords_enabled=False)
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


class TestKeywordsEndpoint:
    def test_success(self, client_kw_enabled, mock_keywords_service):
        mock_keywords_service.extract_batch.return_value = [_mock_keywords()]

        resp = client_kw_enabled.post("/keywords", json={"texts": ["Nvidia semiconductor chip news"]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["results"][0]["keywords"]) == 3
        assert data["results"][0]["keywords"][0]["text"] == "semiconductor chip"
        assert data["results"][0]["keywords"][0]["rank"] == 1
        assert data["results"][0]["text_length"] == len("Nvidia semiconductor chip news")
        assert "latency_ms" in data

    def test_top_n_truncation(self, client_kw_enabled, mock_keywords_service):
        mock_keywords_service.extract_batch.return_value = [_mock_keywords()]

        resp = client_kw_enabled.post("/keywords", json={"texts": ["test"], "top_n": 2})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"][0]["keywords"]) == 2

    def test_multiple_texts(self, client_kw_enabled, mock_keywords_service):
        mock_keywords_service.extract_batch.return_value = [_mock_keywords(), []]

        resp = client_kw_enabled.post("/keywords", json={"texts": ["text one", "text two"]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_empty_texts_validation(self, client_kw_enabled, mock_keywords_service):
        resp = client_kw_enabled.post("/keywords", json={"texts": []})
        assert resp.status_code == 422

    def test_disabled_returns_503(self, client_kw_disabled, mock_keywords_service):
        resp = client_kw_disabled.post("/keywords", json={"texts": ["test"]})
        assert resp.status_code == 503
        assert "disabled" in resp.json()["detail"].lower()

    def test_server_error(self, client_kw_enabled, mock_keywords_service):
        mock_keywords_service.extract_batch.side_effect = RuntimeError("spaCy not found")

        resp = client_kw_enabled.post("/keywords", json={"texts": ["test"]})
        assert resp.status_code == 500
        assert "Keywords extraction failed" in resp.json()["detail"]
