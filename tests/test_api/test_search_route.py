"""Tests for /search/similar endpoint."""

import pytest
from unittest.mock import MagicMock


class TestSearchRoute:
    """Test the semantic search endpoint."""

    def test_search_happy_path(self, client, mock_vector_store_manager):
        """POST /search/similar with valid query returns results."""
        mock_result = MagicMock()
        mock_result.document_id = "doc_123"
        mock_result.score = 0.95
        mock_result.metadata = {
            "platform": "twitter",
            "title": "NVIDIA AI chips",
            "content_preview": "Great quarter for NVIDIA...",
            "url": "https://example.com",
            "author_name": "analyst1",
            "author_verified": True,
            "tickers": ["NVDA"],
            "authority_score": 0.8,
            "timestamp": "2026-02-01T10:00:00Z",
        }
        mock_vector_store_manager.query.return_value = [mock_result]

        resp = client.post("/search/similar", json={
            "query": "NVIDIA GPU demand",
            "limit": 5,
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["results"][0]["document_id"] == "doc_123"
        assert body["results"][0]["score"] == 0.95
        assert body["latency_ms"] >= 0

    def test_search_empty_query_rejected(self, client):
        """POST /search/similar with empty query returns 422."""
        resp = client.post("/search/similar", json={"query": ""})
        assert resp.status_code == 422

    def test_search_with_filters(self, client, mock_vector_store_manager):
        """POST /search/similar with filters passes them through."""
        mock_vector_store_manager.query.return_value = []

        resp = client.post("/search/similar", json={
            "query": "semiconductor supply",
            "platforms": ["twitter"],
            "tickers": ["NVDA"],
            "min_authority_score": 0.5,
        })

        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_search_error_sanitized(self, client, mock_vector_store_manager):
        """POST /search/similar error response doesn't leak internals."""
        mock_vector_store_manager.query.side_effect = RuntimeError(
            "connection refused to pgvector at 10.0.0.5:5432"
        )

        resp = client.post("/search/similar", json={"query": "test query"})
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Search failed"
        assert "10.0.0.5" not in body["detail"]
