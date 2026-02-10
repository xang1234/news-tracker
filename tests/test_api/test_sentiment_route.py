"""Tests for /sentiment endpoint."""

import pytest


class TestSentimentRoute:
    """Test the sentiment analysis endpoint."""

    def test_sentiment_happy_path(self, client, mock_sentiment_service):
        """POST /sentiment with valid input returns sentiment results."""
        resp = client.post("/sentiment", json={
            "texts": ["NVIDIA stock surges on AI demand"],
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["results"][0]["label"] == "positive"
        assert body["results"][0]["confidence"] > 0
        assert body["latency_ms"] >= 0

    def test_sentiment_empty_texts_rejected(self, client):
        """POST /sentiment with empty texts returns 422."""
        resp = client.post("/sentiment", json={"texts": []})
        assert resp.status_code == 422

    def test_sentiment_too_many_texts_rejected(self, client):
        """POST /sentiment with >32 texts returns 422."""
        resp = client.post("/sentiment", json={"texts": ["x"] * 33})
        assert resp.status_code == 422

    def test_sentiment_error_sanitized(self, client, mock_sentiment_service):
        """POST /sentiment error response doesn't leak internal details."""
        mock_sentiment_service.analyze_batch.side_effect = RuntimeError(
            "Model loading failed at /models/finbert"
        )

        resp = client.post("/sentiment", json={"texts": ["test"]})
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Sentiment analysis failed"
        assert "/models/" not in body["detail"]
