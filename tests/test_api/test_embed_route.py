"""Tests for /embed endpoint."""

import pytest


class TestEmbedRoute:
    """Test the embedding endpoint."""

    def test_embed_happy_path(self, client, mock_embedding_service):
        """POST /embed with valid input returns embeddings."""
        mock_embedding_service.embed_batch.return_value = [[0.1] * 768]

        resp = client.post("/embed", json={
            "texts": ["NVIDIA reports record revenue"],
            "model": "auto",
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["model_used"] in ("finbert", "minilm")
        assert body["dimensions"] in (384, 768)
        assert len(body["embeddings"]) == 1
        assert body["latency_ms"] >= 0

    def test_embed_empty_texts_rejected(self, client):
        """POST /embed with empty texts list returns 422."""
        resp = client.post("/embed", json={"texts": []})
        assert resp.status_code == 422

    def test_embed_too_many_texts_rejected(self, client):
        """POST /embed with >64 texts returns 422."""
        resp = client.post("/embed", json={"texts": ["x"] * 65})
        assert resp.status_code == 422

    def test_embed_text_too_long_rejected(self, client):
        """POST /embed with text >10000 chars returns 422."""
        resp = client.post("/embed", json={"texts": ["x" * 10_001]})
        assert resp.status_code == 422

    def test_embed_error_sanitized(self, client, mock_embedding_service):
        """POST /embed error response doesn't leak internal details."""
        mock_embedding_service.embed_batch.side_effect = RuntimeError(
            "CUDA out of memory at /path/to/internal"
        )

        resp = client.post("/embed", json={"texts": ["test"]})
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Embedding generation failed"
        assert "CUDA" not in body["detail"]
        assert "/path/" not in body["detail"]
