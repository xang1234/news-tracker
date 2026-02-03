"""
FastAPI embedding service.

Provides REST API for embedding generation with:
- POST /embed - Batch embedding with model selection
- GET /health - Service health check
"""

from src.api.app import create_app

__all__ = ["create_app"]
