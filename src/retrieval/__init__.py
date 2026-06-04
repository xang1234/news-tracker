"""Claim/assertion retrieval substrate: a semantic index over the structured
fact layer (resolved assertions + evidence claims) for grounded RAG.

The briefing generator and Q&A endpoint consume :class:`ClaimRetrievalService`
to fetch the top-K verified claims for a theme or free-text query, each with
confidence, lane, and lineage.
"""

from __future__ import annotations

from src.retrieval.config import ClaimRetrievalConfig
from src.retrieval.repository import ClaimRetrievalRepository
from src.retrieval.schemas import ClaimRetrievalFilter, RetrievedClaim
from src.retrieval.service import RETRIEVAL_MODEL, ClaimRetrievalService
from src.retrieval.text import claim_embedding_text

__all__ = [
    "RETRIEVAL_MODEL",
    "ClaimRetrievalConfig",
    "ClaimRetrievalFilter",
    "ClaimRetrievalRepository",
    "ClaimRetrievalService",
    "RetrievedClaim",
    "claim_embedding_text",
]
