"""Claim retrieval service: semantic search + embedding backfill.

Orchestrates the substrate that the briefing generator and Q&A endpoint
consume: it embeds a free-text query (or theme query) with MiniLM, ranks
claims by cosine similarity over the structured layer, and returns each
match with full lineage (via :class:`RetrievedClaim`). The indexing methods
populate the embedding column from the composed claim text.

Fixed on MiniLM (384-dim): claims are short factual sentences, MiniLM is the
sentence-similarity model, and the ``embedding`` column is ``vector(384)``.
Query and claims share the model, so their vectors are comparable.
"""

from __future__ import annotations

import logging

from src.claims.repository import ClaimRepository
from src.embedding.service import EmbeddingService, ModelType
from src.retrieval.config import ClaimRetrievalConfig
from src.retrieval.repository import ClaimRetrievalRepository
from src.retrieval.schemas import ClaimRetrievalFilter, RetrievedClaim
from src.retrieval.text import claim_embedding_text
from src.storage.database import Database

logger = logging.getLogger(__name__)

#: The single embedding model backing the substrate (must match the column dim).
RETRIEVAL_MODEL = ModelType.MINILM


def _resolve_limit(limit: int | None, default: int) -> int:
    """Resolve a caller-supplied limit, rejecting non-positive values.

    ``None`` means "use the configured default"; ``0``/negative are invalid
    (they'd reach SQL as a bad ``LIMIT``) and raise rather than silently
    falling back.
    """
    if limit is None:
        return default
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    return limit


class ClaimRetrievalService:
    """Semantic retrieval over verified structured claims."""

    def __init__(
        self,
        database: Database,
        embedding_service: EmbeddingService,
        *,
        claim_repository: ClaimRepository | None = None,
        config: ClaimRetrievalConfig | None = None,
    ) -> None:
        self._config = config or ClaimRetrievalConfig()
        self._repo = ClaimRetrievalRepository(database)
        self._claims = claim_repository or ClaimRepository(database)
        self._embedding = embedding_service

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        filters: ClaimRetrievalFilter | None = None,
    ) -> list[RetrievedClaim]:
        """Return the top-K claims most relevant to ``query``.

        ``filters`` defaults to active claims only (see
        :class:`ClaimRetrievalFilter`); pass one to scope by lane, theme,
        subject, or confidence.
        """
        limit = _resolve_limit(limit, self._config.default_limit)
        if filters is None:
            filters = ClaimRetrievalFilter()
        query_embedding = await self._embedding.embed(query, RETRIEVAL_MODEL)
        pairs = await self._repo.search(
            query_embedding,
            limit=limit,
            threshold=self._config.similarity_threshold,
            filters=filters,
        )
        return [RetrievedClaim(claim=claim, score=score) for claim, score in pairs]

    async def index_pending(self, limit: int | None = None) -> int:
        """Embed and store vectors for un-embedded active claims.

        Returns the number of claims successfully indexed. Idempotent: a
        re-run only picks up claims whose embedding is still NULL.
        """
        limit = _resolve_limit(limit, self._config.index_batch_size)
        claims = await self._repo.list_unembedded(limit)
        if not claims:
            return 0
        texts = [claim_embedding_text(claim) for claim in claims]
        embeddings = await self._embedding.embed_batch(texts, RETRIEVAL_MODEL)
        indexed = 0
        for claim, embedding in zip(claims, embeddings, strict=True):
            if await self._repo.store_embedding(claim.claim_id, embedding):
                indexed += 1
        if indexed < len(claims):
            # Stores can fail only if a claim vanished between fetch and write
            # (delete race). Log it so a `--all` backfill that stops on a
            # zero-progress batch is diagnosable rather than silent.
            logger.warning(
                "claim embedding store incomplete: %d/%d persisted", indexed, len(claims)
            )
        return indexed

    async def index_claim(self, claim_id: str) -> bool:
        """Embed and store the vector for a single claim by ID."""
        claim = await self._claims.get_claim(claim_id)
        if claim is None:
            return False
        embedding = await self._embedding.embed(claim_embedding_text(claim), RETRIEVAL_MODEL)
        return await self._repo.store_embedding(claim_id, embedding)
