"""pgvector wire-format helpers.

A leaf module (no intra-``src`` imports) so every layer that stores or
queries embeddings — documents (``storage``), the vector store
(``vectorstore``), and claim retrieval (``retrieval``) — can share one
formatter without import cycles.
"""

from __future__ import annotations


def to_pgvector_literal(vector: list[float]) -> str:
    """Render an embedding as a pgvector text literal: ``[a,b,c]``.

    asyncpg sends this as text and Postgres casts it to ``vector`` on
    assignment/comparison. Canonical formatter so the wire format lives in
    one place rather than being re-spelled at every call site.
    """
    return f"[{','.join(str(x) for x in vector)}]"
