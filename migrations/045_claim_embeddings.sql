-- Migration 045: Retrieval embeddings on evidence claims
--
-- Backs the claim/assertion retrieval substrate (issue news-tracker-q7g.1):
-- a semantic index over the STRUCTURED fact layer so RAG briefings / Q&A
-- retrieve verified, deduplicated, confidence-scored claims rather than raw
-- documents.
--
--   embedding   — MiniLM 384-dim vector of the composed claim text
--                 (subject + humanized predicate + object + numeric fact).
--                 MiniLM (not FinBERT) because claims are short factual
--                 sentences and query+claims must share one model to be
--                 comparable. Switching models requires changing this dim.
--   embedded_at — when the embedding was last (re)generated; lets a backfill
--                 detect stale vectors after a claim's text changes.
--
-- Both columns are nullable: claims are embedded asynchronously, so a NULL
-- embedding simply means "not yet indexed" (ClaimRetrievalRepository
-- .list_unembedded picks these up).

ALTER TABLE news_intel.evidence_claims
    ADD COLUMN IF NOT EXISTS embedding   vector(384),
    ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMPTZ;

-- HNSW index for cosine-similarity retrieval, matching the documents
-- embedding pattern (m = 16, ef_construction = 64). Rows with a NULL
-- embedding are excluded from similarity results automatically.
CREATE INDEX IF NOT EXISTS idx_claims_embedding_hnsw
    ON news_intel.evidence_claims
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMENT ON COLUMN news_intel.evidence_claims.embedding IS
    'MiniLM 384-dim embedding of the composed claim text for semantic retrieval';
