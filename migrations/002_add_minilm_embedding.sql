-- Migration: Add MiniLM embedding support for lightweight Twitter/short content
-- This migration adds a 384-dimensional embedding column for MiniLM model
-- alongside the existing 768-dimensional FinBERT embedding column.
--
-- Model Selection Strategy:
--   - Twitter posts < 300 chars → MiniLM (384-dim)
--   - All other content → FinBERT (768-dim)
--
-- IMPORTANT: Both columns are optional (NULL allowed) to support gradual backfilling.

-- Step 1: Add MiniLM embedding column
ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_minilm vector(384);

-- Step 2: Create HNSW index for efficient cosine similarity search on MiniLM embeddings
-- Parameters optimized for smaller vectors:
--   m = 16: Number of bidirectional links per node
--   ef_construction = 64: Size of dynamic candidate list during build
CREATE INDEX IF NOT EXISTS idx_documents_embedding_minilm_hnsw
    ON documents
    USING hnsw (embedding_minilm vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Step 3: Add comment for documentation
COMMENT ON COLUMN documents.embedding_minilm IS
    'MiniLM 384-dimensional embedding for Twitter posts and short content (<300 chars)';

-- Step 4: Verify the migration
-- Run this query to confirm the new column exists:
-- SELECT column_name, data_type, udt_name
-- FROM information_schema.columns
-- WHERE table_name = 'documents' AND column_name = 'embedding_minilm';

-- Expected output:
--  column_name      | data_type    | udt_name
-- ------------------+--------------+----------
--  embedding_minilm | USER-DEFINED | vector

-- Step 5: Backfill guidance
-- To backfill MiniLM embeddings for existing Twitter posts:
-- SELECT id FROM documents
-- WHERE platform = 'twitter'
--   AND LENGTH(content) < 300
--   AND embedding_minilm IS NULL
-- ORDER BY timestamp DESC
-- LIMIT 1000;
--
-- Then process these IDs through EmbeddingWorker with model='minilm'
