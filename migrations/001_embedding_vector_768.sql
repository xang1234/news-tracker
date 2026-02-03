-- Migration: Update embedding column to 768 dimensions for FinBERT
-- This migration updates the documents table to use 768-dimensional vectors
-- and adds an HNSW index for efficient similarity search.
--
-- IMPORTANT: Run this migration AFTER backing up your data if you have existing embeddings.
-- Existing 384-dim embeddings will need to be regenerated.

-- Step 1: Drop existing embedding column and index
DROP INDEX IF EXISTS idx_documents_embedding_hnsw;
ALTER TABLE documents DROP COLUMN IF EXISTS embedding;

-- Step 2: Add new 768-dimensional embedding column
ALTER TABLE documents ADD COLUMN embedding vector(768);

-- Step 3: Create HNSW index for efficient cosine similarity search
-- Parameters:
--   m = 16: Number of bidirectional links per node (default: 16)
--   ef_construction = 64: Size of dynamic candidate list during build (default: 64)
CREATE INDEX idx_documents_embedding_hnsw
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Step 4: Verify the migration
-- SELECT column_name, data_type, udt_name
-- FROM information_schema.columns
-- WHERE table_name = 'documents' AND column_name = 'embedding';

-- To backfill embeddings for existing documents, run:
-- SELECT id FROM documents WHERE embedding IS NULL;
-- Then process these IDs through the EmbeddingWorker.run_once() method
