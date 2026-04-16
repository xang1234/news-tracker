-- Migration 030: Reconcile legacy embedding schema without dropping data.
--
-- Legacy databases may still have documents.embedding as vector(384) from the
-- pre-FinBERT schema. Preserve that data in embedding_legacy_384, create the
-- runtime-required vector(768) column, and ensure the HNSW index points at the
-- new column.

CREATE EXTENSION IF NOT EXISTS vector;

DROP INDEX IF EXISTS idx_documents_embedding_hnsw;

DO $$
DECLARE
    embedding_type TEXT;
BEGIN
    SELECT format_type(attribute.atttypid, attribute.atttypmod)
    INTO embedding_type
    FROM pg_attribute AS attribute
    JOIN pg_class AS table_class ON table_class.oid = attribute.attrelid
    JOIN pg_namespace AS namespace ON namespace.oid = table_class.relnamespace
    WHERE namespace.nspname = 'public'
      AND table_class.relname = 'documents'
      AND attribute.attname = 'embedding'
      AND attribute.attnum > 0
      AND NOT attribute.attisdropped;

    IF embedding_type IS NULL THEN
        ALTER TABLE documents ADD COLUMN embedding vector(768);
    ELSIF embedding_type = 'vector(768)' THEN
        NULL;
    ELSIF embedding_type = 'vector(384)' THEN
        IF EXISTS (
            SELECT 1
            FROM pg_attribute AS attribute
            JOIN pg_class AS table_class ON table_class.oid = attribute.attrelid
            JOIN pg_namespace AS namespace ON namespace.oid = table_class.relnamespace
            WHERE namespace.nspname = 'public'
              AND table_class.relname = 'documents'
              AND attribute.attname = 'embedding_legacy_384'
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
        ) THEN
            IF EXISTS (
                SELECT 1
                FROM pg_attribute AS attribute
                JOIN pg_class AS table_class ON table_class.oid = attribute.attrelid
                JOIN pg_namespace AS namespace ON namespace.oid = table_class.relnamespace
                WHERE namespace.nspname = 'public'
                  AND table_class.relname = 'documents'
                  AND attribute.attname = 'embedding_conflict_384'
                  AND attribute.attnum > 0
                  AND NOT attribute.attisdropped
            ) THEN
                RAISE EXCEPTION
                    'Cannot preserve documents.embedding: embedding_legacy_384 and embedding_conflict_384 already exist';
            END IF;

            ALTER TABLE documents RENAME COLUMN embedding TO embedding_conflict_384;
        ELSE
            ALTER TABLE documents RENAME COLUMN embedding TO embedding_legacy_384;
        END IF;

        ALTER TABLE documents ADD COLUMN embedding vector(768);
    ELSE
        RAISE EXCEPTION 'Unsupported documents.embedding type: %', embedding_type;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
