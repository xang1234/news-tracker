-- Migration 001: Initial schema
-- Creates the core tables for document storage

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Main documents table
CREATE TABLE IF NOT EXISTS documents (
    -- Identity
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    url TEXT,

    -- Timestamps
    timestamp TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Author info
    author_id TEXT NOT NULL,
    author_name TEXT NOT NULL,
    author_followers INTEGER,
    author_verified BOOLEAN DEFAULT FALSE,

    -- Content
    content TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'post',
    title TEXT,

    -- Engagement metrics (JSONB for flexibility)
    engagement JSONB NOT NULL DEFAULT '{}',

    -- Extracted entities
    tickers TEXT[] NOT NULL DEFAULT '{}',
    urls_mentioned TEXT[] NOT NULL DEFAULT '{}',

    -- Quality signals
    spam_score REAL NOT NULL DEFAULT 0.0 CHECK (spam_score >= 0 AND spam_score <= 1),
    bot_probability REAL NOT NULL DEFAULT 0.0 CHECK (bot_probability >= 0 AND bot_probability <= 1),

    -- Downstream enrichment
    embedding vector(384),  -- For sentence-transformers/all-MiniLM-L6-v2
    sentiment JSONB,
    theme_ids TEXT[] NOT NULL DEFAULT '{}',

    -- Original data for debugging
    raw_data JSONB
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_documents_platform ON documents(platform);
CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_documents_author_id ON documents(author_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

-- GIN index for ticker array queries
CREATE INDEX IF NOT EXISTS idx_documents_tickers ON documents USING GIN(tickers);

-- Full-text search index on content
CREATE INDEX IF NOT EXISTS idx_documents_content_search
    ON documents USING GIN(to_tsvector('english', content));

-- Partial index for high-quality documents
CREATE INDEX IF NOT EXISTS idx_documents_quality
    ON documents(platform, timestamp DESC)
    WHERE spam_score < 0.5;

-- Processing metrics table (for observability)
CREATE TABLE IF NOT EXISTS processing_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    dimensions JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON processing_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON processing_metrics(metric_name);

-- Function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at on documents
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
