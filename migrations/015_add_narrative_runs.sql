-- Migration 015: Add narrative run tables
-- Narrative runs are ephemeral subclusters that live under durable themes.

CREATE TABLE IF NOT EXISTS narrative_runs (
    run_id TEXT PRIMARY KEY,
    theme_id TEXT NOT NULL REFERENCES themes(theme_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'cooling', 'closed')),
    centroid vector(768) NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMPTZ NOT NULL,
    last_document_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    doc_count INTEGER NOT NULL DEFAULT 0,
    platform_first_seen JSONB NOT NULL DEFAULT '{}',
    ticker_counts JSONB NOT NULL DEFAULT '{}',
    avg_sentiment REAL NOT NULL DEFAULT 0.0,
    avg_authority REAL NOT NULL DEFAULT 0.0,
    platform_count INTEGER NOT NULL DEFAULT 0,
    current_rate_per_hour REAL NOT NULL DEFAULT 0.0,
    current_acceleration REAL NOT NULL DEFAULT 0.0,
    conviction_score REAL NOT NULL DEFAULT 0.0,
    last_signal_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_narrative_runs_theme_status_updated
    ON narrative_runs (theme_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_narrative_runs_status_updated
    ON narrative_runs (status, updated_at DESC);

DROP TRIGGER IF EXISTS update_narrative_runs_updated_at ON narrative_runs;
CREATE TRIGGER update_narrative_runs_updated_at
    BEFORE UPDATE ON narrative_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE IF NOT EXISTS narrative_run_documents (
    run_id TEXT NOT NULL REFERENCES narrative_runs(run_id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    theme_id TEXT NOT NULL REFERENCES themes(theme_id) ON DELETE CASCADE,
    similarity REAL NOT NULL,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_narrative_run_documents_theme_assigned
    ON narrative_run_documents (theme_id, assigned_at DESC);

CREATE INDEX IF NOT EXISTS idx_narrative_run_documents_doc
    ON narrative_run_documents (document_id);

CREATE TABLE IF NOT EXISTS narrative_run_buckets (
    run_id TEXT NOT NULL REFERENCES narrative_runs(run_id) ON DELETE CASCADE,
    bucket_start TIMESTAMPTZ NOT NULL,
    doc_count INTEGER NOT NULL DEFAULT 0,
    platform_counts JSONB NOT NULL DEFAULT '{}',
    ticker_counts JSONB NOT NULL DEFAULT '{}',
    sentiment_sum REAL NOT NULL DEFAULT 0.0,
    sentiment_weight REAL NOT NULL DEFAULT 0.0,
    sentiment_confidence_sum REAL NOT NULL DEFAULT 0.0,
    sentiment_doc_count INTEGER NOT NULL DEFAULT 0,
    authority_sum REAL NOT NULL DEFAULT 0.0,
    high_authority_sentiment_sum REAL NOT NULL DEFAULT 0.0,
    high_authority_weight REAL NOT NULL DEFAULT 0.0,
    high_authority_doc_count INTEGER NOT NULL DEFAULT 0,
    low_authority_sentiment_sum REAL NOT NULL DEFAULT 0.0,
    low_authority_weight REAL NOT NULL DEFAULT 0.0,
    low_authority_doc_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, bucket_start)
);

CREATE INDEX IF NOT EXISTS idx_narrative_run_buckets_recent
    ON narrative_run_buckets (bucket_start DESC);

DROP TRIGGER IF EXISTS update_narrative_run_buckets_updated_at ON narrative_run_buckets;
CREATE TRIGGER update_narrative_run_buckets_updated_at
    BEFORE UPDATE ON narrative_run_buckets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE IF NOT EXISTS narrative_signal_state (
    run_id TEXT NOT NULL REFERENCES narrative_runs(run_id) ON DELETE CASCADE,
    trigger_type TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'inactive'
        CHECK (state IN ('inactive', 'active')),
    last_score REAL NOT NULL DEFAULT 0.0,
    last_alert_at TIMESTAMPTZ,
    last_transition_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cooldown_until TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, trigger_type)
);
