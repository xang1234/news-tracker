-- Migration 006: Add alerts table
-- Persists actionable alerts generated from theme metrics analysis:
-- sentiment velocity changes, extreme sentiment, volume surges,
-- lifecycle transitions, and new theme emergence.

CREATE TABLE IF NOT EXISTS alerts (
    -- Identity
    alert_id        TEXT PRIMARY KEY,
    theme_id        TEXT NOT NULL REFERENCES themes(theme_id) ON DELETE CASCADE,

    -- Classification
    trigger_type    TEXT NOT NULL
        CHECK (trigger_type IN (
            'sentiment_velocity', 'extreme_sentiment',
            'volume_surge', 'lifecycle_change', 'new_theme'
        )),
    severity        TEXT NOT NULL
        CHECK (severity IN ('critical', 'warning', 'info')),

    -- Content
    title           TEXT NOT NULL,
    message         TEXT NOT NULL,
    trigger_data    JSONB NOT NULL DEFAULT '{}',

    -- State
    acknowledged    BOOLEAN NOT NULL DEFAULT FALSE,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rate limit counting: count alerts by severity within a day
CREATE INDEX IF NOT EXISTS idx_alerts_severity_created_at
    ON alerts(severity, created_at);

-- Theme alert history
CREATE INDEX IF NOT EXISTS idx_alerts_theme_created_at
    ON alerts(theme_id, created_at);

-- Recent alerts listing (default sort order)
CREATE INDEX IF NOT EXISTS idx_alerts_created_at_desc
    ON alerts(created_at DESC);
