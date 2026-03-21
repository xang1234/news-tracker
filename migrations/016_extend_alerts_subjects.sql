-- Migration 016: Add generic alert subjects and conviction score

ALTER TABLE alerts ADD COLUMN IF NOT EXISTS subject_type TEXT;
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS subject_id TEXT;
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS conviction_score REAL;

UPDATE alerts
SET subject_type = 'theme',
    subject_id = theme_id
WHERE subject_type IS NULL OR subject_id IS NULL;

ALTER TABLE alerts ALTER COLUMN subject_type SET DEFAULT 'theme';
ALTER TABLE alerts ALTER COLUMN subject_type SET NOT NULL;
ALTER TABLE alerts ALTER COLUMN subject_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_alerts_subject_created_at
    ON alerts(subject_type, subject_id, created_at DESC);
