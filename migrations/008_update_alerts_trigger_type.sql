-- Migration 008: Add propagated_impact trigger type to alerts
-- Extends the trigger_type CHECK constraint to support sentiment propagation alerts.

ALTER TABLE alerts DROP CONSTRAINT IF EXISTS alerts_trigger_type_check;

ALTER TABLE alerts ADD CONSTRAINT alerts_trigger_type_check
    CHECK (trigger_type IN (
        'sentiment_velocity', 'extreme_sentiment',
        'volume_surge', 'lifecycle_change', 'new_theme',
        'propagated_impact'
    ));
