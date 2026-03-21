-- Migration 017: Add narrative trigger types to alerts

ALTER TABLE alerts DROP CONSTRAINT IF EXISTS alerts_trigger_type_check;

ALTER TABLE alerts ADD CONSTRAINT alerts_trigger_type_check
    CHECK (trigger_type IN (
        'sentiment_velocity', 'extreme_sentiment',
        'volume_surge', 'lifecycle_change', 'new_theme',
        'propagated_impact', 'narrative_surge',
        'cross_platform_breakout', 'authority_divergence',
        'sentiment_regime_shift'
    ));
