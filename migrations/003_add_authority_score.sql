-- Migration: Add authority_score column for quality-based filtering
-- This column stores a computed authority score (0.0-1.0) based on:
-- - Author verification status (+0.2)
-- - Follower count (log-scaled, max +0.3)
-- - Engagement metrics (log-scaled, max +0.3)
-- - Inverse spam score (+0.2 * (1 - spam_score))

-- Add authority_score column
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS authority_score REAL;

-- Add index for filtering by authority score
CREATE INDEX IF NOT EXISTS idx_documents_authority_score
    ON documents(authority_score DESC)
    WHERE authority_score IS NOT NULL;

-- Composite index for common search pattern (platform + authority)
CREATE INDEX IF NOT EXISTS idx_documents_platform_authority
    ON documents(platform, authority_score DESC)
    WHERE authority_score IS NOT NULL AND embedding IS NOT NULL;

-- Backfill existing documents with a basic authority score
-- This uses a simplified calculation that can be refined later
UPDATE documents
SET authority_score = LEAST(1.0, GREATEST(0.0,
    -- Verified bonus: +0.2
    CASE WHEN author_verified THEN 0.2 ELSE 0.0 END
    -- Follower score: log-scaled, max +0.3
    + CASE
        WHEN author_followers IS NULL THEN 0.0
        WHEN author_followers <= 0 THEN 0.0
        ELSE LEAST(0.3, 0.3 * LN(author_followers + 1) / LN(1000001))
      END
    -- Engagement score from JSONB: log-scaled, max +0.3
    + LEAST(0.3, 0.3 * LN(
        COALESCE((engagement->>'likes')::int, 0)
        + COALESCE((engagement->>'shares')::int, 0) * 2
        + COALESCE((engagement->>'comments')::int, 0)
        + 1
    ) / LN(10001))
    -- Inverse spam: +0.2 * (1 - spam_score)
    + 0.2 * (1.0 - COALESCE(spam_score, 0.0))
))
WHERE authority_score IS NULL;
