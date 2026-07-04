-- Migration 047: deactivate sources pruned from the default catalog
--
-- PR #32 removes broad tech-press RSS feeds and noisy social/community
-- defaults from the code-owned seed lists. Existing databases already seeded
-- with those rows need an explicit state reconciliation so ingestion no
-- longer treats them as active when DB-backed sources are enabled.

UPDATE sources
SET is_active = FALSE,
    updated_at = NOW()
WHERE is_active = TRUE
  AND (platform, identifier) IN (
      VALUES
          ('rss', 'ars-technica'),
          ('rss', 'techcrunch'),
          ('rss', 'the-verge'),
          ('rss', 'toms-hardware'),
          ('twitter', 'AMD'),
          ('twitter', 'Broadcom'),
          ('twitter', 'DeItaone'),
          ('twitter', 'MicronTech'),
          ('twitter', 'Qualcomm'),
          ('twitter', 'SKhynix'),
          ('twitter', 'Samsung_SD'),
          ('twitter', 'StockMKTNewz'),
          ('twitter', 'TechAltar'),
          ('twitter', 'intel'),
          ('twitter', 'nvidia'),
          ('twitter', 'unusual_whales'),
          ('reddit', 'AMD_Stock'),
          ('reddit', 'intel'),
          ('reddit', 'investing'),
          ('reddit', 'nvidia'),
          ('reddit', 'options'),
          ('reddit', 'stockmarket'),
          ('reddit', 'stocks'),
          ('reddit', 'wallstreetbets')
  );
