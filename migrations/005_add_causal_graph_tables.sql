-- Migration 005: Add causal graph tables
-- Provides graph infrastructure for modeling semiconductor supply chain
-- relationships (supplier→customer, competitor, technology driver/blocker).
-- Supports recursive CTE traversal for upstream/downstream impact analysis.

-- ============================================================
-- Table: causal_nodes
-- Entities in the causal graph (tickers, themes, technologies)
-- ============================================================
CREATE TABLE IF NOT EXISTS causal_nodes (
    node_id     TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL
        CHECK (node_type IN ('ticker', 'theme', 'technology')),
    name        TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Filter by type (e.g., get all tickers)
CREATE INDEX IF NOT EXISTS idx_causal_nodes_type
    ON causal_nodes(node_type);

-- Recency queries
CREATE INDEX IF NOT EXISTS idx_causal_nodes_updated_at
    ON causal_nodes(updated_at DESC);

-- Reuse the update_updated_at_column() trigger function from 001
DROP TRIGGER IF EXISTS update_causal_nodes_updated_at ON causal_nodes;
CREATE TRIGGER update_causal_nodes_updated_at
    BEFORE UPDATE ON causal_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Table: causal_edges
-- Directed relationships between nodes
-- ============================================================
CREATE TABLE IF NOT EXISTS causal_edges (
    source      TEXT NOT NULL REFERENCES causal_nodes(node_id) ON DELETE CASCADE,
    target      TEXT NOT NULL REFERENCES causal_nodes(node_id) ON DELETE CASCADE,
    relation    TEXT NOT NULL
        CHECK (relation IN ('depends_on', 'supplies_to', 'competes_with', 'drives', 'blocks')),
    confidence  REAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    source_doc_ids TEXT[] NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata    JSONB NOT NULL DEFAULT '{}',

    PRIMARY KEY (source, target, relation)
);

-- Traverse outgoing edges from a node (downstream queries)
CREATE INDEX IF NOT EXISTS idx_causal_edges_source
    ON causal_edges(source);

-- Traverse incoming edges to a node (upstream queries)
CREATE INDEX IF NOT EXISTS idx_causal_edges_target
    ON causal_edges(target);
